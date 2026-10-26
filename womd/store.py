"""Turns one Waymo Scenario proto into staged arrays of agent tracks, map dots
one metre apart and traffic-light histories, all in the scene frame.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

from womd import formats, frames, layout
from womd.formats import StagedFile
from womd.frames import Frame
from womd.layout import SIGNAL_HISTORY_SHAPE
from womd_protos import map_pb2, scenario_pb2

# Map feature kinds whose points form a closed polygon rather
# than an open polyline.
MAP_POLYGON_KINDS = ("crosswalk", "speed_bump", "driveway")

Track = scenario_pb2.Track
MapFeature = map_pb2.MapFeature


class SceneFeature(NamedTuple):
    points: np.ndarray  # (D, 2)
    directions: np.ndarray  # (D, 2) unit vectors
    crossing_codes: np.ndarray  # (D, 2) left and right boundary codes
    kind_index: int


class StagedMap(NamedTuple):
    """The map features of a scenario, all dots in one array and then one
    entry per feature.
    """

    map_rows: np.ndarray  # (D, 32)
    feature_lengths: np.ndarray  # (F,) dots per feature
    feature_ids: np.ndarray  # (F,)
    polyline_signal_histories: np.ndarray  # (F, 11, 9)
    feature_is_interpolating: np.ndarray  # (F,)


class TrafficSignals(NamedTuple):
    """Per signalled lane, its one-hot state history over the history steps
    and its stop point in world coordinates.
    """

    histories: dict[int, np.ndarray]  # lane id to (11, 9)
    stop_points: dict[int, np.ndarray]  # lane id to (2,)


def empty_staged_map() -> StagedMap:
    return StagedMap(
        map_rows=np.zeros((0, layout.MAP_FEATURE_DIM)),
        feature_lengths=np.zeros(0, dtype=np.int64),
        feature_ids=np.zeros(0, dtype=np.int64),
        polyline_signal_histories=np.zeros((0,) + SIGNAL_HISTORY_SHAPE),
        feature_is_interpolating=np.zeros(0, dtype=bool),
    )


def scene_frame(scenario: scenario_pb2.Scenario) -> Frame:
    """The scene frame, the self-driving car's world position and heading at
    the current step.
    """
    sdc_track = scenario.tracks[scenario.sdc_track_index]
    sdc_state = sdc_track.states[layout.CURRENT_STEP_INDEX]
    if not sdc_state.valid:
        raise ValueError(f"invalid SDC at current step, {scenario.scenario_id}")
    origin = np.array([sdc_state.center_x, sdc_state.center_y])
    return Frame(origin, sdc_state.heading)


def type_one_hot(track: Track) -> np.ndarray:
    """The track's object type as a one-hot over AGENT_TYPES, with every type
    the model does not predict counted as TYPE_OTHER.
    """
    type_name = Track.ObjectType.Name(track.object_type)
    if type_name not in layout.AGENT_TYPES:
        type_name = "TYPE_OTHER"
    one_hot = np.zeros(layout.NUM_AGENT_TYPES)
    one_hot[layout.AGENT_TYPES.index(type_name)] = 1.0
    return one_hot


def track_rows(track: Track, frame: Frame, is_sdc: bool) -> np.ndarray:
    """One agent row per step for a track, in the scene frame; the row layout
    is drawn in womd/layout.py.
    """
    states = track.states
    positions = np.array([[state.center_x, state.center_y] for state in states])
    headings = np.array([state.heading for state in states])
    velocities = np.array(
        [[state.velocity_x, state.velocity_y] for state in states]
    )
    dimensions = np.array([[state.length, state.width] for state in states])

    scene_headings = frames.headings_to_frame(headings, frame.heading)
    rows = np.zeros((layout.TOTAL_STEPS, layout.AGENT_FEATURE_DIM))
    rows[:, layout.AGENT_POSITION] = frames.positions_to_frame(positions, frame)
    rows[:, layout.AGENT_HEADING_COSINE] = np.cos(scene_headings)
    rows[:, layout.AGENT_HEADING_SINE] = np.sin(scene_headings)
    rows[:, layout.AGENT_VELOCITY] = frames.directions_to_frame(
        velocities, frame.heading
    )
    rows[:, layout.AGENT_DIMENSIONS] = dimensions
    rows[:, layout.AGENT_TYPE] = type_one_hot(track)
    rows[:, layout.AGENT_IS_SDC] = float(is_sdc)
    return rows


def track_valid(track: Track) -> np.ndarray:
    return np.array([state.valid for state in track.states])


def feature_points(feature: MapFeature) -> np.ndarray | None:
    """One map feature's raw world-frame points (N, 2), or None if the
    feature has no usable geometry.
    """
    kind_name = feature.WhichOneof("feature_data")
    if kind_name is None:
        return None
    if kind_name == "stop_sign":
        raw_points = [feature.stop_sign.position]
    elif kind_name in MAP_POLYGON_KINDS:
        corners = getattr(feature, kind_name).polygon
        if len(corners) < 2:
            return None
        # The polygon is closed by repeating its first corner.
        raw_points = list(corners) + [corners[0]]
    else:
        raw_points = getattr(feature, kind_name).polyline
    if len(raw_points) == 0:
        return None
    return np.array([[point.x, point.y] for point in raw_points])


def arc_lengths(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length along a polyline, starting at 0."""
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def sample_distances(lengths: np.ndarray) -> np.ndarray:
    """Arc lengths to resample at, one per metre plus the final endpoint."""
    spacing = layout.MAP_POINT_SPACING_METRES
    return np.append(np.arange(0.0, lengths[-1], spacing), lengths[-1])


def resample_polyline(points: np.ndarray) -> np.ndarray:
    """The polyline's points one metre apart along its length, by linear
    interpolation of x and y against arc length.
    """
    if len(points) < 2:
        return points
    lengths = arc_lengths(points)
    if lengths[-1] == 0.0:
        return points[:1]
    distances = sample_distances(lengths)
    resampled_x = np.interp(distances, lengths, points[:, 0])
    resampled_y = np.interp(distances, lengths, points[:, 1])
    return np.stack([resampled_x, resampled_y], axis=1)


def polyline_directions(points: np.ndarray) -> np.ndarray:
    """Unit direction per point along a polyline; the last point repeats the
    final segment's direction and a zero-length step repeats the one before.
    """
    if len(points) < 2:
        return np.zeros_like(points)
    steps = np.diff(points, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1, keepdims=True)
    zero_steps = np.flatnonzero(step_lengths[:, 0] == 0.0)
    step_lengths[zero_steps] = 1.0
    directions = steps / step_lengths
    for zero_step in zero_steps:
        directions[zero_step] = directions[zero_step - 1]
    return np.concatenate([directions, directions[-1:]])


def crossing_codes(feature: MapFeature, lengths: np.ndarray) -> np.ndarray:
    """Per-dot left and right boundary crossing codes for a lane feature, 0 for
    no boundary and otherwise 1 + the road line type. The raw points' arc
    lengths place each boundary segment on the resampled dots.
    """
    distances = sample_distances(lengths)
    codes = np.zeros((len(distances), 2))
    if feature.WhichOneof("feature_data") != "lane":
        return codes
    for side_index, side_name in enumerate(layout.LANE_SIDES):
        for segment in getattr(feature.lane, f"{side_name}_boundaries"):
            if segment.boundary_type >= len(layout.ROAD_LINE_TYPES):
                raise ValueError(
                    f"lane {feature.id}: unknown boundary type"
                    f" {segment.boundary_type}"
                )
            start_length = lengths[segment.lane_start_index]
            end_length = lengths[segment.lane_end_index]
            first_dot = np.searchsorted(distances, start_length, side="left")
            last_dot = np.searchsorted(distances, end_length, side="right")
            codes[first_dot:last_dot, side_index] = 1.0 + segment.boundary_type
    return codes


def feature_in_scene(feature: MapFeature, frame: Frame) -> SceneFeature | None:
    """Resamples a map feature to one-metre dots, derives directions and
    crossing codes, and keeps the dots inside the staging radius.
    """
    points = feature_points(feature)
    if points is None:
        return None
    lengths = arc_lengths(points)
    spaced = resample_polyline(points)
    directions = polyline_directions(spaced)
    codes = crossing_codes(feature, lengths)
    scene_points = frames.positions_to_frame(spaced, frame)
    scene_directions = frames.directions_to_frame(directions, frame.heading)
    inside = (
        np.linalg.norm(scene_points, axis=1)
        <= layout.STAGING_CROP_RADIUS_METRES
    )
    return SceneFeature(
        scene_points[inside],
        scene_directions[inside],
        codes[inside],
        layout.MAP_POLYLINE_KINDS.index(feature.WhichOneof("feature_data")),
    )


def traffic_signals(scenario: scenario_pb2.Scenario) -> TrafficSignals:
    histories: dict[int, np.ndarray] = {}
    stop_points: dict[int, np.ndarray] = {}
    history_states = scenario.dynamic_map_states[: layout.HISTORY_STEPS]
    for step_index, dynamic_state in enumerate(history_states):
        for lane_state in dynamic_state.lane_states:
            state_name = map_pb2.TrafficSignalLaneState.State.Name(
                lane_state.state
            )
            state_index = layout.TRAFFIC_SIGNAL_STATES.index(state_name)
            history = histories.setdefault(
                lane_state.lane, np.zeros(SIGNAL_HISTORY_SHAPE)
            )
            history[step_index, state_index] = 1.0
            stop_point = np.array(
                [lane_state.stop_point.x, lane_state.stop_point.y]
            )
            stop_points.setdefault(lane_state.lane, stop_point)
    return TrafficSignals(histories, stop_points)


def is_interpolating(feature: MapFeature) -> bool:
    is_lane = feature.WhichOneof("feature_data") == "lane"
    return is_lane and feature.lane.interpolating


class MapStager:
    """Turns map features into staged rows, in one scene frame and with that
    scenario's traffic signals.
    """

    def __init__(self, frame: Frame, signals: TrafficSignals) -> None:
        self.frame = frame
        self.signals = signals

    def rows(self, feature: MapFeature) -> np.ndarray | None:
        """One map row per dot of the feature, with the columns of its kind
        filled; the row layout is drawn in womd/layout.py.
        """
        scene_feature = feature_in_scene(feature, self.frame)
        if scene_feature is None or len(scene_feature.points) == 0:
            return None
        kind_name = layout.MAP_POLYLINE_KINDS[scene_feature.kind_index]
        rows = np.zeros((len(scene_feature.points), layout.MAP_FEATURE_DIM))
        rows[:, layout.MAP_POSITION] = scene_feature.points
        rows[:, layout.MAP_DIRECTION] = scene_feature.directions
        rows[:, layout.MAP_KIND.start + scene_feature.kind_index] = 1.0
        rows[:, layout.MAP_LEFT_BOUNDARY_CROSSING] = (
            scene_feature.crossing_codes[:, 0]
        )
        rows[:, layout.MAP_RIGHT_BOUNDARY_CROSSING] = (
            scene_feature.crossing_codes[:, 1]
        )

        if kind_name == "lane":
            if feature.id in self.signals.stop_points:
                stop_point = self.signals.stop_points[feature.id]
                rows[:, layout.MAP_STOP_POINT] = frames.positions_to_frame(
                    stop_point, self.frame
                )
            rows[:, layout.MAP_LANE_TYPE.start + feature.lane.type] = 1.0
            rows[:, layout.MAP_SPEED_LIMIT] = feature.lane.speed_limit_mph
        elif kind_name == "road_line":
            rows[:, layout.MAP_BOUNDARY_TYPE.start + feature.road_line.type] = (
                1.0
            )
        elif kind_name == "road_edge":
            first_edge_column = layout.MAP_BOUNDARY_TYPE.start + len(
                layout.ROAD_LINE_TYPES
            )
            rows[:, first_edge_column + feature.road_edge.type] = 1.0
        return rows

    def signal_history(self, feature: MapFeature) -> np.ndarray:
        """A lane feature's signal history, or all zeros if it is not a lane
        or carries no signal.
        """
        is_lane = feature.WhichOneof("feature_data") == "lane"
        if is_lane and feature.id in self.signals.histories:
            return self.signals.histories[feature.id]
        return np.zeros(SIGNAL_HISTORY_SHAPE)

    def staged(self, features: Iterable[MapFeature]) -> StagedMap:
        """Every feature with usable geometry, as one row array plus the
        per-feature entries.
        """
        rows_per_feature = []
        kept = []
        for feature in features:
            rows = self.rows(feature)
            if rows is None:
                continue
            rows_per_feature.append(rows)
            kept.append(feature)
        if not kept:
            return empty_staged_map()
        histories = [self.signal_history(feature) for feature in kept]
        lengths = [len(rows) for rows in rows_per_feature]
        return StagedMap(
            map_rows=np.concatenate(rows_per_feature),
            feature_lengths=np.array(lengths, dtype=np.int64),
            feature_ids=np.array(
                [feature.id for feature in kept], dtype=np.int64
            ),
            polyline_signal_histories=np.stack(histories),
            feature_is_interpolating=np.array(
                [is_interpolating(feature) for feature in kept]
            ),
        )


def worst_spacing_deviation(scenario: scenario_pb2.Scenario) -> float:
    """How far the logged timestamps stray from 10 Hz, in seconds."""
    gaps = np.diff(np.array(scenario.timestamps_seconds))
    if len(gaps) == 0:
        return 0.0
    return float(np.max(np.abs(gaps - layout.TIMESTEP_SECONDS)))


def staged_file(scenario: scenario_pb2.Scenario) -> StagedFile:
    if scenario.current_time_index != layout.CURRENT_STEP_INDEX:
        raise ValueError(
            f"current_time_index {scenario.current_time_index},"
            f" {scenario.scenario_id}"
        )
    frame = scene_frame(scenario)
    tracks = scenario.tracks
    sdc_index = scenario.sdc_track_index
    designated = {
        required.track_index for required in scenario.tracks_to_predict
    }
    interest_ids = set(scenario.objects_of_interest)
    rows = []
    for index, track in enumerate(tracks):
        rows.append(track_rows(track, frame, is_sdc=index == sdc_index))
    stager = MapStager(frame, traffic_signals(scenario))
    scene_map = stager.staged(scenario.map_features)
    histories = scene_map.polyline_signal_histories
    return StagedFile(
        track_rows=np.stack(rows).astype(np.float32),
        track_valid=np.stack([track_valid(track) for track in tracks]),
        track_ids=np.array([track.id for track in tracks], dtype=np.int64),
        is_designated_target=np.array(
            [i in designated for i in range(len(tracks))]
        ),
        is_object_of_interest=np.array(
            [track.id in interest_ids for track in tracks]
        ),
        map_rows=scene_map.map_rows.astype(np.float32),
        feature_lengths=scene_map.feature_lengths,
        feature_ids=scene_map.feature_ids,
        feature_is_interpolating=scene_map.feature_is_interpolating,
        polyline_signal_histories=histories.astype(np.float32),
        frame_origin=frame.origin.astype(np.float32),
        frame_heading=np.float32(frame.heading),
        scenario_id=scenario.scenario_id,
        provenance=formats.artifact_provenance(
            "stage.py", scenario.scenario_id
        ),
    )
