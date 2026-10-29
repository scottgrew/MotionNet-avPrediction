"""One test per correctness property of the frames, staging, loader, model,
loss, pruning, metrics and submission.
"""

import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import reference_implementations
import torch

import fit_anchors
import train
from womd import (
    baseline,
    checkpoint,
    formats,
    frames,
    layout,
    loss,
    metrics,
    model,
    pipeline,
    pruning,
    store,
    views,
)
from womd.frames import Frame
from womd.model import ModePredictions
from womd.pipeline import BatchSettings
from womd.views import TargetInputs
from womd_protos import scenario_pb2

STAGED_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "staged"
RUNNER_PATH = Path(__file__).parent.parent / "container" / "runner.py"
TWO_STAGED_SCENARIOS = BatchSettings(
    batch_size=2,
    worker_count=0,
    prefetch_batches=0,
    seed=0,
    designated_only=True,
)


class SyntheticSize(NamedTuple):
    """How large a synthetic scene batch is."""

    samples: int
    agents: int
    polylines: int
    dots_per_polyline: int


def lane_polyline_rows(first_dot_x, dot_count):
    """Map rows for a straight lane along x with one dot per metre."""
    rows = np.zeros((dot_count, layout.MAP_FEATURE_DIM), dtype=np.float32)
    along_x = first_dot_x + np.arange(dot_count, dtype=np.float64)
    rows[:, layout.MAP_POSITION] = np.stack(
        [along_x, np.zeros(dot_count)], axis=1
    )
    rows[:, layout.MAP_DIRECTION] = np.array([1.0, 0.0])
    lane_column = layout.MAP_KIND.start + layout.MAP_POLYLINE_KINDS.index(
        "lane"
    )
    rows[:, lane_column] = 1.0
    return rows


def synthetic_inputs(size):
    """A random batch of size.samples targets, one object type per target in
    turn, each with size.agents agents (the target first) and size.polylines
    chunks of size.dots_per_polyline dots.
    """
    history = layout.HISTORY_STEPS
    signals = layout.NUM_TRAFFIC_SIGNAL_STATES
    batch = size.samples
    agent_shape = (batch, size.agents, history)
    agent_history = torch.randn(*agent_shape, layout.AGENT_FEATURE_DIM)
    agent_history[..., layout.AGENT_TYPE] = 0.0
    for sample_index in range(batch):
        object_type = sample_index % layout.NUM_OBJECT_TYPES
        type_column = layout.AGENT_TYPE.start + object_type
        agent_history[sample_index, :, :, type_column] = 1.0
    dot_shape = (batch, size.polylines, size.dots_per_polyline)
    chunk_rows = torch.randn(*dot_shape, layout.MAP_FEATURE_DIM)
    codes = torch.randint(0, layout.NUM_BOUNDARY_CROSSING_CODES, dot_shape)
    chunk_rows[..., layout.MAP_LEFT_BOUNDARY_CROSSING] = codes.float()
    chunk_rows[..., layout.MAP_RIGHT_BOUNDARY_CROSSING] = codes.float()
    return TargetInputs(
        agent_history=agent_history,
        agent_history_mask=torch.ones(agent_shape, dtype=torch.bool),
        chunk_rows=chunk_rows,
        chunk_dot_mask=torch.ones(dot_shape, dtype=torch.bool),
        chunk_signal_history=torch.rand(
            batch, size.polylines, history, signals
        ),
        future_positions=torch.randn(batch, layout.FUTURE_STEPS, 2),
        future_mask=torch.ones(batch, layout.FUTURE_STEPS, dtype=torch.bool),
    )


def future_targets(future_positions, future_mask):
    """A TargetInputs holding only a future, for the loss and monitor tests."""
    blanks = [None] * (len(TargetInputs._fields) - 2)
    return TargetInputs(*blanks, future_positions, future_mask)


def test_agent_frame_transform_inverts():
    """Into an agent's frame and back out returns the same world positions."""
    random_generator = np.random.default_rng(3)
    world_positions = random_generator.uniform(-120.0, 120.0, size=(64, 2))
    frame = Frame(np.array([13.5, -42.25]), 0.9137)

    local = frames.positions_to_frame(world_positions, frame)
    recovered = frames.positions_from_frame(local, frame)
    assert np.allclose(recovered, world_positions, atol=1e-9)


def test_agent_frame_x_axis_points_ahead():
    """A point straight ahead of the agent lands on its forward axis."""
    origin = np.array([13.5, -42.25])
    heading = 0.9137
    frame = Frame(origin, heading)
    forward = np.array([np.cos(heading), np.sin(heading)])
    leftward = np.array([-np.sin(heading), np.cos(heading)])
    distance = 17.0

    straight_ahead = frames.positions_to_frame(
        origin + distance * forward, frame
    )
    off_to_the_left = frames.positions_to_frame(
        origin + distance * leftward, frame
    )

    assert np.allclose(straight_ahead, np.array([distance, 0.0]), atol=1e-9)
    assert np.allclose(off_to_the_left, np.array([0.0, distance]), atol=1e-9)

    transposed_convention = (distance * forward) @ frames.rotation_matrix(
        heading
    ).T
    assert not np.allclose(transposed_convention, straight_ahead, atol=1e-9)


def test_heading_wrap_stays_in_range():
    """Wrapped angles land in the half-open range from -pi to pi."""
    angles = np.array([-7.0, -np.pi, 0.0, np.pi, 7.0, 100.0])
    wrapped = frames.wrap_to_pi(angles)
    assert (wrapped >= -np.pi).all() and (wrapped < np.pi).all()


def turning_track():
    """A vehicle track that drives an arc, every step valid."""
    track = scenario_pb2.Track()
    track.object_type = scenario_pb2.Track.TYPE_VEHICLE
    for step_index in range(layout.TOTAL_STEPS):
        turn_angle = 0.03 * step_index
        state = track.states.add()
        state.center_x = 40.0 + 1.5 * step_index * np.cos(turn_angle)
        state.center_y = -25.0 + 1.5 * step_index * np.sin(turn_angle)
        state.heading = turn_angle + 0.4
        state.velocity_x = 15.0 * np.cos(turn_angle + 0.4)
        state.velocity_y = 15.0 * np.sin(turn_angle + 0.4)
        state.length = 4.5
        state.width = 2.0
        state.valid = True
    return track


def test_frame_transforms_compose():
    """World to scene to agent must equal world to agent directly."""
    track = turning_track()
    sdc_frame = Frame(np.array([-12.0, 31.0]), -1.2)
    stored_rows = store.track_rows(track, sdc_frame, is_sdc=False)
    track_rows = stored_rows.astype(np.float32)

    frame = frames.agent_frame(track_rows)
    two_step = frames.rows_to_frame(track_rows, frame)

    world_positions = np.array(
        [[state.center_x, state.center_y] for state in track.states]
    )
    world_headings = np.array([state.heading for state in track.states])
    world_velocities = np.array(
        [[state.velocity_x, state.velocity_y] for state in track.states]
    )
    now_state = track.states[layout.CURRENT_STEP_INDEX]
    now_frame = Frame(
        np.array([now_state.center_x, now_state.center_y]), now_state.heading
    )

    direct_positions = frames.positions_to_frame(world_positions, now_frame)
    direct_headings = world_headings - now_state.heading
    direct_velocities = frames.directions_to_frame(
        world_velocities, now_state.heading
    )

    assert np.allclose(
        two_step[:, layout.AGENT_POSITION], direct_positions, atol=1e-3
    )
    assert np.allclose(
        two_step[:, layout.AGENT_HEADING_COSINE],
        np.cos(direct_headings),
        atol=1e-3,
    )
    assert np.allclose(
        two_step[:, layout.AGENT_HEADING_SINE],
        np.sin(direct_headings),
        atol=1e-3,
    )
    assert np.allclose(
        two_step[:, layout.AGENT_VELOCITY], direct_velocities, atol=1e-3
    )


def rotated_scenario(scenario, angle, shift):
    """The same staged scenario with every position, direction and heading
    turned by angle about the origin and then moved by shift, computed with
    its own rotation matrix rather than frame_ops.
    """
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    tracks = scenario.track_rows.astype(np.float64).copy()
    for columns in (layout.AGENT_VELOCITY, slice(2, 4)):
        tracks[..., columns] = tracks[..., columns] @ rotation.T
    positions = tracks[..., layout.AGENT_POSITION]
    tracks[..., layout.AGENT_POSITION] = positions @ rotation.T + shift
    rows = scenario.map_rows.astype(np.float64).copy()
    directions = rows[:, layout.MAP_DIRECTION]
    rows[:, layout.MAP_DIRECTION] = directions @ rotation.T
    rows[:, layout.MAP_POSITION] = rows[:, layout.MAP_POSITION] @ rotation.T
    rows[:, layout.MAP_POSITION] += shift
    stops = rows[:, layout.MAP_STOP_POINT]
    has_stop = (stops != 0.0).any(axis=1)
    rows[has_stop, layout.MAP_STOP_POINT] = stops[has_stop] @ rotation.T
    rows[has_stop, layout.MAP_STOP_POINT] += shift
    return scenario._replace(
        track_rows=tracks.astype(np.float32), map_rows=rows.astype(np.float32)
    )


def test_inputs_ignore_where_the_scene_sits():
    """Turning and moving a whole staged scene must leave every target's
    inputs unchanged, since they are all expressed in the target's frame.
    """
    scenario_paths = sorted(STAGED_DIRECTORY.glob("*.npz"))[:1]
    if not scenario_paths:
        pytest.skip(f"no staged scenarios under {STAGED_DIRECTORY}")
    scenario = formats.StagedFile.load(scenario_paths[0])
    tracks = views.designated_tracks(scenario)
    moved = rotated_scenario(scenario, 2.1, np.array([-310.0, 75.5]))
    original = views.scene_inputs(scenario, tracks)
    turned = views.scene_inputs(moved, tracks)
    for name, before, after in zip(original._fields, original, turned):
        assert np.allclose(before, after, atol=2e-3), name


def test_lane_ahead_lands_ahead():
    """A target at (10, 5) facing +y sees a lane dot at (10, 15) 10 m straight
    ahead, at (10, 0) in its own frame.
    """
    lane = lane_polyline_rows(0.0, 1)
    lane[0, layout.MAP_POSITION] = [10.0, 15.0]
    lane[0, layout.MAP_DIRECTION] = [0.0, 1.0]
    frame = Frame(np.array([10.0, 5.0]), np.pi / 2)
    moved = frames.map_rows_to_frame(lane, frame)
    assert np.allclose(moved[0, layout.MAP_POSITION], [10.0, 0.0])
    assert np.allclose(moved[0, layout.MAP_DIRECTION], [1.0, 0.0])
    assert np.all(moved[0, layout.MAP_STOP_POINT] == 0.0)


def test_path_distance_matches_dense_sampling():
    """The closed-form distance to the 8 s path equals the smallest distance
    to 2,001 points sampled along it.
    """
    generator = np.random.default_rng(3)
    points = generator.uniform(-60.0, 60.0, (200, 2))
    for path_end in (np.array([30.0, -8.0]), np.zeros(2)):
        samples = np.linspace(0.0, 1.0, 2001)[:, None] * path_end
        gaps = np.linalg.norm(points[:, None] - samples[None], axis=-1)
        dense = gaps.min(axis=1)
        closed_form = views.segment_distance(points, path_end)
        assert np.allclose(closed_form, dense, atol=0.02)


class SyntheticMap(NamedTuple):
    """The three map fields views.scene_chunks reads."""

    feature_lengths: np.ndarray
    map_rows: np.ndarray
    polyline_signal_histories: np.ndarray


def one_dot_chunks(positions):
    """One single-dot polyline, so one chunk, per position."""
    rows = np.zeros((len(positions), layout.MAP_FEATURE_DIM), np.float32)
    rows[:, layout.MAP_POSITION] = positions
    lights = np.zeros((len(positions),) + layout.SIGNAL_HISTORY_SHAPE)
    lengths = np.ones(len(positions), dtype=np.int64)
    return views.scene_chunks(SyntheticMap(lengths, rows, lights))


def test_bounded_chunk_search_matches_a_full_sort():
    """Skipping chunks that cannot be among the nearest picks the same chunks
    in the same order as sorting them all: for every designated target of
    40 staged scenes, and for exact ties around a standing and a moving
    target, where the stable order by chunk index must survive.
    """
    scenario_paths = sorted(STAGED_DIRECTORY.glob("*.npz"))[:40]
    if not scenario_paths:
        pytest.skip(f"no staged scenarios under {STAGED_DIRECTORY}")
    for path in scenario_paths:
        scenario = formats.StagedFile.load(path)
        chunks = views.scene_chunks(scenario)
        for target in views.designated_tracks(scenario):
            now = scenario.track_rows[target, layout.CURRENT_STEP_INDEX]
            velocity = now[layout.AGENT_VELOCITY]
            travel = velocity * layout.FUTURE_HORIZON_SECONDS
            scene_path = views.ScenePath(now[layout.AGENT_POSITION], travel)
            expected = reference_implementations.nearest_chunks_full_sort(
                chunks, scene_path
            )
            chosen = views.nearest_chunks(chunks, scene_path)
            assert np.array_equal(chosen, expected), path.name

    # 200 chunks on each of four points 10 m away, then 200 at 40 m: the
    # cut at 256 falls inside a run of exact ties.
    ring = [[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0], [0.0, -10.0]]
    positions = np.array(ring * 200 + [[40.0, 0.0]] * 200, np.float32)
    shuffled = positions[np.random.default_rng(4).permutation(1000)]
    chunks = one_dot_chunks(shuffled)
    standing = np.zeros(2, np.float32)
    for travel in (np.zeros(2, np.float32), np.array([0.0, 30.0], np.float32)):
        scene_path = views.ScenePath(standing, travel)
        expected = reference_implementations.nearest_chunks_full_sort(
            chunks, scene_path
        )
        chosen = views.nearest_chunks(chunks, scene_path)
        assert np.array_equal(chosen, expected)


def test_moving_only_stop_points_matches_moving_every_dot():
    """map_rows_to_frame moves only the dots that carry a stop point; the
    rows must match moving every dot and zeroing the rest, byte for byte,
    including a stop point stored as negative zero.
    """
    generator = np.random.default_rng(6)
    rows = generator.normal(0.0, 50.0, (64, 20, layout.MAP_FEATURE_DIM))
    rows = rows.astype(np.float32)
    has_stop_point = generator.random((64, 20)) < 0.05
    rows[~has_stop_point, layout.MAP_STOP_POINT] = 0.0
    rows[0, 0, layout.MAP_STOP_POINT] = -0.0
    rows[0, 1, layout.MAP_STOP_POINT] = [0.0, 3.0]
    frame = Frame(np.array([12.5, -40.0], np.float32), 2.3)
    moved = frames.map_rows_to_frame(rows, frame)
    expected = reference_implementations.map_rows_to_frame_every_dot(
        rows, frame
    )
    assert moved.dtype == expected.dtype
    assert moved.tobytes() == expected.tobytes()


def test_padding_matches_numpy_pad():
    """padded fills to the slot count with zeros of the same dtype, and hands
    back an array that already fills them unchanged.
    """
    history = np.arange(3 * 11 * 13, dtype=np.float32).reshape(3, 11, 13)
    mask = np.ones((3, 11), dtype=bool)
    for array in (history, mask):
        widths = [(0, 5)] + [(0, 0)] * (array.ndim - 1)
        padded = views.padded(array, 8)
        assert padded.dtype == array.dtype
        assert np.array_equal(padded, np.pad(array, widths))
    assert views.padded(history, 3) is history


def test_padded_slots_change_nothing():
    """Extra empty agent and chunk slots, and masked history steps and map
    dots, all holding NaN under their masks, must not move any prediction.
    """
    torch.manual_seed(5)
    predictor = model.MotionPredictor(
        reference_implementations.unit_anchor_offsets_per_type()
    ).eval()
    size = SyntheticSize(samples=3, agents=4, polylines=5, dots_per_polyline=8)
    batch = synthetic_inputs(size)
    batch.chunk_dot_mask[:, :, 6:] = False
    batch.agent_history_mask[:, 1:, :3] = False
    wider = size._replace(agents=7, polylines=9)
    garbage = synthetic_inputs(wider)
    nan = float("nan")
    garbage.agent_history.fill_(nan)
    garbage.chunk_rows.fill_(nan)
    garbage.chunk_signal_history.fill_(nan)
    poisoned_history = batch.agent_history.clone()
    poisoned_history[~batch.agent_history_mask] = nan
    poisoned_rows = batch.chunk_rows.clone()
    poisoned_rows[~batch.chunk_dot_mask] = nan
    padded = TargetInputs(
        agent_history=torch.cat(
            [poisoned_history, garbage.agent_history[:, 4:]], dim=1
        ),
        agent_history_mask=torch.cat(
            [batch.agent_history_mask, torch.zeros(3, 3, 11, dtype=torch.bool)],
            dim=1,
        ),
        chunk_rows=torch.cat([poisoned_rows, garbage.chunk_rows[:, 5:]], 1),
        chunk_dot_mask=torch.cat(
            [batch.chunk_dot_mask, torch.zeros(3, 4, 8, dtype=torch.bool)], 1
        ),
        chunk_signal_history=torch.cat(
            [batch.chunk_signal_history, garbage.chunk_signal_history[:, 5:]],
            dim=1,
        ),
    )
    with torch.no_grad():
        base = predictor(batch)[-1]
        widened = predictor(padded)[-1]
    assert torch.allclose(base.trajectories, widened.trajectories, atol=1e-4)
    assert torch.allclose(
        base.confidence_logits, widened.confidence_logits, atol=1e-5
    )


def test_anchor_fitting_recovers_separated_clusters():
    """No anchor may be left without endpoints."""
    random_generator = np.random.default_rng(17)
    jitter = random_generator.uniform(-0.08, 0.08, (model.QUERY_COUNT, 2))
    cluster_centres = (
        reference_implementations.unit_anchor_offsets() * 10.0
        + torch.tensor(jitter, dtype=torch.float32)
    )
    spread = random_generator.normal(0.0, 0.005, (model.QUERY_COUNT, 50, 2))
    endpoints = cluster_centres[:, None, :] + torch.tensor(
        spread, dtype=torch.float32
    )
    endpoints = endpoints.reshape(-1, 2)

    fitted, counts = fit_anchors.fit_type("vehicle", endpoints)

    assert fitted.shape == (model.QUERY_COUNT, 2)
    nearest_cluster = torch.cdist(fitted, cluster_centres).argmin(dim=1)
    assert sorted(nearest_cluster.tolist()) == list(range(model.QUERY_COUNT))
    assert torch.allclose(fitted, cluster_centres[nearest_cluster], atol=0.01)
    assert int(counts.min()) > 0


def test_anchor_endpoints_are_eight_second_positions(tmp_path):
    """A track that ends before 8 s gives no endpoint, since an anchor is
    where the agent is at 8 s; one that runs the full horizon gives its
    position at the last step.
    """
    scenario = two_lane_signal_scenario(
        2, np.zeros(layout.SIGNAL_HISTORY_SHAPE)
    )
    track_rows = scenario.track_rows.copy()
    track_valid = scenario.track_valid.copy()
    # Both vehicles drive along x at 10 m/s, one metre per step.
    track_rows[:, :, layout.AGENT_TYPE.start] = 1.0
    track_rows[:, :, layout.AGENT_POSITION.start] = np.arange(
        layout.TOTAL_STEPS
    )
    track_rows[:, :, layout.AGENT_VELOCITY.start] = 10.0
    track_valid[1, 50:] = False
    path = tmp_path / "scene.npz"
    scenario._replace(
        track_rows=track_rows,
        track_valid=track_valid,
        provenance=formats.artifact_provenance("test", "endpoint test"),
    ).save(path)

    endpoints, type_index = fit_anchors.reachable_endpoints([path])

    eight_seconds = float(layout.FUTURE_STEPS)
    assert torch.allclose(endpoints, torch.tensor([[eight_seconds, 0.0]]))
    assert type_index.tolist() == [0]


def test_batched_pruning_matches_the_loop():
    """The plain single-sample loop is the reference."""
    torch.manual_seed(7)
    trajectories = pruning.PRUNE_DISTANCE_METRES * torch.randn(
        2, model.QUERY_COUNT, layout.FUTURE_STEPS, 2
    )
    confidence_logits = torch.randn(2, model.QUERY_COUNT)
    trajectories[1] = trajectories[1, :1]

    pruned = pruning.prune_modes(trajectories, confidence_logits)
    kept_counts = pruned.kept_count
    assert kept_counts[0] > 1 and kept_counts[0] <= layout.NUM_PREDICTED_MODES
    assert kept_counts[1] == 1
    for sample_index in range(len(kept_counts)):
        walked = reference_implementations.prune_modes_single_sample(
            trajectories[sample_index], confidence_logits[sample_index]
        )
        walked_trajectories, walked_logits = walked
        assert torch.equal(
            walked_trajectories, pruned.trajectories[sample_index]
        )
        assert torch.equal(
            walked_logits, pruned.confidence_logits[sample_index]
        )


def test_opposite_endpoints_get_opposite_anchors():
    """A future ending ahead and one ending behind are assigned anchors on those
    sides.
    """
    unit_anchors = reference_implementations.unit_anchor_offsets() * 40.0
    future_positions = torch.zeros(2, layout.FUTURE_STEPS, 2)
    future_positions[0, :, 0] = torch.linspace(0.5, 40.0, layout.FUTURE_STEPS)
    future_positions[1, :, 0] = torch.linspace(-0.5, -40.0, layout.FUTURE_STEPS)
    future_mask = torch.ones(2, layout.FUTURE_STEPS, dtype=torch.bool)

    targets = future_targets(future_positions, future_mask)
    assigned_mode = loss.anchor_assigned_mode(
        unit_anchors.expand(2, -1, -1), targets
    )

    assert assigned_mode[0] != assigned_mode[1]
    assert unit_anchors[assigned_mode[0], 0] > 0
    assert unit_anchors[assigned_mode[1], 0] < 0
    assert (
        unit_anchors[assigned_mode[0]] - unit_anchors[assigned_mode[1]]
    ).norm() > 20.0


def test_assignment_ignores_padded_steps():
    """Assignment scores the whole valid path, never the padded tail."""
    unit_anchors = reference_implementations.unit_anchor_offsets() * 40.0
    anchors = unit_anchors.expand(1, -1, -1)
    future_positions = torch.zeros(1, layout.FUTURE_STEPS, 2)
    future_positions[0, :10, 0] = 40.0
    future_mask = torch.zeros(1, layout.FUTURE_STEPS, dtype=torch.bool)
    future_mask[0, :10] = True

    assigned_mode = loss.anchor_assigned_mode(
        anchors, future_targets(future_positions, future_mask)
    )
    all_valid = torch.ones_like(future_mask)
    padded_tail_mode = loss.anchor_assigned_mode(
        anchors, future_targets(future_positions, all_valid)
    )

    assert torch.allclose(
        unit_anchors[assigned_mode[0]], torch.tensor([40.0, 0.0]), atol=1e-4
    )
    assert padded_tail_mode[0] != assigned_mode[0]
    assert float(unit_anchors[padded_tail_mode[0]].norm()) == pytest.approx(
        40.0 / reference_implementations.ANCHOR_DISTANCE_COUNT
    )

    waits_then_arrives = torch.zeros(1, layout.FUTURE_STEPS, 2)
    waits_then_arrives[0, -1, 0] = 40.0
    late_mode = loss.anchor_assigned_mode(
        anchors, future_targets(waits_then_arrives, all_valid)
    )
    assert float(unit_anchors[late_mode[0]].norm()) < 40.0


def test_submission_returns_to_the_world_frame():
    """The logged future, sent through the submission transform, must land where
    it was logged.
    """
    scenario_paths = sorted(STAGED_DIRECTORY.glob("*.npz"))[:1]
    if not scenario_paths:
        pytest.skip(f"no staged scenarios under {STAGED_DIRECTORY}")

    scenario = formats.StagedFile.load(scenario_paths[0])
    tracks = views.eligible_tracks(scenario, designated_only=True)[:1]
    track_index = tracks[0]
    targets = views.scene_inputs(scenario, tracks)
    agent_frame_future = targets.future_positions[0]
    frame = frames.agent_frame(scenario.track_rows[track_index])

    world_future = frames.to_world(
        agent_frame_future, frame, scenario.scene_frame
    )
    world_coordinate_resolution = (
        np.finfo(np.float32).eps * np.abs(world_future).max()
    )

    back_to_scene_frame = frames.positions_to_frame(
        world_future, scenario.scene_frame
    )
    back_to_agent_frame = frames.positions_to_frame(back_to_scene_frame, frame)
    assert np.allclose(
        back_to_agent_frame,
        agent_frame_future,
        atol=world_coordinate_resolution,
    )

    future_steps = slice(layout.CURRENT_STEP_INDEX + 1, None)
    logged_scene_future = scenario.track_rows[
        track_index, future_steps, layout.AGENT_POSITION
    ]
    logged_world_future = frames.positions_from_frame(
        logged_scene_future, scenario.scene_frame
    )
    logged = scenario.track_valid[track_index, future_steps]
    assert logged.any()
    assert np.allclose(
        world_future[logged],
        logged_world_future[logged],
        atol=world_coordinate_resolution,
    )


def test_constant_velocity_null_keeps_the_logged_speed():
    """A straight-moving agent is carried at exactly its logged velocity."""
    logged_velocity = torch.tensor([[11.0, -4.0], [0.0, 0.0]])
    start = torch.tensor([3.0, 5.0])
    agent_history = torch.zeros(
        2, layout.HISTORY_STEPS, layout.AGENT_FEATURE_DIM
    )
    agent_history[:, :, layout.AGENT_HEADING_COSINE] = 1.0
    agent_history[:, layout.CURRENT_STEP_INDEX, layout.AGENT_VELOCITY] = (
        logged_velocity
    )
    agent_history[:, layout.CURRENT_STEP_INDEX, layout.AGENT_POSITION] = start

    trajectories, _ = baseline.constant_velocity(agent_history)
    step_numbers = torch.arange(1, layout.FUTURE_STEPS + 1, dtype=torch.float32)
    elapsed_seconds = layout.TIMESTEP_SECONDS * step_numbers
    straight_line = start + logged_velocity[0] * elapsed_seconds[:, None]

    assert torch.allclose(trajectories[0, 0], straight_line, atol=1e-4)
    step_length = float(logged_velocity[0].norm() * layout.TIMESTEP_SECONDS)
    assert torch.allclose(
        trajectories[0, 0].diff(dim=0).norm(dim=-1),
        torch.full((layout.FUTURE_STEPS - 1,), step_length),
        atol=1e-4,
    )
    assert torch.allclose(
        trajectories[1, 0], start.expand(layout.FUTURE_STEPS, 2), atol=1e-4
    )


def test_artifact_provenance_round_trips():
    """A missing stamp, or one from another code version, is refused."""
    stamp = formats.artifact_provenance(
        "test_invariants.py", "unit-test-source"
    )
    checked = formats.check_provenance(stamp, "in-memory", "Regenerate.")
    assert checked["code_version"] == layout.STAGING_CODE_VERSION
    assert checked["producer"] == "test_invariants.py"

    with pytest.raises(ValueError):
        formats.check_provenance(None, "in-memory", "Regenerate.")

    stale = json.dumps(
        {"code_version": "not-the-working-tree", "producer": "x", "source": "y"}
    )
    with pytest.raises(ValueError):
        formats.check_provenance(stale, "in-memory", "Regenerate.")


def test_anchor_file_from_an_older_fit_is_refused(tmp_path):
    """The staging code version cannot tell anchor fits apart, so an anchor
    file also carries the fit version; one without it is refused.
    """
    anchor_shape = (layout.NUM_OBJECT_TYPES, model.QUERY_COUNT, 2)
    anchors = np.zeros(anchor_shape, dtype=np.float32)
    current = tmp_path / "current.npz"
    np.savez(
        current,
        unit_anchors=anchors,
        provenance=formats.anchor_provenance("unit-test-source"),
    )
    assert checkpoint.load_anchor_file(current).shape == anchor_shape

    older = tmp_path / "older.npz"
    np.savez(
        older,
        unit_anchors=anchors,
        provenance=formats.artifact_provenance("fit_anchors.py", "y"),
    )
    with pytest.raises(ValueError, match=layout.ANCHOR_FIT_VERSION):
        checkpoint.load_anchor_file(older)


def write_scorer_scenario(path, scenario_id, agent_count):
    """A staged file of agent_count agents at x = 0, 1, 2, ... with no map."""
    track_rows = np.zeros(
        (agent_count, layout.TOTAL_STEPS, layout.AGENT_FEATURE_DIM), np.float32
    )
    track_rows[:, :, layout.AGENT_HEADING_COSINE] = 1.0
    track_rows[:, :, layout.AGENT_TYPE.start] = 1.0
    for track_index in range(agent_count):
        track_rows[track_index, :, layout.AGENT_POSITION.start] = float(
            track_index
        )
    signal_shape = (0, layout.HISTORY_STEPS, layout.NUM_TRAFFIC_SIGNAL_STATES)
    np.savez(
        path,
        track_rows=track_rows,
        track_valid=np.ones((agent_count, layout.TOTAL_STEPS), dtype=bool),
        track_ids=np.arange(10, 10 * (agent_count + 1), 10, dtype=np.int64),
        is_designated_target=np.ones(agent_count, dtype=bool),
        is_object_of_interest=np.zeros(agent_count, dtype=bool),
        map_rows=np.zeros((0, layout.MAP_FEATURE_DIM), dtype=np.float32),
        feature_lengths=np.zeros(0, dtype=np.int64),
        feature_ids=np.zeros(0, dtype=np.int64),
        feature_is_interpolating=np.zeros(0, dtype=bool),
        polyline_signal_histories=np.zeros(signal_shape, dtype=np.float32),
        frame_origin=np.zeros(2, dtype=np.float32),
        frame_heading=np.float32(0.0),
        scenario_id=scenario_id,
    )


def test_scorer_tensors_group_by_scenario(tmp_path):
    """Ground truth holds every agent, not only the predicted ones."""
    runner_spec = importlib.util.spec_from_file_location(
        "container_runner", RUNNER_PATH
    )
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)

    scenario_agent_counts = {"scn_a": 3, "scn_b": 2}
    scenario_target_track_ids = {"scn_a": [20, 30], "scn_b": [10]}
    for scenario_id, agent_count in scenario_agent_counts.items():
        write_scorer_scenario(
            tmp_path / f"{scenario_id}.npz", scenario_id, agent_count
        )

    prediction_scenario_ids = []
    prediction_track_ids = []
    for scenario_id, target_track_ids in scenario_target_track_ids.items():
        for track_id in target_track_ids:
            prediction_scenario_ids.append(scenario_id)
            prediction_track_ids.append(track_id)
    target_count = len(prediction_track_ids)
    trajectory_shape = (
        target_count,
        layout.NUM_PREDICTED_MODES,
        layout.SUBMISSION_STEPS,
        2,
    )
    predictions = formats.Submission(
        scenario_id=np.array(prediction_scenario_ids),
        track_id=np.array(prediction_track_ids, dtype=np.int64),
        world_trajectories=np.random.default_rng(0).normal(
            size=trajectory_shape
        ),
        confidences=np.full(
            (target_count, layout.NUM_PREDICTED_MODES), 1.0 / 6.0
        ),
        provenance=formats.artifact_provenance("test", "scorer-tensors"),
    )

    tensors = runner.metric_inputs(predictions, tmp_path)

    assert tensors.ground_truth_trajectory.shape == (
        2,
        3,
        layout.TOTAL_STEPS,
        7,
    )
    assert tensors.prediction_trajectory.shape == (
        2,
        2,
        layout.NUM_PREDICTED_MODES,
        1,
        layout.SUBMISSION_STEPS,
        2,
    )
    assert (
        int(tensors.prediction_ground_truth_indices_mask.sum()) == target_count
    )

    scenario_row = {
        scenario_id: row for row, scenario_id in enumerate(tensors.scenario_id)
    }
    a_row, b_row = scenario_row["scn_a"], scenario_row["scn_b"]
    assert tensors.ground_truth_is_valid[a_row].all()
    assert tensors.ground_truth_is_valid[b_row, :2].all()
    assert not tensors.ground_truth_is_valid[b_row, 2:].any()
    assert not tensors.prediction_ground_truth_indices_mask[b_row, 1:].any()

    for slot_index, expected_track_id in enumerate(
        scenario_target_track_ids["scn_a"]
    ):
        agent_row = int(
            tensors.prediction_ground_truth_indices[a_row, slot_index, 0]
        )
        assert tensors.object_id[a_row, agent_row] == expected_track_id
        first_x = tensors.ground_truth_trajectory[a_row, agent_row, 0, 0]
        assert first_x == pytest.approx(agent_row * 1.0)


def test_input_order_does_not_change_predictions():
    """Other agents, map chunks and the dots inside each chunk are shuffled;
    trajectories must not move.
    """
    torch.manual_seed(11)
    predictor = model.MotionPredictor(
        reference_implementations.unit_anchor_offsets_per_type()
    ).eval()
    size = SyntheticSize(samples=2, agents=6, polylines=5, dots_per_polyline=16)
    batch = synthetic_inputs(size)
    batch.agent_history_mask[:, 5] = False
    agent_order = torch.cat(
        [torch.zeros(1, dtype=torch.long), 1 + torch.randperm(5)]
    )
    chunk_order = torch.randperm(5)
    dot_order = torch.randperm(16)
    chunk_rows = batch.chunk_rows[:, chunk_order][:, :, dot_order]
    permuted = TargetInputs(
        agent_history=batch.agent_history[:, agent_order],
        agent_history_mask=batch.agent_history_mask[:, agent_order],
        chunk_rows=chunk_rows,
        chunk_dot_mask=batch.chunk_dot_mask[:, chunk_order][:, :, dot_order],
        chunk_signal_history=batch.chunk_signal_history[:, chunk_order],
    )
    with torch.no_grad():
        base = predictor(batch)[-1]
        shuffled = predictor(permuted)[-1]
    assert torch.allclose(base.trajectories, shuffled.trajectories, atol=1e-4)
    assert torch.allclose(
        base.confidence_logits, shuffled.confidence_logits, atol=1e-5
    )


def two_lane_signal_scenario(track_count, signalled_history):
    """A staged scenario with a signalled lane under the first track and an
    unsignalled lane 40 m away.
    """
    signalled_lane = lane_polyline_rows(0.0, 11)
    unsignalled_lane = lane_polyline_rows(0.0, 11)
    unsignalled_lane[:, layout.MAP_POSITION.start + 1] = 40.0
    signal_shape = (2, layout.HISTORY_STEPS, layout.NUM_TRAFFIC_SIGNAL_STATES)
    polyline_signal_histories = np.zeros(signal_shape, dtype=np.float32)
    polyline_signal_histories[0] = signalled_history
    track_shape = (track_count, layout.TOTAL_STEPS, layout.AGENT_FEATURE_DIM)
    track_rows = np.zeros(track_shape, dtype=np.float32)
    track_valid = np.ones((track_count, layout.TOTAL_STEPS), dtype=bool)
    track_rows[:, :, layout.AGENT_HEADING_COSINE] = 1.0
    track_rows[0, :, layout.AGENT_POSITION] = np.array([5.0, 0.0])
    track_rows[1, :, layout.AGENT_POSITION] = np.array([5.0, 40.0])
    if track_count > 2:
        track_rows[2, :, layout.AGENT_POSITION] = np.array([7.0, 0.0])
        track_valid[2, layout.CURRENT_STEP_INDEX] = False
    return formats.StagedFile(
        track_rows=track_rows,
        track_valid=track_valid,
        track_ids=np.arange(track_count),
        is_designated_target=np.ones(track_count, dtype=bool),
        is_object_of_interest=np.zeros(track_count, dtype=bool),
        map_rows=np.concatenate([signalled_lane, unsignalled_lane]),
        feature_lengths=np.array([11, 11], dtype=np.int64),
        feature_ids=np.array([101, 102], dtype=np.int64),
        feature_is_interpolating=np.zeros(2, dtype=bool),
        polyline_signal_histories=polyline_signal_histories,
        frame_origin=np.zeros(2, dtype=np.float32),
        frame_heading=np.float32(0.0),
        scenario_id=np.array("two-lane"),
    )


def test_lane_chunks_carry_their_light():
    """The chunk of a signalled lane carries that lane's light history, the
    other lane's chunk carries none, and the prediction depends on it.
    """
    signal_shape = (layout.HISTORY_STEPS, layout.NUM_TRAFFIC_SIGNAL_STATES)
    signalled_history = np.zeros(signal_shape, dtype=np.float32)
    stop_column = layout.TRAFFIC_SIGNAL_STATES.index("LANE_STATE_STOP")
    go_column = layout.TRAFFIC_SIGNAL_STATES.index("LANE_STATE_GO")
    signalled_history[:8, stop_column] = 1.0
    signalled_history[8:, go_column] = 1.0
    scenario = two_lane_signal_scenario(3, signalled_history)
    target_inputs = views.scene_inputs(scenario, np.array([0]))
    lights = target_inputs.chunk_signal_history[0]
    lit_chunks = [chunk for chunk in lights if chunk.any()]
    assert len(lit_chunks) == 1
    assert np.array_equal(lit_chunks[0], signalled_history)

    batch = target_inputs.each(torch.from_numpy)
    dark = batch._replace(
        chunk_signal_history=torch.zeros_like(batch.chunk_signal_history)
    )
    torch.manual_seed(47)
    predictor = model.MotionPredictor(
        reference_implementations.unit_anchor_offsets_per_type()
    ).eval()
    with torch.no_grad():
        lit = predictor(batch)[-1].trajectories
        unlit = predictor(dark)[-1].trajectories
    assert not torch.allclose(lit, unlit)


def test_target_prediction_ignores_batch_company():
    """A target predicted alone must equal the same target batched beside other
    targets from another scene.
    """
    signal_shape = (layout.HISTORY_STEPS, layout.NUM_TRAFFIC_SIGNAL_STATES)
    signalled_history = np.zeros(signal_shape, dtype=np.float32)
    three_track_scenario = two_lane_signal_scenario(3, signalled_history)
    two_track_scenario = two_lane_signal_scenario(2, signalled_history)

    torch.manual_seed(53)
    predictor = model.MotionPredictor(
        reference_implementations.unit_anchor_offsets_per_type()
    ).eval()
    alone = views.scene_inputs(three_track_scenario, np.array([1]))
    first = views.scene_inputs(two_track_scenario, np.array([0]))
    second = views.scene_inputs(three_track_scenario, np.array([0, 1]))
    together = TargetInputs(*map(np.concatenate, zip(first, second)))
    with torch.no_grad():
        alone_paths = predictor(alone.each(torch.from_numpy))[-1].trajectories
        together_paths = predictor(together.each(torch.from_numpy))[
            -1
        ].trajectories
    assert torch.allclose(alone_paths[0], together_paths[2], atol=1e-4)


def test_zero_head_output_lands_on_the_anchor():
    """With a zero head a mode ends at its anchor, and the loss assigns by that
    same anchor.
    """
    torch.manual_seed(37)
    reach = torch.tensor([40.0, 24.0, 12.0])[:, None, None]
    unit_anchors = (
        reference_implementations.unit_anchor_offsets_per_type() * reach
    )
    predictor = model.MotionPredictor(unit_anchors).eval()
    with torch.no_grad():
        predictor.mode_decoder.trajectory_head[-1].weight.zero_()
        predictor.mode_decoder.trajectory_head[-1].bias.zero_()
    size = SyntheticSize(
        samples=layout.NUM_OBJECT_TYPES,
        agents=2,
        polylines=2,
        dots_per_polyline=10,
    )
    batch = synthetic_inputs(size)
    with torch.no_grad():
        predicted = predictor(batch)[-1]
    assert torch.allclose(
        predicted.trajectories[:, :, -1], unit_anchors, atol=1e-4
    )
    assert torch.allclose(predicted.anchors, unit_anchors)
    ramp = torch.arange(1, layout.FUTURE_STEPS + 1) / layout.FUTURE_STEPS
    logged_futures = unit_anchors[0][:, None, :] * ramp[None, :, None]
    all_valid = torch.ones(
        model.QUERY_COUNT, layout.FUTURE_STEPS, dtype=torch.bool
    )
    assigned_mode = loss.anchor_assigned_mode(
        unit_anchors[0].expand(model.QUERY_COUNT, -1, -1),
        future_targets(logged_futures, all_valid),
    )
    assert torch.equal(assigned_mode, torch.arange(model.QUERY_COUNT))


def test_regression_term_is_gaussian_nll():
    """Checked against the closed form at the uncertainty the model states."""
    torch.manual_seed(53)
    sample_count = 4
    future_positions = torch.randn(sample_count, layout.FUTURE_STEPS, 2)
    future_mask = torch.ones(
        sample_count, layout.FUTURE_STEPS, dtype=torch.bool
    )
    targets = future_targets(future_positions, future_mask)
    confidence_logits = torch.zeros(sample_count, model.QUERY_COUNT)
    unit_anchors = reference_implementations.unit_anchor_offsets().expand(
        sample_count, -1, -1
    )
    displacement = torch.tensor([0.3, -0.4])

    def regression_at(log_standard_deviation, error):
        """The regression loss when every mode misses by error at the given log
        standard deviation.
        """
        trajectories = (
            future_positions.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1, -1)
            + error
        )
        log_sigma = torch.full_like(trajectories, log_standard_deviation)
        predicted = ModePredictions(
            trajectories, log_sigma, confidence_logits, unit_anchors
        )
        return float(loss.loss_terms(predicted, targets).regression)

    floor = model.MINIMUM_LOG_STANDARD_DEVIATION
    squared_miss = float((displacement**2).sum())
    exact_form = (
        2.0 * floor
        + 2.0 * loss.HALF_LOG_TWO_PI
        + 0.5 * squared_miss / math.exp(floor) ** 2
    )
    assert regression_at(floor, displacement) == pytest.approx(
        exact_form, rel=1e-5
    )
    for log_standard_deviation in (
        floor,
        0.0,
        model.MAXIMUM_LOG_STANDARD_DEVIATION,
    ):
        exact_hit = regression_at(log_standard_deviation, torch.zeros(2))
        assert exact_hit < regression_at(log_standard_deviation, displacement)


def test_classification_term_is_cross_entropy():
    """Taken over anchors, against the assigned mode."""
    sample_count = 3
    future_positions = torch.zeros(sample_count, layout.FUTURE_STEPS, 2)
    future_positions[:, -1, 0] = 1.0
    future_mask = torch.ones(
        sample_count, layout.FUTURE_STEPS, dtype=torch.bool
    )
    trajectories = future_positions.unsqueeze(1).expand(
        -1, model.QUERY_COUNT, -1, -1
    )
    unit_anchors = torch.zeros(sample_count, model.QUERY_COUNT, 2)
    unit_anchors[:, 0, 0] = 1.0
    predicted = ModePredictions(
        trajectories,
        torch.zeros_like(trajectories),
        torch.zeros(sample_count, model.QUERY_COUNT),
        unit_anchors,
    )
    terms = loss.loss_terms(
        predicted, future_targets(future_positions, future_mask)
    )
    assert float(terms.classification) == pytest.approx(
        math.log(model.QUERY_COUNT), rel=1e-5
    )


def test_masked_steps_move_no_loss():
    """Values at masked future steps cannot change any loss component."""
    torch.manual_seed(103)
    sample_count = 4
    future_mask = torch.ones(
        sample_count, layout.FUTURE_STEPS, dtype=torch.bool
    )
    future_mask[1, layout.FUTURE_STEPS // 2 :] = False
    future_mask[2] = False
    future_positions = torch.randn(sample_count, layout.FUTURE_STEPS, 2)
    trajectories = torch.randn(
        sample_count, model.QUERY_COUNT, layout.FUTURE_STEPS, 2
    )
    log_standard_deviation = torch.randn_like(trajectories)
    confidence_logits = torch.randn(sample_count, model.QUERY_COUNT)
    unit_anchors = (
        reference_implementations.unit_anchor_offsets().expand(
            sample_count, -1, -1
        )
        * 40.0
    )

    def components(logged_positions, predicted_positions):
        predicted = ModePredictions(
            predicted_positions,
            log_standard_deviation,
            confidence_logits,
            unit_anchors,
        )
        return torch.stack(
            loss.loss_terms(
                predicted, future_targets(logged_positions, future_mask)
            )
        )

    masked_steps = ~future_mask
    poisoned_positions = future_positions.clone()
    poisoned_positions[masked_steps] = float("nan")
    poisoned_trajectories = trajectories.clone()
    poisoned_trajectories[
        masked_steps.unsqueeze(1).expand(-1, model.QUERY_COUNT, -1)
    ] = 1e6
    assert torch.equal(
        components(future_positions, trajectories),
        components(poisoned_positions, poisoned_trajectories),
    )


def test_every_model_output_moves_the_loss():
    """An output the loss ignored could drift freely, so each one is perturbed
    and the loss must change.
    """
    torch.manual_seed(83)
    predictor = model.MotionPredictor(
        reference_implementations.unit_anchor_offsets_per_type()
    ).eval()
    size = SyntheticSize(
        samples=layout.NUM_OBJECT_TYPES,
        agents=3,
        polylines=2,
        dots_per_polyline=10,
    )
    batch = synthetic_inputs(size)
    with torch.no_grad():
        emitted = predictor(batch)[-1]
    perturbations = {
        "trajectories": lambda value: value + 1.0,
        "log_standard_deviation": lambda value: value + 1.0,
        "confidence_logits": lambda value: torch.cat(
            [value[..., :1] + 1.0, value[..., 1:]], dim=-1
        ),
        "anchors": lambda value: -value,
    }
    assert set(perturbations) == set(ModePredictions._fields)

    def total(predicted):
        return loss.loss_terms(predicted, batch).total

    unperturbed = total(emitted)
    for name, perturb in perturbations.items():
        perturbed = emitted._replace(**{name: perturb(getattr(emitted, name))})
        assert not torch.equal(total(perturbed), unperturbed), name


def test_no_step_uncertainty_starts_saturated():
    """Untrained modes must also start out distinct from one another."""
    torch.manual_seed(97)
    predictor = model.MotionPredictor(
        reference_implementations.unit_anchor_offsets_per_type()
    ).eval()
    size = SyntheticSize(
        samples=layout.NUM_OBJECT_TYPES,
        agents=2,
        polylines=2,
        dots_per_polyline=10,
    )
    batch = synthetic_inputs(size)
    with torch.no_grad():
        predicted = predictor(batch)[-1]
    assert predicted.trajectories.isfinite().all()
    assert predicted.confidence_logits.isfinite().all()
    log_sigma = predicted.log_standard_deviation
    assert not (log_sigma == model.MINIMUM_LOG_STANDARD_DEVIATION).any()
    assert not (log_sigma == model.MAXIMUM_LOG_STANDARD_DEVIATION).any()
    endpoints = predicted.trajectories[:, :, -1]
    assert not torch.allclose(endpoints[:, :1], endpoints, atol=1e-4)


def test_anchors_are_frozen_buffers():
    """They take no gradient and are saved inside the checkpoint."""
    predictor = model.MotionPredictor(
        reference_implementations.unit_anchor_offsets_per_type()
    )
    parameter_names = {name for name, _ in predictor.named_parameters()}
    assert "mode_decoder.unit_anchors" not in parameter_names
    assert "mode_decoder.unit_anchors" in predictor.state_dict()
    assert not predictor.unit_anchors.requires_grad


def test_training_step_runs_on_staged_scenarios(tmp_path):
    """One optimiser step on two real staged scenarios gives finite losses and
    gradients. Skipped when no staged data is present.
    """
    scenario_paths = sorted(STAGED_DIRECTORY.glob("*.npz"))[:2]
    if not scenario_paths:
        pytest.skip(f"no staged scenarios under {STAGED_DIRECTORY}")
    torch.manual_seed(0)
    predictor = model.MotionPredictor(
        reference_implementations.unit_anchor_offsets_per_type()
    )
    optimizer = torch.optim.AdamW(
        train.parameter_groups(predictor), lr=train.LEARNING_RATE
    )
    batch = next(iter(pipeline.batches(scenario_paths, TWO_STAGED_SCENARIOS)))
    predicted = predictor(batch)[-1]
    losses = loss.loss_terms(predicted, batch)
    assert torch.stack(list(losses)).isfinite().all()
    losses.total.backward()
    starved = [
        name
        for name, parameter in predictor.named_parameters()
        if parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
    ]
    assert not starved, starved
    optimizer.step()
    progress = metrics.EpochProgress(torch.device("cpu"))
    detached = ModePredictions(*(tensor.detach() for tensor in predicted))
    progress.add_kept_mode_errors(detached, batch)
    scalars = progress.epoch_scalars()
    assert math.isfinite(scalars["epoch/ade_80step"])
    assert math.isfinite(scalars["epoch/fde_80step"])
    checkpoint_path = tmp_path / "predictor.pt"
    torch.save(predictor.state_dict(), checkpoint_path)
    reloaded_state = torch.load(checkpoint_path)
    assert all(
        torch.equal(reloaded_state[name], parameter)
        for name, parameter in predictor.state_dict().items()
    )


def test_pruning_conserves_probability_mass():
    """A dropped duplicate's share goes to the mode kept in its place."""
    torch.manual_seed(131)
    trajectories = torch.zeros(1, model.QUERY_COUNT, layout.FUTURE_STEPS, 2)
    trajectories[0, :, -1, 0] = 10.0 * torch.arange(
        model.QUERY_COUNT, dtype=torch.float32
    )
    trajectories[0, 1, -1, 0] = 0.5
    confidence_logits = torch.zeros(1, model.QUERY_COUNT)
    confidence_logits[0, 0] = 3.0
    confidence_logits[0, 1] = 2.0
    pruned = pruning.prune_modes(trajectories, confidence_logits)
    kept_trajectories = pruned.trajectories
    confidences = pruning.kept_confidences(
        trajectories, confidence_logits, kept_trajectories
    )
    probabilities = torch.softmax(confidence_logits, dim=-1)[0]
    assert float(confidences.sum()) == pytest.approx(1.0, rel=1e-6)
    assert float(kept_trajectories[0, 0, -1, 0]) == 0.0
    assert not (kept_trajectories[0, :, -1, 0] == 0.5).any()
    merged = float(probabilities[0] + probabilities[1])
    assert float(confidences[0, 0]) == pytest.approx(merged, rel=1e-6)
    assert float(confidences[0, 1:].sum()) == pytest.approx(
        1.0 - merged, rel=1e-5
    )


def test_step_head_walks_a_line_or_an_arc():
    """Constant steps walk a straight line; steps that rotate walk an arc."""
    torch.manual_seed(137)
    decoder = model.ModeDecoder(
        reference_implementations.unit_anchor_offsets_per_type() * 0.0
    ).eval()
    memory = model.Memory(
        torch.randn(1, 5, model.HIDDEN_DIM), torch.ones(1, 5, dtype=torch.bool)
    )
    type_index = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        decoder.trajectory_head[-1].weight.zero_()
        bias = decoder.trajectory_head[-1].bias.view(2, layout.FUTURE_STEPS, 2)
        bias.zero_()
        bias[0, :, 0] = 0.5
        straight = decoder(memory, type_index)[-1].trajectories
        angles = 0.02 * torch.arange(layout.FUTURE_STEPS, dtype=torch.float32)
        bias[0, :, 0] = 0.5 * angles.cos()
        bias[0, :, 1] = 0.5 * angles.sin()
        arc = decoder(memory, type_index)[-1].trajectories
    step_numbers = torch.arange(1, layout.FUTURE_STEPS + 1, dtype=torch.float32)
    expected_straight = torch.stack(
        [0.5 * step_numbers, torch.zeros(layout.FUTURE_STEPS)], dim=-1
    )
    assert torch.allclose(straight[0, 0], expected_straight, atol=1e-5)
    steps = arc[0, 0].diff(dim=0)
    assert torch.allclose(
        steps.norm(dim=-1),
        torch.full((layout.FUTURE_STEPS - 1,), 0.5),
        atol=1e-5,
    )
    turning = torch.atan2(steps[:, 1], steps[:, 0]).diff()
    assert torch.allclose(turning, torch.full_like(turning, 0.02), atol=1e-5)


def test_unreadable_staged_file_names_itself(tmp_path):
    """One bad file among 122,352 must say which file it is."""
    source = sorted(STAGED_DIRECTORY.glob("*.npz"))[0]
    file_bytes = source.read_bytes()
    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes(file_bytes[: len(file_bytes) // 2])
    with np.load(source) as staged:
        kept_names = [name for name in staged.files if name != "map_rows"]
        kept_arrays = {name: staged[name] for name in kept_names}
    missing = tmp_path / "missing.npz"
    np.savez_compressed(missing, **kept_arrays)
    for path in (truncated, missing):
        with pytest.raises(ValueError, match=re.escape(str(path))):
            formats.StagedFile.load(path)
    with pytest.raises(ValueError, match=re.escape(str(truncated))):
        formats.TrackArrays.load(truncated)


def test_step_count_matches_a_serial_count():
    """The threaded count must equal a serial count that re-derives each
    eligible track from the raw staged arrays.
    """
    paths = sorted(STAGED_DIRECTORY.glob("*.npz"))
    worker_count = 3
    batch_size = 8
    settings = TWO_STAGED_SCENARIOS._replace(
        batch_size=batch_size, worker_count=worker_count, designated_only=False
    )
    now = layout.CURRENT_STEP_INDEX
    expected = 0
    for worker_index in range(worker_count):
        targets = 0
        for path in paths[worker_index::worker_count]:
            with np.load(path) as staged:
                valid_now = staged["track_valid"][:, now]
                type_rows = staged["track_rows"][:, now, layout.AGENT_TYPE]
            is_predicted = type_rows[:, : layout.NUM_OBJECT_TYPES].any(axis=1)
            targets += int((valid_now & is_predicted).sum())
        expected += math.ceil(targets / batch_size)
    assert pipeline.steps_per_epoch(paths, settings) == expected


def test_pruning_fills_six_slots_from_three_modes():
    """With fewer modes than slots, the kept modes match the plain loop and
    the backfilled slots carry no confidence.
    """
    endpoints = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    ramp = torch.linspace(0.0, 1.0, layout.FUTURE_STEPS)[:, None]
    trajectories = (endpoints[:, None, :] * ramp)[None]
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    pruned = pruning.prune_modes(trajectories, logits)
    walked_trajectories, walked_logits = (
        reference_implementations.prune_modes_single_sample(
            trajectories[0], logits[0]
        )
    )
    assert pruned.kept_count.item() == 3
    assert torch.equal(pruned.trajectories[0, :3], walked_trajectories)
    assert torch.equal(pruned.confidence_logits[0, :3], walked_logits)
    submitted = pruning.submitted_modes(trajectories, logits)
    assert torch.allclose(submitted.confidences.sum(dim=-1), torch.ones(1))
    assert torch.all(submitted.confidences[0, 3:] == 0.0)


def test_broken_submission_is_refused(tmp_path):
    """A submission row the scorer would misread must fail on save."""
    slots = layout.NUM_PREDICTED_MODES
    steps = len(layout.SUBMISSION_FUTURE_INDICES)
    confidences = np.zeros((2, slots))
    confidences[:, 0] = 1.0
    good = formats.Submission(
        scenario_id=np.array(["scene", "scene"]),
        track_id=np.array([1, 2], dtype=np.int64),
        world_trajectories=np.zeros((2, slots, steps, 2)),
        confidences=confidences,
        provenance=formats.artifact_provenance("test", tmp_path),
    )
    good.save(tmp_path / "good.npz")
    assert len(formats.Submission.load(tmp_path / "good.npz").track_id) == 2
    negative = confidences.copy()
    negative[0, :2] = [1.5, -0.5]
    not_finite = good.world_trajectories.copy()
    not_finite[1, 0, 0, 0] = np.nan
    broken = [
        good._replace(confidences=negative),
        good._replace(confidences=confidences * 0.5),
        good._replace(track_id=np.array([1, 1], dtype=np.int64)),
        good._replace(world_trajectories=not_finite),
    ]
    for submission in broken:
        with pytest.raises(ValueError):
            submission.save(tmp_path / "broken.npz")


def test_checkpoint_of_another_design_is_refused(tmp_path):
    """Weights from a different model design must be refused by name."""
    anchors = reference_implementations.unit_anchor_offsets_per_type()
    model_state = model.MotionPredictor(anchors).state_dict()
    renamed = {f"old.{name}": value for name, value in model_state.items()}
    renamed[checkpoint.ANCHORS_KEY] = model_state[checkpoint.ANCHORS_KEY]
    path = tmp_path / "other_design.pt"
    stored = {
        "code_version": layout.STAGING_CODE_VERSION,
        "model_state": renamed,
    }
    torch.save(stored, path)
    with pytest.raises(ValueError, match="different model design"):
        checkpoint.load_predictor(path)


EXIT_PROBE = """
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[2])
from womd import formats, frames, layout, pipeline, views
from womd.pipeline import BatchSettings

if __name__ == "__main__":
    paths = sorted(Path(sys.argv[1]).glob("*.npz"))[:1]
    settings = BatchSettings(4, 1, 1, 0, True)
    for batch in pipeline.batches(paths, settings):
        pass
"""


def test_worker_run_exits(tmp_path):
    """A process that iterated a one-worker DataLoader must exit, which on
    macOS needs the resource-tracker pipe kept from torch_shm_manager.
    """
    script = tmp_path / "exit_probe.py"
    script.write_text(EXIT_PROBE)
    repository = Path(__file__).resolve().parents[1]
    arguments = [str(script), str(STAGED_DIRECTORY), str(repository)]
    command = [sys.executable, *arguments]
    finished = subprocess.run(
        command, timeout=60, capture_output=True, check=False
    )
    assert finished.returncode == 0, finished.stderr.decode()[-2000:]
