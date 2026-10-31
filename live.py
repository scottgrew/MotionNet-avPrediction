"""Plays staged scenes to a Foxglove WebSocket viewer such as Lichtblick at
their own 10 Hz. At the current step the model runs on the scene and each
designated target's 6 kept futures are drawn, then the logged future plays
so the prediction and what the agents did are on screen together.
Import lichtblick_layout.json once in Lichtblick to see the three layers.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
import functools
import time
from pathlib import Path

import foxglove
import numpy as np
from foxglove import messages
from foxglove.channels import SceneUpdateChannel
from foxglove.messages import Timestamp
from numpy import ndarray

import submit
from submit import Predict
from womd import frames, layout, views
from womd.checkpoint import load_predictor
from womd.formats import StagedFile
from womd.pruning import SubmittedModes as Kept

PORT = 8765
FRAME_ID = "scene"
WAIT_SECONDS = 0.1

# Drawing sizes in metres. Staged rows carry no height, so every box gets
# the same one.
BOX_HEIGHT = 1.5
MAP_LINE_WIDTH = 0.15
FUTURE_LINE_WIDTH = 0.3
# A future's opacity is this floor plus the rest scaled by its confidence
# over the target's most confident future.
FUTURE_OPACITY_FLOOR = 0.2


def color(red: float, green: float, blue: float, alpha: float = 1.0):
    return messages.Color(r=red, g=green, b=blue, a=alpha)


# One map colour per entry of layout.MAP_POLYLINE_KINDS, in its order.
MAP_KIND_COLORS = (
    color(0.35, 0.35, 0.35),
    color(0.85, 0.85, 0.85),
    color(0.6, 0.6, 0.6),
    color(0.9, 0.2, 0.2),
    color(0.5, 0.5, 0.8),
    color(0.8, 0.6, 0.2),
    color(0.4, 0.4, 0.3),
)
# One agent colour per column of the type one-hot, vehicle, pedestrian,
# cyclist and then other.
AGENT_TYPE_COLORS = (
    color(0.3, 0.5, 0.9),
    color(0.3, 0.8, 0.4),
    color(0.9, 0.8, 0.3),
    color(0.6, 0.6, 0.6),
)
SELF_DRIVING_CAR_COLOR = color(0.9, 0.9, 0.9)
TARGET_RGB = (1.0, 0.5, 0.1)
TARGET_COLOR = color(*TARGET_RGB)


def flat_pose(position: ndarray, heading: float, height: float):
    """A pose above the ground plane, turned by heading about the vertical."""
    position_3d = messages.Vector3(x=position[0], y=position[1], z=height)
    half_turn = 0.5 * heading
    orientation = messages.Quaternion(
        x=0.0, y=0.0, z=np.sin(half_turn), w=np.cos(half_turn)
    )
    return messages.Pose(position=position_3d, orientation=orientation)


IDENTITY_POSE = flat_pose(np.zeros(2), 0.0, 0.0)


def line_strip(positions: ndarray, width: float, line_color: messages.Color):
    points = [messages.Point3(x=x, y=y, z=0.0) for x, y in positions]
    return messages.LinePrimitive(
        type=messages.LinePrimitiveLineType.LineStrip,
        pose=IDENTITY_POSE,
        thickness=width,
        scale_invariant=False,
        points=points,
        color=line_color,
    )


def scene_update(name: str, stamp: Timestamp, **primitives: list):
    """One scene-frame entity; sending another with the same name replaces
    it in the viewer.
    """
    entity = messages.SceneEntity(
        timestamp=stamp, frame_id=FRAME_ID, id=name, **primitives
    )
    return messages.SceneUpdate(entities=[entity])


def map_update(scenario: StagedFile, stamp: Timestamp) -> messages.SceneUpdate:
    """Every map feature as one line through its dots, coloured by kind."""
    split_at = np.cumsum(scenario.feature_lengths)[:-1]
    lines = []
    for feature_rows in np.split(scenario.map_rows, split_at):
        kind = feature_rows[0, layout.MAP_KIND].argmax()
        positions = feature_rows[:, layout.MAP_POSITION]
        kind_color = MAP_KIND_COLORS[kind]
        lines.append(line_strip(positions, MAP_LINE_WIDTH, kind_color))
    return scene_update("map", stamp, lines=lines)


def agent_color(rows: ndarray, is_target: bool) -> messages.Color:
    if is_target:
        return TARGET_COLOR
    if rows[layout.AGENT_IS_SDC] == 1.0:
        return SELF_DRIVING_CAR_COLOR
    return AGENT_TYPE_COLORS[rows[layout.AGENT_TYPE].argmax()]


def agents_update(scenario, tracks: ndarray, step: int, stamp: Timestamp):
    """A box for every track valid at the step, the predicted tracks in
    orange.
    """
    cubes = []
    for track_index in np.flatnonzero(scenario.track_valid[:, step]):
        rows = scenario.track_rows[track_index, step]
        sine = rows[layout.AGENT_HEADING_SINE]
        cosine = rows[layout.AGENT_HEADING_COSINE]
        heading = np.arctan2(sine, cosine)
        center = rows[layout.AGENT_POSITION]
        pose = flat_pose(center, heading, 0.5 * BOX_HEIGHT)
        length, width = rows[layout.AGENT_DIMENSIONS]
        size = messages.Vector3(x=length, y=width, z=BOX_HEIGHT)
        box_color = agent_color(rows, track_index in tracks)
        box = messages.CubePrimitive(pose=pose, size=size, color=box_color)
        cubes.append(box)
    return scene_update("agents", stamp, cubes=cubes)


def futures_update(scenario, tracks: ndarray, kept: Kept, stamp: Timestamp):
    """Each predicted track's 6 futures in the scene frame, starting at its
    current position, more opaque the more confident.
    """
    floor = FUTURE_OPACITY_FLOOR
    lines = []
    kept_rows = zip(tracks, kept.trajectories.numpy(), kept.confidences)
    for track_index, paths, confidences in kept_rows:
        frame = frames.agent_frame(scenario.track_rows[track_index])
        scene_paths = frames.positions_from_frame(paths, frame)
        relative_confidences = confidences / confidences.max()
        for path, confidence in zip(scene_paths, relative_confidences):
            start = frame.origin[None]
            positions = np.concatenate([start, path])
            opacity = floor + (1.0 - floor) * float(confidence)
            line_color = color(*TARGET_RGB, alpha=opacity)
            line = line_strip(positions, FUTURE_LINE_WIDTH, line_color)
            lines.append(line)
    return scene_update("futures", stamp, lines=lines)


def deletion_update(stamp: Timestamp) -> messages.SceneUpdate:
    deletion = messages.SceneEntityDeletion(
        timestamp=stamp, type=messages.SceneEntityDeletionType.All, id=""
    )
    return messages.SceneUpdate(deletions=[deletion])


class LiveView:
    """Owns the viewer connection and one channel per layer, and plays scenes
    through them at the logged pace.
    """

    def __init__(self, predict: Predict) -> None:
        self.predict = predict
        self.server = foxglove.start_server(port=PORT)
        self.map_channel = SceneUpdateChannel("/map")
        self.agents_channel = SceneUpdateChannel("/agents")
        self.futures_channel = SceneUpdateChannel("/futures")

    def wait_for_viewer(self) -> None:
        print(f"waiting for a viewer on ws://127.0.0.1:{PORT}")
        while not self.map_channel.has_sinks():
            time.sleep(WAIT_SECONDS)

    def predict_now(self, scenario: StagedFile, tracks: ndarray) -> None:
        start = time.perf_counter()
        kept = submit.predict_scene(self.predict, scenario, tracks)
        milliseconds = 1000 * (time.perf_counter() - start)
        now = messages.Timestamp.now()
        self.futures_channel.log(futures_update(scenario, tracks, kept, now))
        print(
            f"{scenario.scenario_id}: {len(tracks)} targets predicted in"
            f" {milliseconds:.0f} ms"
        )

    def play(self, scenario: StagedFile) -> None:
        """Sends the scene's steps, each due 0.1 s after the last; a step
        that falls behind is sent at once. The model runs at the current
        step on the designated targets.
        """
        tracks = views.designated_tracks(scenario)
        now = messages.Timestamp.now()
        self.futures_channel.log(deletion_update(now))
        self.map_channel.log(map_update(scenario, now))
        start = time.monotonic()
        step_count = scenario.track_valid.shape[1]
        for step in range(step_count):
            due = start + step * layout.TIMESTEP_SECONDS
            time.sleep(max(0.0, due - time.monotonic()))
            step_time = messages.Timestamp.now()
            self.agents_channel.log(
                agents_update(scenario, tracks, step, step_time)
            )
            if step == layout.CURRENT_STEP_INDEX:
                self.predict_now(scenario, tracks)


def main() -> None:
    """Waits for a viewer, plays each staged scene in name order, then
    stops.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("staged_directory", type=Path)
    arguments = parser.parse_args()
    predictor = load_predictor(arguments.checkpoint)
    predict = functools.partial(submit.model_modes, predictor)
    view = LiveView(predict)
    view.wait_for_viewer()
    for path in sorted(arguments.staged_directory.glob("*.npz")):
        view.play(StagedFile.load(path))
    view.server.stop()


if __name__ == "__main__":
    main()
