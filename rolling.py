"""Runs the trained model at every scenario step and writes one MCAP per staged
file. Open a recording in Lichtblick and import rolling_layout.json from the
Layouts tab to watch map, agents, futures, truth and the 3 s error plot.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
import functools
import json
import time
from pathlib import Path

import foxglove
import numpy as np
from foxglove.channels import SceneUpdateChannel
from foxglove.messages import Timestamp
from numpy import ndarray

import submit
from live import (
    agents_update,
    color,
    deletion_update,
    futures_update,
    line_strip,
    map_update,
    scene_update,
)
from submit import Predict
from womd import frames, layout, views
from womd.checkpoint import load_predictor
from womd.formats import StagedFile
from womd.pruning import SubmittedModes as Kept

TIMESTEP_NANOSECONDS = int(layout.TIMESTEP_SECONDS * 1_000_000_000)
# A fixed horizon, so the curve moves with what the model knows, not with
# how much of the future is left to miss.
ERROR_HORIZON_SECONDS = 3.0
# Thinner than a future and half transparent, so the orange futures stay
# what is watched and the logged path reads as the reference under them.
TRUTH_LINE_WIDTH = 0.12
TRUTH_OPACITY = 0.5

ERROR_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "min_fde_3s_metres": {"type": "number"},
            "targets": {"type": "integer"},
        },
        "required": ["min_fde_3s_metres", "targets"],
    }
).encode()


def log_timestamp(step: int) -> Timestamp:
    log_time = step * TIMESTEP_NANOSECONDS
    return Timestamp(
        sec=log_time // 1_000_000_000, nsec=log_time % 1_000_000_000
    )


def error_submission_slot() -> int:
    future_index = round(ERROR_HORIZON_SECONDS / layout.TIMESTEP_SECONDS) - 1
    assert future_index in layout.SUBMISSION_FUTURE_INDICES
    return layout.SUBMISSION_FUTURE_INDICES.index(future_index)


def min_fde_3s_metres(shifted: StagedFile, tracks: ndarray, kept: Kept):
    """Mean over tracks of each one's smallest mode error at the fixed 3 s
    horizon, in the scene frame. Returns None when no track has a valid 3 s
    position.
    """
    submission_slot = error_submission_slot()
    horizon_step = layout.CURRENT_STEP_INDEX + round(
        ERROR_HORIZON_SECONDS / layout.TIMESTEP_SECONDS
    )
    errors = []
    for track_index, paths in zip(tracks, kept.trajectories.numpy()):
        if not shifted.track_valid[track_index, horizon_step]:
            continue
        logged = shifted.track_rows[
            track_index, horizon_step, layout.AGENT_POSITION
        ]
        frame = frames.agent_frame(shifted.track_rows[track_index])
        scene_paths = frames.positions_from_frame(paths, frame)
        mode_points = scene_paths[:, submission_slot]
        distances = np.linalg.norm(mode_points - logged, axis=1)
        errors.append(float(distances.min()))
    if not errors:
        return None
    return {"min_fde_3s_metres": float(np.mean(errors)), "targets": len(errors)}


def truth_paths(scenario, tracks: ndarray, step: int) -> list[ndarray]:
    """Each track's logged positions at its valid steps from the step on, in
    the scene frame; a track with none is left out.
    """
    paths = []
    for track_index in tracks:
        valid_steps = step + np.flatnonzero(
            scenario.track_valid[track_index, step:]
        )
        if len(valid_steps) > 0:
            rows = scenario.track_rows[track_index, valid_steps]
            paths.append(rows[:, layout.AGENT_POSITION])
    return paths


def truth_update(scenario, tracks: ndarray, step: int, stamp: Timestamp):
    white = color(1.0, 1.0, 1.0, alpha=TRUTH_OPACITY)
    paths = truth_paths(scenario, tracks, step)
    lines = [line_strip(path, TRUTH_LINE_WIDTH, white) for path in paths]
    return scene_update("truth", stamp, lines=lines)


def recording(scenario, predict: Predict):
    """Yields (topic, log_time_ns, message) in non-decreasing log time."""
    step_ten_targets = views.eligible_tracks(scenario, designated_only=True)
    yield ("/map", 0, map_update(scenario, log_timestamp(0)))
    for step in range(layout.TOTAL_STEPS):
        log_time = step * TIMESTEP_NANOSECONDS
        stamp = log_timestamp(step)
        if step >= layout.CURRENT_STEP_INDEX:
            shifted = views.scene_at_step(scenario, step)
            predict_tracks = views.eligible_tracks(
                shifted, designated_only=True
            )
            orange = predict_tracks
        else:
            orange = step_ten_targets
        yield (
            "/agents",
            log_time,
            agents_update(scenario, orange, step, stamp),
        )
        if step < layout.CURRENT_STEP_INDEX:
            continue
        kept = None
        if len(predict_tracks) == 0:
            yield ("/futures", log_time, deletion_update(stamp))
        else:
            kept = submit.predict_scene(predict, shifted, predict_tracks)
            yield (
                "/futures",
                log_time,
                futures_update(shifted, predict_tracks, kept, stamp),
            )
        yield (
            "/truth",
            log_time,
            truth_update(scenario, predict_tracks, step, stamp),
        )
        if kept is not None:
            error = min_fde_3s_metres(shifted, predict_tracks, kept)
            if error is not None:
                yield ("/error", log_time, error)


def write_scene(scenario, predict: Predict, output_path: Path) -> float:
    """Writes one MCAP and returns seconds taken."""
    context = foxglove.Context()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = foxglove.open_mcap(
        output_path, context=context, allow_overwrite=True
    )
    map_channel = SceneUpdateChannel("/map", context=context)
    agents_channel = SceneUpdateChannel("/agents", context=context)
    futures_channel = SceneUpdateChannel("/futures", context=context)
    truth_channel = SceneUpdateChannel("/truth", context=context)
    error_channel = foxglove.Channel(
        "/error",
        message_encoding="json",
        schema=foxglove.Schema(
            name="error", encoding="jsonschema", data=ERROR_SCHEMA
        ),
        context=context,
    )
    channels = {
        "/map": map_channel,
        "/agents": agents_channel,
        "/futures": futures_channel,
        "/truth": truth_channel,
        "/error": error_channel,
    }
    start = time.perf_counter()
    for topic, log_time, message in recording(scenario, predict):
        channel = channels[topic]
        if topic == "/error":
            channel.log(json.dumps(message).encode(), log_time=log_time)
        else:
            channel.log(message, log_time=log_time)
    writer.close()
    return time.perf_counter() - start


def main() -> None:
    """Writes one MCAP per staged scene with rolling predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("scenes", nargs="+", type=Path)
    arguments = parser.parse_args()
    predictor = load_predictor(arguments.checkpoint)
    predict = functools.partial(submit.model_modes, predictor)
    for path in arguments.scenes:
        scenario = StagedFile.load(path)
        directory = arguments.output_directory
        output_path = directory / f"{scenario.scenario_id}.mcap"
        seconds = write_scene(scenario, predict, output_path)
        print(f"{scenario.scenario_id}: {output_path} in {seconds:.1f} s")


if __name__ == "__main__":
    main()
