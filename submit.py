"""Step 4 of 5.

Runs a trained model over staged scenarios and writes 6 world-frame futures per
designated target.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
import functools
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from womd import baseline, formats, frames, pruning, views
from womd.checkpoint import load_predictor
from womd.formats import StagedFile, Submission
from womd.model import MotionPredictor
from womd.views import TargetInputs

# A prediction function takes a batch and returns every target's modes,
# trajectories (B, modes, 80, 2) and confidence logits (B, modes).
Modes = tuple[Tensor, Tensor]
Predict = Callable[[TargetInputs], Modes]
# A selection picks the indices of the tracks to predict in one scene.
Select = Callable[[StagedFile], ndarray]


def model_modes(predictor: MotionPredictor, batch: TargetInputs) -> Modes:
    predicted = predictor(batch)[-1]
    return predicted.trajectories, predicted.confidence_logits


def constant_velocity_modes(batch: TargetInputs) -> Modes:
    """The control: each target carried on at its current velocity."""
    return baseline.constant_velocity(batch.agent_history[:, 0])


def predict_scene(predict: Predict, scenario: StagedFile, tracks: ndarray):
    """The given tracks' 6 kept modes at Waymo's 2 Hz, in each one's own
    frame. A non-finite mode is refused here, because pruning would turn it
    into confidences that still sum to 1.
    """
    scene = views.scene_inputs(scenario, tracks)
    batch = scene.each(torch.from_numpy)
    with torch.no_grad():
        trajectories, logits = predict(batch)
    finite = torch.isfinite(trajectories).all() & torch.isfinite(logits).all()
    if not finite:
        message = f"{scenario.scenario_id}: the model output is not finite"
        raise ValueError(message)
    return pruning.submitted_modes(trajectories, logits)


def submission(predict: Predict, staged: Path, select: Select) -> Submission:
    """Predicts the tracks `select` picks in each scene, one scene at a time,
    and converts each track's 6 futures to the world frame.
    """
    scenario_ids = []
    track_ids = []
    world_trajectories = []
    confidences = []
    for path in sorted(staged.glob("*.npz")):
        scenario = formats.StagedFile.load(path)
        tracks = select(scenario)
        if len(tracks) == 0:
            continue
        kept = predict_scene(predict, scenario, tracks)
        kept_paths = kept.trajectories.numpy()
        kept_rows = zip(tracks, kept_paths, kept.confidences.numpy())
        for track_index, paths, scores in kept_rows:
            frame = frames.agent_frame(scenario.track_rows[track_index])
            world = frames.to_world(paths, frame, scenario.scene_frame)
            scenario_ids.append(scenario.scenario_id)
            track_ids.append(scenario.track_ids[track_index])
            world_trajectories.append(world)
            confidences.append(scores)
    if not scenario_ids:
        raise ValueError(f"{staged}: no staged scene has a track to predict")
    return Submission(
        scenario_id=np.array(scenario_ids),
        track_id=np.array(track_ids, dtype=np.int64),
        world_trajectories=np.stack(world_trajectories),
        confidences=np.stack(confidences).astype(np.float64),
        provenance=formats.artifact_provenance("submit.py", staged),
    )


def main() -> None:
    """Writes a submission .npz from a trained model or from the constant-
    velocity baseline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--constant-velocity", action="store_true")
    parser.add_argument(
        "--within",
        type=float,
        metavar="METRES",
        help="predict every agent within METRES of the SDC, not the"
        " designated targets",
    )
    parser.add_argument("paths", nargs="+", type=Path)
    arguments = parser.parse_args()
    select = views.designated_tracks
    if arguments.within is not None:
        select = functools.partial(
            views.tracks_within, radius_metres=arguments.within
        )
    if arguments.constant_velocity:
        if len(arguments.paths) != 2:
            raise SystemExit(
                "usage: submit.py --constant-velocity STAGED OUTPUT"
            )
        staged, output = arguments.paths
        predict = constant_velocity_modes
    else:
        if len(arguments.paths) != 3:
            raise SystemExit("usage: submit.py CHECKPOINT STAGED OUTPUT")
        checkpoint_path, staged, output = arguments.paths
        predictor = load_predictor(checkpoint_path)
        predict = functools.partial(model_modes, predictor)

    predicted = submission(predict, staged, select)
    predicted.save(output)
    size = output.stat().st_size
    print(
        f"{len(predicted.track_id)} tracks written to {output}"
        f" ({size} bytes)"
    )


if __name__ == "__main__":
    main()
