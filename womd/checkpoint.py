"""Saves and loads checkpoints and anchor files, refusing any artifact the
current code version did not produce.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from womd import formats, layout
from womd.model import MotionPredictor

Checkpoint = dict[str, Any]
Device = str | torch.device
GradScaler = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)
# The anchors live inside every checkpoint as a model buffer.
ANCHORS_KEY = "mode_decoder.unit_anchors"


def load_checkpoint(path: Path | str, device: Device = "cpu") -> Checkpoint:
    """Loads a checkpoint and checks its code version against the working
    tree; load_state_dict then refuses any parameter of the wrong shape.
    """
    checkpoint = torch.load(path, map_location=device)
    code_version = checkpoint.get("code_version")
    if code_version != layout.STAGING_CODE_VERSION:
        raise ValueError(
            f"{path} is from code version {code_version!r}, this tree is"
            f" {layout.STAGING_CODE_VERSION!r}"
        )
    return checkpoint


def check_model_state(model: torch.nn.Module, state: Checkpoint, path) -> None:
    """Refuses weights saved by a different model design. The code version
    stamp follows the staged data, not the model, so it cannot catch this.
    """
    expected = set(model.state_dict())
    stored = set(state)
    missing = sorted(expected - stored)
    unexpected = sorted(stored - expected)
    if missing or unexpected:
        raise ValueError(
            f"{path} holds weights of a different model design:"
            f" {len(missing)} parameters missing, such as {missing[:3]},"
            f" and {len(unexpected)} unexpected, such as {unexpected[:3]}"
        )


def load_predictor(path: Path | str) -> MotionPredictor:
    """The trained model in eval mode, built from the anchors its checkpoint
    holds, so no anchor file can disagree with the weights.
    """
    model_state = load_checkpoint(path)["model_state"]
    if ANCHORS_KEY not in model_state:
        raise ValueError(f"{path} holds no anchors under {ANCHORS_KEY}")
    predictor = MotionPredictor(model_state[ANCHORS_KEY])
    check_model_state(predictor, model_state, path)
    predictor.load_state_dict(model_state)
    return predictor.eval()


def load_anchor_file(path: Path | str) -> torch.Tensor:
    """Loads the fitted per-type anchors from a .npz file and checks their
    provenance; the model checks their shape.
    """
    with np.load(path) as anchors_file:
        stored = set(anchors_file.files)
        provenance = (
            anchors_file["provenance"] if "provenance" in stored else None
        )
        hint = "Refit them with fit_anchors.py."
        checked = formats.check_provenance(provenance, path, hint)
        anchor_fit = checked.get("anchor_fit")
        if anchor_fit != layout.ANCHOR_FIT_VERSION:
            raise ValueError(
                f"{path} is from anchor fit {anchor_fit!r}, this tree fits"
                f" {layout.ANCHOR_FIT_VERSION!r}. {hint}"
            )
        return torch.from_numpy(anchors_file["unit_anchors"])


def previous_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".previous")


def save_checkpoint(checkpoint_path: Path, state: Checkpoint) -> None:
    """Writes to a temporary file, then swaps it in, so a crash never truncates
    the checkpoint; the prior one is kept as a fallback.
    """
    suffix = checkpoint_path.suffix + ".partial"
    partial_path = checkpoint_path.with_suffix(suffix)
    torch.save(state, partial_path)
    if checkpoint_path.exists():
        checkpoint_path.replace(previous_path(checkpoint_path))
    partial_path.replace(checkpoint_path)
