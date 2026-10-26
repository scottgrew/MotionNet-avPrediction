"""The two files passed between steps, the staged scenario and the
submission: reading, writing, validating and the provenance stamp that
ties each file to the code version that wrote it.
"""

from __future__ import annotations

import io
import json
import zipfile
import zlib
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy import ndarray

from womd.frames import Frame
from womd.layout import (
    ANCHOR_FIT_VERSION,
    NUM_PREDICTED_MODES,
    STAGING_CODE_VERSION,
    SUBMISSION_FUTURE_INDICES,
)


def artifact_provenance(producer: str, source: Path | str) -> str:
    """The JSON stamp saved into each artifact, naming the code version, the
    script that wrote it and what it was made from.
    """
    stamp = {
        "code_version": STAGING_CODE_VERSION,
        "producer": producer,
        "source": str(source),
    }
    return json.dumps(stamp)


def anchor_provenance(source: Path | str) -> str:
    """fit_anchors.py's stamp, which also names the anchor fit version."""
    stamp = json.loads(artifact_provenance("fit_anchors.py", source))
    return json.dumps({**stamp, "anchor_fit": ANCHOR_FIT_VERSION})


def check_provenance(stamp: Any, path: Path | str, hint: str) -> dict[str, str]:
    """Raises if an artifact has no stamp or was written by another code
    version; hint says how to regenerate it.
    """
    if stamp is None:
        raise ValueError(f"{path} has no provenance stamp. {hint}")
    try:
        provenance = json.loads(str(stamp))
        code_version = provenance["code_version"]
    except (ValueError, KeyError, TypeError) as error:
        message = f"{path} has an unreadable provenance stamp ({error}). {hint}"
        raise ValueError(message) from error
    if code_version != STAGING_CODE_VERSION:
        raise ValueError(
            f"{path} is from code version {code_version!r},"
            f" this tree is {STAGING_CODE_VERSION!r}. {hint}"
        )
    return provenance


def read_in_one_call(path: Path | str) -> io.BytesIO:
    """The whole file from one read call, for np.load to parse in memory.
    np.load on a path makes a seek and a read per zip member, and each of
    those calls waits on Kaggle's /kaggle/input mount.
    """
    file_bytes = Path(path).read_bytes()
    return io.BytesIO(file_bytes)


# What numpy and zipfile raise on a truncated, empty or corrupt archive, or
# on a file that is not an .npz archive at all.
ARCHIVE_ERRORS = (
    zipfile.BadZipFile,
    zlib.error,
    EOFError,
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
)


def read_staged_arrays(path: Path | str, record: Any) -> dict[str, np.ndarray]:
    """The arrays a staged file holds for each field of a NamedTuple record;
    fields with a default are optional. A file that is not a readable
    archive, or lacks a required array, raises an error naming the file, so
    one bad file among 122,352 can be found.
    """
    in_memory = read_in_one_call(path)
    required = [
        name for name in record._fields if name not in record._field_defaults
    ]
    try:
        with np.load(in_memory) as staged:
            stored = set(staged.files)
            present = [name for name in record._fields if name in stored]
            arrays = {name: staged[name] for name in present}
    except ARCHIVE_ERRORS as error:
        message = f"{path} is not a readable staged file ({error!r})"
        raise ValueError(message) from error
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"{path} lacks the staged arrays {missing}")
    return arrays


class StagedFile(NamedTuple):
    """One staged scenario as stage.py writes it, every array in the scene
    frame. T tracks, F map features, D map dots, 91 steps. Files staged by
    older code carry five extra lane-topology arrays and no provenance stamp;
    load reads only these fields and checks the stamp when present.
    """

    track_rows: np.ndarray  # (T, 91, 13) float32, agent rows
    track_valid: np.ndarray  # (T, 91) bool, step observed
    track_ids: np.ndarray  # (T,) int64
    is_designated_target: np.ndarray  # (T,) bool, Waymo scores it
    is_object_of_interest: np.ndarray  # (T,) bool
    map_rows: np.ndarray  # (D, 32) float32, dots grouped by feature
    feature_lengths: np.ndarray  # (F,) int64, dots per feature
    feature_ids: np.ndarray  # (F,) int64
    feature_is_interpolating: np.ndarray  # (F,) bool
    polyline_signal_histories: np.ndarray  # (F, 11, 9) float32
    frame_origin: np.ndarray  # (2,) float32, the scene frame in the world
    frame_heading: np.ndarray  # () float32
    scenario_id: np.ndarray  # () str
    provenance: np.ndarray | None = None  # () str

    @property
    def scene_frame(self) -> Frame:
        return Frame(self.frame_origin, float(self.frame_heading))

    def save(self, path: Path) -> None:
        """Writes to a .partial path and renames it, so a killed process
        never leaves a half-written file in place.
        """
        partial_path = path.with_suffix(path.suffix + ".partial")
        with open(partial_path, "wb") as partial_file:
            np.savez_compressed(partial_file, **self._asdict())
        partial_path.replace(path)

    @classmethod
    def load(cls, path: Path | str) -> StagedFile:
        arrays = read_staged_arrays(path, cls)
        if "provenance" in arrays:
            hint = "Restage the directory with stage.py."
            check_provenance(arrays["provenance"], path, hint)
        dot_count = int(arrays["feature_lengths"].sum())
        row_count = len(arrays["map_rows"])
        if dot_count != row_count:
            raise ValueError(
                f"{path}: feature_lengths sum to {dot_count}"
                f" over {row_count} map rows"
            )
        return cls(**arrays)


# Confidences come from a float32 softmax, so their sums sit within about
# 1e-7 of 1; a chosen value.
CONFIDENCE_SUM_TOLERANCE = 1e-5


class Submission(NamedTuple):
    """Six world-frame futures per designated target at 2 Hz, one row per
    target, as submit.py writes them and the container scores them.
    """

    scenario_id: np.ndarray  # (targets,) str
    track_id: np.ndarray  # (targets,) int64
    world_trajectories: np.ndarray  # (targets, 6, 16, 2) float64 metres
    confidences: np.ndarray  # (targets, 6) float64, summing to 1
    provenance: np.ndarray  # () str

    def save(self, path: Path) -> None:
        self.check(path)
        np.savez_compressed(path, **self._asdict())

    @classmethod
    def load(cls, path: Path | str) -> Submission:
        with np.load(path) as stored:
            arrays = {name: stored[name] for name in stored.files}
        hint = "Regenerate the predictions with submit.py."
        check_provenance(arrays.get("provenance"), path, hint)
        submission = cls(**arrays)
        submission.check(path)
        return submission

    def check(self, path: Path | str) -> None:
        """Raises unless every row is one target's 6 finite futures with
        confidences that are non-negative and sum to 1, each target once.
        """
        target_count = len(self.track_id)
        expected_shape = (target_count, NUM_PREDICTED_MODES)
        trajectory_shape = expected_shape + (len(SUBMISSION_FUTURE_INDICES), 2)
        problems = []
        if self.scenario_id.shape != (target_count,):
            problems.append(f"scenario_id shape {self.scenario_id.shape}")
        if self.world_trajectories.shape != trajectory_shape:
            problems.append(f"trajectory shape {self.world_trajectories.shape}")
        if self.confidences.shape != expected_shape:
            problems.append(f"confidence shape {self.confidences.shape}")
        if problems:
            raise ValueError(f"{path}: {', '.join(problems)}")
        if not np.isfinite(self.world_trajectories).all():
            problems.append("non-finite trajectories")
        if not np.isfinite(self.confidences).all():
            problems.append("non-finite confidences")
        if (self.confidences < 0).any():
            problems.append("negative confidences")
        confidence_sums = self.confidences.sum(axis=1)
        if not np.allclose(confidence_sums, 1.0, atol=CONFIDENCE_SUM_TOLERANCE):
            problems.append("confidences that do not sum to 1")
        targets = set(zip(self.scenario_id.tolist(), self.track_id.tolist()))
        if len(targets) != target_count:
            problems.append(f"{target_count - len(targets)} repeated targets")
        if problems:
            raise ValueError(f"{path}: {', '.join(problems)}")


class TrackArrays(NamedTuple):
    """The three staged arrays that decide which tracks are predicted."""

    track_rows: ndarray  # (T, 91, 13)
    track_valid: ndarray  # (T, 91)
    is_designated_target: ndarray  # (T,)

    @classmethod
    def load(cls, path: Path | str) -> TrackArrays:
        """Decompresses only these three arrays and skips the provenance
        check, for the step count over every training file.
        """
        return cls(**read_staged_arrays(path, cls))
