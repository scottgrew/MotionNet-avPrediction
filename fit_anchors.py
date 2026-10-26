"""Step 2 of 5.

Clusters where training agents ended up after 8 s into 54 anchor endpoints per
object type.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Tensor

from womd import formats, frames, layout, model, views

# Fixed, so a refit on the same endpoints gives the same anchors.
# Every other k-means setting is scikit-learn's default.
RANDOM_STATE_SEED = 0


def reachable_endpoints(paths: list[Path]) -> tuple[Tensor, Tensor]:
    """Every designated target's position at 8 s (N, 2) in its own frame,
    and its predicted type (N,). An anchor is where an agent is at 8 s, so a
    track that ends sooner is left out of the fit, as is an endpoint further
    away than the agent could have driven; both stay training samples.
    """
    histories = []
    positions = []
    ended_early = 0
    for path in paths:
        scenario = formats.StagedFile.load(path)
        eligible = views.eligible_tracks(scenario, designated_only=True)
        for track_index in eligible:
            if not scenario.track_valid[track_index, -1]:
                ended_early += 1
                continue
            rows = scenario.track_rows[track_index]
            own_rows = frames.rows_to_frame(rows, frames.agent_frame(rows))
            histories.append(own_rows[layout.HISTORY])
            positions.append(own_rows[-1, layout.AGENT_POSITION])
    history = torch.from_numpy(np.stack(histories))
    endpoints = torch.from_numpy(np.stack(positions))
    type_index = model.predicted_type_index(history)
    reachable_metres = model.agent_reachable_distance(history)
    reachable = endpoints.norm(dim=-1) <= reachable_metres
    print(
        f"{len(endpoints)} endpoints; {ended_early} tracks ending before 8 s"
        f" and {int((~reachable).sum())} endpoints beyond reachable distance"
        f" left out of the fit"
    )
    return endpoints[reachable], type_index[reachable]


def fit_type(type_name: str, positions: Tensor) -> tuple[Tensor, Tensor]:
    """Fits one object type's anchors (54, 2) with k-means and returns them
    most-used first, with how many endpoints each drew (54,).
    """
    if len(positions) < model.QUERY_COUNT:
        raise ValueError(
            f"{type_name} has {len(positions)} endpoints,"
            f" fewer than {model.QUERY_COUNT} anchors"
        )
    kmeans = KMeans(
        n_clusters=model.QUERY_COUNT, random_state=RANDOM_STATE_SEED
    )
    fitted = kmeans.fit(positions.numpy())
    assignment = torch.from_numpy(fitted.labels_).to(torch.long)
    counts = torch.zeros(model.QUERY_COUNT, dtype=torch.long)
    counts.index_add_(0, assignment, torch.ones_like(assignment))
    order = torch.argsort(counts, descending=True, stable=True)
    centres = torch.from_numpy(fitted.cluster_centers_).to(positions.dtype)
    stop_reason = "convergence"
    if fitted.n_iter_ >= fitted.max_iter:
        stop_reason = "the iteration cap"
    separations = torch.cdist(centres, centres)
    separations.fill_diagonal_(float("inf"))
    print(
        f"{type_name} | samples {len(positions)}"
        f" | k-means ran {fitted.n_iter_} iterations, stopped by {stop_reason}"
        f" | largest share {int(counts.max()) / len(positions):.1%}"
        f" | closest two anchors {float(separations.min()):.3f} m"
    )
    return centres[order], counts[order]


def main() -> None:
    """Fits one set of 54 anchor endpoints per object type from the training
    futures and writes them with a provenance stamp.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("output_path", type=Path)
    arguments = parser.parse_args()

    paths = sorted(arguments.staged_directory.glob("*.npz"))
    endpoints, type_index = reachable_endpoints(paths)
    per_type_anchors = []
    for index, type_name in enumerate(layout.PREDICTED_OBJECT_TYPES):
        anchors, _ = fit_type(type_name, endpoints[type_index == index])
        per_type_anchors.append(anchors)

    unit_anchors = torch.stack(per_type_anchors)
    provenance = formats.anchor_provenance(arguments.staged_directory)
    np.savez(
        arguments.output_path,
        unit_anchors=unit_anchors.numpy().astype(np.float32),
        provenance=provenance,
    )
    print(
        f"wrote {arguments.output_path}, one set of anchors per"
        f" {layout.PREDICTED_OBJECT_TYPES} in that order, most-used first"
    )


if __name__ == "__main__":
    main()
