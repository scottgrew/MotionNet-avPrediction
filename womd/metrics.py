"""Training-time monitor: losses, minADE and minFDE on the kept modes, how many
anchors never win, and optimiser health.

It steers runs; reported numbers come only from Waymo's scorer.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import Tensor

from womd import layout
from womd.loss import LossTerms, mean_over_valid_steps
from womd.model import QUERY_COUNT, ModePredictions
from womd.pruning import prune_modes
from womd.views import TargetInputs

# A log line is printed once per this many batches, and the pruned
# minADE and minFDE run on that batch only, because pruning is slow.
LOG_EVERY_BATCHES = 20

# Waymo's three scoring horizons, as counts of 10 Hz future steps, and the
# predicted types in their one-hot order (layout.PREDICTED_OBJECT_TYPES).
HORIZON_STEPS = (("3s", 30), ("5s", 50), ("8s", 80))
TYPE_NAMES = ("vehicle", "pedestrian", "cyclist")

Scalars = dict[str, float]


def mean_or_nan(total: Tensor | float, count: Tensor | int) -> float:
    """A running sum over its count, or nan when nothing was counted."""
    return float(total / count) if count else float("nan")


def mode_distances(trajectories: Tensor, targets: TargetInputs) -> Tensor:
    """Each mode's mean distance to the logged future over its valid steps."""
    logged = targets.future_positions.unsqueeze(1)
    step_distances = (trajectories - logged).norm(dim=-1)
    return mean_over_valid_steps(step_distances, targets.future_mask)


class EpochProgress:
    """Running totals for one epoch: losses, optimiser health, samples seen,
    minADE and minFDE over the 6 kept modes of every LOG_EVERY_BATCHES-th
    batch, and which anchors' modes came closest since the last log line, to
    spot anchors that never win.
    """

    def __init__(self, device: torch.device) -> None:
        self.start = time.perf_counter()
        self.batch_count = 0
        self.finite_batch_count = 0
        self.sample_count = 0
        self.loss_sums = np.zeros(len(LossTerms._fields))
        self.ade_sum = 0.0
        self.ade_count = 0
        self.fde_sum = 0.0
        self.fde_count = 0
        self.non_finite_count = 0
        self.skipped_count = 0
        self.clipped_count = 0
        self.data_wait_seconds = 0.0
        self.winner_counts = torch.zeros(
            QUERY_COUNT, dtype=torch.long, device=device
        )

    def record(self, losses: LossTerms, clipped: bool, skipped: bool) -> bool:
        """Folds one step's losses and optimiser health into the totals and
        returns whether its total loss was finite.
        """
        values = [float(term.detach()) for term in losses]
        self.batch_count += 1
        loss_is_finite = bool(np.isfinite(values).all())
        if loss_is_finite:
            self.loss_sums += values
            self.finite_batch_count += 1
        self.non_finite_count += int(not loss_is_finite)
        self.clipped_count += int(clipped)
        self.skipped_count += int(skipped)
        return bool(loss_is_finite)

    def monitor_batch(self, predicted: ModePredictions, targets: TargetInputs):
        """Counts the batch's samples and closest modes, and on every
        LOG_EVERY_BATCHES-th batch adds it to the pruned minADE and minFDE.
        """
        self.sample_count += targets.agent_history.shape[0]
        detached = (tensor.detach().float() for tensor in predicted)
        predicted = ModePredictions(*detached)
        with torch.no_grad():
            distances = mode_distances(predicted.trajectories, targets)
            winners = distances.argmin(dim=1)
            self.winner_counts.scatter_add_(
                0, winners, torch.ones_like(winners)
            )
            if self.batch_count % LOG_EVERY_BATCHES == 0:
                self.add_kept_mode_errors(predicted, targets)

    def add_kept_mode_errors(self, predicted: ModePredictions, targets):
        logits = predicted.confidence_logits
        pruned = prune_modes(predicted.trajectories, logits)
        self.add_pruned_errors(pruned.trajectories, targets)

    def add_pruned_errors(self, kept: Tensor, targets: TargetInputs) -> None:
        """Adds minADE and minFDE over modes already pruned to the 6 kept."""
        logged = targets.future_positions.unsqueeze(1)
        distances = (kept - logged).norm(dim=-1)
        average_distances = mode_distances(kept, targets)

        has_valid_step = targets.future_mask.any(dim=-1)
        best_average = average_distances.min(dim=-1).values
        self.ade_sum += (best_average * has_valid_step).sum()
        self.ade_count += has_valid_step.sum()

        final_valid = targets.future_mask[:, -1]
        best_final = distances[:, :, -1].min(dim=-1).values
        self.fde_sum += (best_final * final_valid).sum()
        self.fde_count += final_valid.sum()

    def loss_means(self, prefix: str) -> Scalars:
        """Mean losses over the finite steps; NaN when no step was finite.
        A skipped non-finite step changed no weight, so it is left out.
        """
        names = [prefix + name for name in LossTerms._fields]
        if self.finite_batch_count == 0:
            return dict.fromkeys(names, math.nan)
        means = self.loss_sums / self.finite_batch_count
        return dict(zip(names, means.tolist()))

    def health_scalars(self) -> Scalars:
        return {
            "health/non_finite_losses": self.non_finite_count,
            "health/skipped_steps": self.skipped_count,
            "health/clipped_steps": self.clipped_count,
        }

    def log_scalars(self, learning_rate: float) -> Scalars:
        """The table printed every LOG_EVERY_BATCHES, after which the
        never-win count starts again.
        """
        never_win_count = int((self.winner_counts == 0).sum())
        self.winner_counts.zero_()
        elapsed = time.perf_counter() - self.start
        return {
            **self.loss_means("loss/"),
            "monitor/ade_80step": mean_or_nan(self.ade_sum, self.ade_count),
            "monitor/fde_80step": mean_or_nan(self.fde_sum, self.fde_count),
            "monitor/never_win_anchors": never_win_count,
            **self.health_scalars(),
            "optimisation/learning_rate": learning_rate,
            "throughput/samples_per_second": self.sample_count / elapsed,
            "time_share/data_wait": self.data_wait_seconds / elapsed,
        }

    def epoch_scalars(self) -> Scalars:
        return {
            **self.loss_means("epoch/loss_"),
            "epoch/ade_80step": mean_or_nan(self.ade_sum, self.ade_count),
            "epoch/fde_80step": mean_or_nan(self.fde_sum, self.fde_count),
            **self.health_scalars(),
            "epoch/seconds": time.perf_counter() - self.start,
        }


def target_types(targets: TargetInputs) -> Tensor:
    """Each target's object type index, from the type one-hot of its own row
    (agent slot 0) at the current step.
    """
    now = targets.agent_history[:, 0, layout.CURRENT_STEP_INDEX]
    type_one_hot = now[:, layout.AGENT_TYPE][:, : layout.NUM_OBJECT_TYPES]
    return type_one_hot.argmax(dim=-1)


class TypeHorizonErrors:
    """minADE and minFDE over the 6 kept modes for each object type at Waymo's
    3, 5 and 8 s horizons, summed over a pass, so a type falling behind
    shows during training. Averages are over each horizon's valid steps, as
    the monitor's 80-step numbers are.
    """

    def __init__(self) -> None:
        shape = (layout.NUM_OBJECT_TYPES, len(HORIZON_STEPS))
        self.ade_sum = np.zeros(shape)
        self.ade_count = np.zeros(shape)
        self.fde_sum = np.zeros(shape)
        self.fde_count = np.zeros(shape)
        self.target_count = np.zeros(layout.NUM_OBJECT_TYPES)

    def add(self, kept: Tensor, targets: TargetInputs) -> None:
        """Folds in one batch's kept modes (B, 6, 80, 2)."""
        type_index = target_types(targets)
        by_type = torch.nn.functional.one_hot(
            type_index, layout.NUM_OBJECT_TYPES
        ).T.double()
        self.target_count += by_type.sum(dim=1).cpu().numpy()
        logged = targets.future_positions.unsqueeze(1)
        gaps = (kept - logged).norm(dim=-1)
        for column, (_, steps) in enumerate(HORIZON_STEPS):
            mask = targets.future_mask[:, :steps]
            average = mean_over_valid_steps(gaps[..., :steps], mask)
            best_average = average.min(dim=-1).values.double()
            has_valid_step = mask.any(dim=-1).double()
            best_final = gaps[:, :, steps - 1].min(dim=-1).values.double()
            final_valid = mask[:, -1].double()
            sums = [
                by_type @ (best_average * has_valid_step),
                by_type @ has_valid_step,
                by_type @ (best_final * final_valid),
                by_type @ final_valid,
            ]
            ade_sum, ade_count, fde_sum, fde_count = (
                total.cpu().numpy() for total in sums
            )
            self.ade_sum[:, column] += ade_sum
            self.ade_count[:, column] += ade_count
            self.fde_sum[:, column] += fde_sum
            self.fde_count[:, column] += fde_count

    def pooled_ade(self, horizon: str) -> float:
        """minADE at one horizon over every type together."""
        column = [label for label, _ in HORIZON_STEPS].index(horizon)
        total = self.ade_sum[:, column].sum()
        return mean_or_nan(total, self.ade_count[:, column].sum())

    def scalars(self, prefix: str) -> Scalars:
        """One table per type, e.g. prefix + "vehicle/ade_3s"."""
        table = {}
        for row, type_name in enumerate(TYPE_NAMES):
            group = f"{prefix}{type_name}/"
            table[group + "targets"] = int(self.target_count[row])
            for column, (label, _) in enumerate(HORIZON_STEPS):
                ade = self.ade_sum[row, column], self.ade_count[row, column]
                fde = self.fde_sum[row, column], self.fde_count[row, column]
                table[f"{group}ade_{label}"] = mean_or_nan(*ade)
                table[f"{group}fde_{label}"] = mean_or_nan(*fde)
        return table
