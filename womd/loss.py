"""The two loss terms, the Gaussian likelihood of the assigned anchor's path
and cross-entropy over which anchor that was.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

from womd.model import ModePredictions, RoundPredictions, anchor_lines
from womd.views import TargetInputs

# The constant term of a Gaussian's negative log likelihood, per
# dimension.
HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)


class LossTerms(NamedTuple):
    """The total loss and the two likelihood terms it is the sum of."""

    total: Tensor
    regression: Tensor
    classification: Tensor


def mean_over_valid_steps(per_step: Tensor, mask: Tensor) -> Tensor:
    """Averages a (batch, modes, steps) quantity over each sample's valid
    future steps; a sample with no valid step averages to zero.
    """
    validity = mask.unsqueeze(1).to(per_step.dtype)
    valid_step_count = validity.sum(dim=-1).clamp_min(1.0)
    return (per_step * validity).sum(dim=-1) / valid_step_count


def observed_future(targets: TargetInputs) -> Tensor:
    """The logged future with invalid steps replaced by zeros. Masking a NaN
    or inf step after the NLL would still leave 0 * inf in the gradient.
    """
    step_is_valid = targets.future_mask.unsqueeze(-1)
    future = targets.future_positions
    zeros = torch.zeros_like(future)
    return torch.where(step_is_valid, future, zeros)


def anchor_assigned_mode(anchors: Tensor, targets: TargetInputs) -> Tensor:
    """Assigns each sample the anchor whose constant-speed straight path is
    closest on average to the logged future.
    """
    future = observed_future(targets)
    anchor_paths = anchor_lines(anchors.to(future.dtype))
    step_distances = (anchor_paths - future.unsqueeze(1)).norm(dim=-1)
    mean_distance = mean_over_valid_steps(step_distances, targets.future_mask)
    return mean_distance.argmin(dim=1)


def gaussian_nll(predicted: ModePredictions, target: Tensor) -> Tensor:
    """Per-step, per-mode Gaussian negative log likelihood against the target
    trajectory, with a diagonal covariance parameterised by log sigma.
    """
    log_sigma = predicted.log_standard_deviation
    error = predicted.trajectories - target.unsqueeze(1)
    # Per axis the NLL is log sigma + (error / sigma)^2 / 2 + log(2 pi) / 2.
    standardised_error = error / log_sigma.exp()
    per_axis = log_sigma + 0.5 * standardised_error**2 + HALF_LOG_TWO_PI
    return per_axis.sum(dim=-1)


def loss_terms(predicted: ModePredictions, targets, assigned_mode=None):
    """Combines the assigned anchor's Gaussian NLL with cross-entropy over
    anchors for a TargetInputs batch, averaged only over samples with a
    valid future. The assignment (batch,) is computed here unless the caller
    already has it.
    """
    future = observed_future(targets)
    step_nll = gaussian_nll(predicted, future)
    mode_nll = mean_over_valid_steps(step_nll, targets.future_mask)
    scoreable = targets.future_mask.any(dim=-1).to(step_nll.dtype)
    scoreable_count = scoreable.sum().clamp_min(1.0)
    if assigned_mode is None:
        assigned_mode = anchor_assigned_mode(predicted.anchors, targets)

    # Only the assigned anchor's per-mode NLL counts toward the
    # regression term.
    assigned_nll = mode_nll.gather(1, assigned_mode[:, None]).squeeze(1)
    regression = (assigned_nll * scoreable).sum() / scoreable_count
    cross_entropy = torch.nn.functional.cross_entropy(
        predicted.confidence_logits, assigned_mode, reduction="none"
    )
    classification = (cross_entropy * scoreable).sum() / scoreable_count
    return LossTerms(regression + classification, regression, classification)


def round_loss_terms(rounds: RoundPredictions, targets: TargetInputs):
    """Every decoder round's loss, averaged over the rounds, so each round
    must give a usable answer and the total keeps one round's scale. Every
    round shares the same anchors, so the assignment is computed once.
    """
    assigned_mode = anchor_assigned_mode(rounds[-1].anchors, targets)
    per_round = [
        loss_terms(predicted, targets, assigned_mode) for predicted in rounds
    ]
    averaged = [torch.stack(terms).mean() for terms in zip(*per_round)]
    return LossTerms(*averaged)
