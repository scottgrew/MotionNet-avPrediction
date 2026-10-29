"""Step 3 of 5.

Trains the model on staged scenarios, on one device or, launched by torchrun,
one process per GPU; resumable, with a time budget so a Kaggle session stops
cleanly before it is cut off.
"""

from __future__ import annotations

import womd.runtime_env  # noqa: F401

import argparse
import contextlib
import faulthandler
import hashlib
import math
import os
import signal
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, distributed, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from womd import layout, loss, pipeline
from womd.checkpoint import (
    ANCHORS_KEY,
    GradScaler,
    check_model_state,
    load_anchor_file,
    load_checkpoint,
    previous_path,
    save_checkpoint,
)
from womd.metrics import (
    LOG_EVERY_BATCHES,
    EpochProgress,
    Scalars,
    TypeHorizonErrors,
    mean_or_nan,
)
from womd.model import MotionPredictor
from womd.pipeline import BatchSettings
from womd.pruning import prune_modes
from womd.views import TargetInputs

# The peak learning rate unless --learning-rate sets another, and
# AdamW's own default weight decay.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
SECONDS_PER_HOUR = 3600
STACK_DUMP_SECONDS = 1800

Batches = Iterable[TargetInputs]


class HoldoutScores(NamedTuple):
    """The held-out pass's pooled table and its per-type, per-horizon one."""

    overall: Scalars
    by_type: Scalars


class EpochPlan(NamedTuple):
    """The optimiser steps in one epoch and the settings that fix it. Counting
    opens every staged file, so a checkpoint stores the plan and a resumed run
    with the same settings reuses it.
    """

    scenario_names_digest: str
    process_count: int
    worker_count: int
    batch_size: int
    designated_only: bool
    steps_per_epoch: int

    @classmethod
    def stored(cls, record: dict[str, Any] | None) -> EpochPlan | None:
        """The plan a checkpoint holds, or None when it holds none or was
        written with other fields.
        """
        if record is None or set(record) != set(cls._fields):
            return None
        return cls(**record)


class Schedule(NamedTuple):
    """The learning rate as a function of the step count alone: a linear
    warm-up, a hold, then an optional cosine fall to zero.
    """

    peak: float
    warmup_steps: int
    decay_start_step: int | None
    decay_end_step: int | None

    def at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak * (step + 1) / self.warmup_steps
        if self.decay_start_step is None or step < self.decay_start_step:
            return self.peak
        decay_length = max(self.decay_end_step - 1 - self.decay_start_step, 1)
        fraction = min((step - self.decay_start_step) / decay_length, 1.0)
        # The rate is peak * (1 + cos(pi f)) / 2 for fraction f through the
        # fall.
        return self.peak * 0.5 * (1.0 + math.cos(math.pi * fraction))


class PlateauState(NamedTuple):
    """Where the flat phase stands, stored in the checkpoint so a resumed
    session carries on the same count. flat_epochs is set when the phase
    ends, fall_epochs once the fall has begun, after which its length is
    fixed.
    """

    best_ade: float
    epochs_without_improvement: int
    flat_epochs: int | None
    fall_epochs: int | None

    @classmethod
    def fresh(cls) -> PlateauState:
        return cls(math.inf, 0, None, None)


class PlateauRule(NamedTuple):
    """Ends the flat phase after an epoch once held-out minADE has gone
    patience epochs without beating its best by min_improvement metres,
    never before min_flat_epochs and always at max_flat_epochs. The best
    moves only on a counted improvement, so neither a noisy bump nor a run
    of tiny gains restarts the wait.
    """

    patience: int
    min_improvement: float
    min_flat_epochs: int
    max_flat_epochs: int

    def after_epoch(self, state: PlateauState, epoch: int, ade: float):
        """The state after 1-based epoch scored held-out minADE ade."""
        if state.flat_epochs is not None:
            return state
        # A NaN compares false, so an unmeasured epoch counts as no gain.
        improved = ade <= state.best_ade - self.min_improvement
        waiting = 0 if improved else state.epochs_without_improvement + 1
        stalled = waiting >= self.patience and epoch >= self.min_flat_epochs
        ends = stalled or epoch >= self.max_flat_epochs
        return state._replace(
            best_ade=ade if improved else state.best_ade,
            epochs_without_improvement=waiting,
            flat_epochs=epoch if ends else None,
        )


def parameter_groups(predictor: nn.Module) -> list[dict[str, Any]]:
    """Splits parameters so weight decay applies to matrices only, not to
    biases or norms.
    """
    parameters = list(predictor.parameters())
    decayed = [parameter for parameter in parameters if parameter.ndim >= 2]
    undecayed = [parameter for parameter in parameters if parameter.ndim < 2]
    return [
        {"params": decayed, "weight_decay": WEIGHT_DECAY},
        {"params": undecayed, "weight_decay": 0.0},
    ]


def parse_arguments() -> argparse.Namespace:
    """The command line. Everything that shapes a run is required, so a launch
    script states its recipe in full.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("staged_directory", type=Path)
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--checkpoint-every-seconds", type=int, required=True)
    parser.add_argument("--stop-after-seconds", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--gradient-clip-norm", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--decay-from-epoch", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--holdout-directory", type=Path, default=None)
    # Instead of --decay-from-epoch: hold the rate until held-out minADE
    # stalls (PlateauRule), end the session there, and fall over
    # --decay-epochs when resumed. --epochs is then the most the run takes.
    parser.add_argument("--decay-on-plateau", action="store_true")
    parser.add_argument("--decay-epochs", type=int, default=None)
    parser.add_argument("--plateau-patience", type=int, default=None)
    parser.add_argument("--plateau-min-improvement", type=float, default=None)
    parser.add_argument("--min-flat-epochs", type=int, default=None)
    return parser.parse_args()


PLATEAU_OPTIONS = (
    "decay_epochs",
    "plateau_patience",
    "plateau_min_improvement",
    "min_flat_epochs",
)


def plateau_rule(arguments: argparse.Namespace) -> PlateauRule | None:
    """The rule --decay-on-plateau asks for, every one of its settings
    stated; None without it.
    """
    if not arguments.decay_on_plateau:
        return None
    if arguments.holdout_directory is None:
        raise ValueError(
            "--decay-on-plateau reads held-out minADE: give"
            " --holdout-directory"
        )
    if arguments.decay_from_epoch is not None:
        raise ValueError(
            "--decay-on-plateau decides where the fall starts; drop"
            " --decay-from-epoch"
        )
    missing = [
        "--" + name.replace("_", "-")
        for name in PLATEAU_OPTIONS
        if getattr(arguments, name) is None
    ]
    if missing:
        raise ValueError(f"--decay-on-plateau needs {', '.join(missing)}")
    max_flat_epochs = arguments.epochs - arguments.decay_epochs
    if max_flat_epochs < arguments.min_flat_epochs:
        raise ValueError(
            f"--epochs {arguments.epochs} leaves {max_flat_epochs} flat"
            f" epochs before a {arguments.decay_epochs}-epoch fall, fewer"
            f" than --min-flat-epochs {arguments.min_flat_epochs}"
        )
    return PlateauRule(
        patience=arguments.plateau_patience,
        min_improvement=arguments.plateau_min_improvement,
        min_flat_epochs=arguments.min_flat_epochs,
        max_flat_epochs=max_flat_epochs,
    )


class TrainingRun:
    """One launch of train.py, set up in full before the first batch: the
    model with its optimiser and scaler restored from any checkpoint, the
    epoch plan and the learning-rate schedule. run() is the epoch loop.
    """

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.launch_time = time.perf_counter()
        self.arguments = arguments
        # torchrun sets these three; a plain launch is one process.
        self.process_count = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_main = self.rank == 0
        if arguments.batch_size % self.process_count:
            raise ValueError(
                f"--batch-size {arguments.batch_size} does not split over"
                f" {self.process_count} processes"
            )
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda", local_rank)
            torch.cuda.set_device(self.device)
        if self.process_count > 1:
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            distributed.init_process_group(backend)
        torch.manual_seed(arguments.seed)
        anchors = load_anchor_file(arguments.anchors)
        self.predictor = MotionPredictor(anchors).to(self.device)
        # DDP averages every gradient across the processes; with one
        # process the model is trained directly.
        self.trainer = self.predictor
        if self.process_count > 1:
            on_gpu = self.device.type == "cuda"
            device_ids = [local_rank] if on_gpu else None
            self.trainer = DistributedDataParallel(
                self.predictor, device_ids=device_ids
            )
        groups = parameter_groups(self.predictor)
        self.optimizer = torch.optim.AdamW(groups, lr=arguments.learning_rate)
        scaling = arguments.mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=scaling)
        self.rule = plateau_rule(arguments)
        if self.rule is not None and self.process_count > 1:
            raise ValueError(
                "--decay-on-plateau reads held-out scores, which only the"
                " main process computes; train on one process"
            )
        self.completed_epochs, stored_plan, stored_plateau = self.resume()
        self.saved_epochs = self.completed_epochs
        self.plateau = self.restore_plateau(stored_plateau)

        self.all_paths = sorted(arguments.staged_directory.glob("*.npz"))
        if not self.all_paths:
            raise FileNotFoundError(
                f"no .npz scenarios in {arguments.staged_directory}"
            )
        self.paths = self.all_paths[self.rank :: self.process_count]
        self.holdout_paths = self.find_holdout_paths()
        self.plan = self.plan_epochs(stored_plan)
        self.schedule = self.learning_rate_schedule()
        tensorboard = arguments.checkpoint_path.parent / "tensorboard"
        self.writer = SummaryWriter(tensorboard) if self.is_main else None

    def find_holdout_paths(self) -> list[Path]:
        """The held-out scenes, none without --holdout-directory; a given
        directory with no scenes is refused before any work starts.
        """
        directory = self.arguments.holdout_directory
        if directory is None:
            return []
        paths = sorted(directory.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"no .npz held-out scenes in {directory}")
        return paths

    def evaluate_holdout(self) -> HoldoutScores:
        """Loss, and minADE and minFDE over the 6 kept modes, for every
        designated target in --holdout-directory, scenes training never
        sees, pooled and then per object type at 3, 5 and 8 s. Runs in
        float32 and changes no weight.
        """
        settings = BatchSettings(
            batch_size=self.arguments.batch_size,
            worker_count=0,
            prefetch_batches=0,
            seed=0,
            designated_only=True,
        )
        progress = EpochProgress(self.device)
        by_type = TypeHorizonErrors()
        self.predictor.eval()
        with torch.no_grad():
            for host_batch in pipeline.batches(self.holdout_paths, settings):
                batch = host_batch.each(lambda array: array.to(self.device))
                rounds = self.predictor(batch)
                progress.record(
                    loss.round_loss_terms(rounds, batch), False, False
                )
                progress.sample_count += batch.agent_history.shape[0]
                final = rounds[-1]
                kept = prune_modes(
                    final.trajectories, final.confidence_logits
                ).trajectories
                progress.add_pruned_errors(kept, batch)
                by_type.add(kept, batch)
        self.predictor.train()
        overall = {
            **progress.loss_means("holdout/loss_"),
            "holdout/ade_80step": mean_or_nan(
                progress.ade_sum, progress.ade_count
            ),
            "holdout/fde_80step": mean_or_nan(
                progress.fde_sum, progress.fde_count
            ),
            "holdout/targets": progress.sample_count,
        }
        return HoldoutScores(overall, by_type.scalars("holdout_"))

    def main_decides(self, decision: bool) -> bool:
        """The main process's decision, sent to every process, so all of them
        stop or go together.
        """
        if self.process_count == 1:
            return decision
        shared = torch.tensor(int(decision), device=self.device)
        distributed.broadcast(shared, src=0)
        return bool(shared.item())

    def smallest(self, value: int) -> int:
        """The smallest value across the processes."""
        if self.process_count == 1:
            return value
        shared = torch.tensor(value, device=self.device)
        distributed.all_reduce(shared, op=distributed.ReduceOp.MIN)
        return int(shared.item())

    def report(self, heading: str, scalars: Scalars, step: int) -> None:
        """Writes a table of scalars to TensorBoard and prints it as the log
        line, so the console and the board never drift. Only the main
        process reports.
        """
        if not self.is_main:
            return
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)
        printed = [f"{name} {value:.4g}" for name, value in scalars.items()]
        print(" | ".join([heading] + printed))
        self.writer.flush()

    def resume(self) -> tuple[int, EpochPlan | None]:
        """Loads the checkpoint when --resume is set and returns how many epochs
        are complete and the plan they were trained under. Falls back to the
        previous checkpoint, which a crash between save_checkpoint's two
        renames leaves behind.
        """
        if not self.arguments.resume:
            return 0, None, None
        path = self.arguments.checkpoint_path
        fallback = previous_path(path)
        missing = not path.exists() and fallback.exists()
        if self.is_main and missing:
            # A crash between save_checkpoint's two renames leaves only the
            # fallback, which is moved back so the checkpoint path holds it.
            fallback.replace(path)
            print(f"checkpoint missing, restored it from {fallback}")
        if self.process_count > 1:
            distributed.barrier()
        if not path.exists():
            return 0, None, None
        checkpoint = load_checkpoint(path, self.device)
        plan_record = checkpoint.get("epoch_plan") or {}
        if not plan_record.get("designated_only", True):
            raise ValueError(
                f"{path} was trained on every eligible agent, not on the"
                " designated targets; start a fresh run instead"
            )
        model_state = checkpoint["model_state"]
        check_model_state(self.predictor, model_state, path)
        stored_anchors = model_state[ANCHORS_KEY].to(self.device)
        given_anchors = self.predictor.state_dict()[ANCHORS_KEY]
        if not torch.equal(stored_anchors, given_anchors):
            raise ValueError(
                f"{path} was trained with other anchors than"
                f" {self.arguments.anchors}; resume with the anchors it holds"
            )
        self.predictor.load_state_dict(model_state)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler.load_state_dict(checkpoint["gradient_scaler_state"])
        completed = checkpoint["completed_epochs"]
        print(f"resuming {path}: {completed} epochs complete")
        stored = EpochPlan.stored(checkpoint.get("epoch_plan"))
        return completed, stored, checkpoint.get("plateau")

    def restore_plateau(self, record: dict | None) -> PlateauState | None:
        """The plateau state a resumed run carries on from, checked against
        this command line: a run trained one way is not resumed the other,
        and a fall already under way keeps its length. A branch copy saved
        as the flat phase ended takes any --decay-epochs.
        """
        path = self.arguments.checkpoint_path
        if self.rule is None:
            if record is not None:
                raise ValueError(
                    f"{path} was trained with --decay-on-plateau; resume it"
                    " with the same plateau settings"
                )
            return None
        if record is None:
            if self.completed_epochs:
                raise ValueError(
                    f"{path} was trained without --decay-on-plateau; start"
                    " a fresh run instead"
                )
            return PlateauState.fresh()
        state = PlateauState(**record)
        fall = self.arguments.decay_epochs
        if state.fall_epochs is not None and state.fall_epochs != fall:
            raise ValueError(
                f"{path} is part way through a {state.fall_epochs}-epoch"
                f" fall; resume with --decay-epochs {state.fall_epochs}"
            )
        if state.flat_epochs is not None:
            needed = state.flat_epochs + fall
            if needed > self.arguments.epochs:
                raise ValueError(
                    f"{state.flat_epochs} flat epochs and a {fall}-epoch"
                    f" fall need --epochs {needed} or more"
                )
        return state

    def planned_epochs(self) -> int:
        """The epochs this run trains in all: --epochs, or once the flat
        phase has ended, its length plus the fall.
        """
        if self.plateau is None or self.plateau.flat_epochs is None:
            return self.arguments.epochs
        return self.plateau.flat_epochs + self.arguments.decay_epochs

    def batch_settings(self, epoch_index: int) -> BatchSettings:
        arguments = self.arguments
        return BatchSettings(
            batch_size=arguments.batch_size // self.process_count,
            worker_count=arguments.workers,
            prefetch_batches=arguments.prefetch,
            seed=arguments.seed + epoch_index,
            # Waymo scores only its designated targets; most of the other
            # agents stand still, so they would swamp the loss.
            designated_only=True,
        )

    def plan_epochs(self, stored: EpochPlan | None) -> EpochPlan:
        """The optimiser steps in one epoch, which the learning-rate schedule is
        measured in, read from the checkpoint when the settings match.
        """
        settings = self.batch_settings(0)
        names = "\n".join(path.name for path in self.all_paths).encode()
        settings_now = (
            hashlib.sha256(names).hexdigest(),
            self.process_count,
            settings.worker_count,
            settings.batch_size,
            settings.designated_only,
        )
        if stored is not None and stored[:-1] == settings_now:
            print("optimiser step count read from the checkpoint")
            return stored
        # Each process counts its own share of the files; every process
        # then stops at the smallest count, so none waits on another.
        own_count = pipeline.steps_per_epoch(self.paths, settings)
        count = self.smallest(own_count)
        epochs = self.arguments.epochs
        print(
            f"{count} optimiser steps per epoch,"
            f" {count * epochs} over {epochs} epochs"
        )
        return EpochPlan(*settings_now, steps_per_epoch=count)

    def learning_rate_schedule(self) -> Schedule:
        arguments = self.arguments
        steps = self.plan.steps_per_epoch
        decay_start = None
        decay_end = None
        if self.plateau is not None and self.plateau.flat_epochs is not None:
            flat_epochs = self.plateau.flat_epochs
            decay_start = max(flat_epochs * steps, arguments.warmup_steps)
            decay_end = (flat_epochs + arguments.decay_epochs) * steps
        if arguments.decay_from_epoch is not None:
            # The fall starts no earlier than the end of warm-up, so the
            # rate never drops from the peak in one step.
            first_decay_step = (arguments.decay_from_epoch - 1) * steps
            decay_start = max(first_decay_step, arguments.warmup_steps)
            decay_end = arguments.epochs * steps
        peak = arguments.learning_rate
        return Schedule(peak, arguments.warmup_steps, decay_start, decay_end)

    def save(self, completed_epochs: int, path: Path | None = None) -> None:
        """Writes the checkpoint. The resume file is updated only when path
        is omitted or is that file, so a sidecar cannot replace a finished
        epoch.
        """
        if not self.is_main:
            return
        destination = path or self.arguments.checkpoint_path
        model_state = self.predictor.state_dict()
        state = {
            "model_state": model_state,
            "optimizer_state": self.optimizer.state_dict(),
            "gradient_scaler_state": self.scaler.state_dict(),
            "completed_epochs": completed_epochs,
            "code_version": layout.STAGING_CODE_VERSION,
            "epoch_plan": self.plan._asdict(),
        }
        if self.plateau is not None:
            state["plateau"] = self.plateau._asdict()
        save_checkpoint(destination, state)
        updates_resume = destination == self.arguments.checkpoint_path
        if updates_resume:
            self.saved_epochs = completed_epochs

    def optimise(self, total: Tensor, rate: float) -> tuple[bool, bool]:
        """Sets this step's rate, then backward pass, gradient clip and
        optimiser step. Returns whether the gradient was clipped and whether
        the scaler skipped the step over non-finite gradients.
        """
        clip = self.arguments.gradient_clip_norm
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.zero_grad()
        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        parameters = self.predictor.parameters()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
        # A disabled scaler steps on any gradient, so a non-finite one is
        # skipped here instead.
        step_is_finite = bool(torch.isfinite(gradient_norm))
        if not self.scaler.is_enabled() and not step_is_finite:
            self.optimizer.zero_grad()
            return False, True
        self.scaler.step(self.optimizer)
        # The scaler lowers its scale only when it skipped a step over
        # non-finite gradients, which is how skips are detected.
        scale_before = self.scaler.get_scale()
        self.scaler.update()
        skipped = self.scaler.get_scale() < scale_before
        # An infinite norm is larger than every clip; that step was skipped.
        clipped = step_is_finite and float(gradient_norm) > clip
        return clipped, skipped

    def train_epoch(self, batches: Batches, epoch_index: int) -> Scalars:
        """One pass over the data: forward and loss, the optimiser step and
        the monitor for each batch, then a log line every LOG_EVERY_BATCHES
        and a checkpoint every --checkpoint-every-seconds.
        """
        progress = EpochProgress(self.device)
        steps_before = epoch_index * self.plan.steps_per_epoch
        mixed_precision = self.scaler.is_enabled()
        last_checkpoint = time.perf_counter()
        wait_start = time.perf_counter()
        for host_batch in batches:
            progress.data_wait_seconds += time.perf_counter() - wait_start
            batch = host_batch.each(
                lambda array: array.to(self.device, non_blocking=True)
            )
            with torch.amp.autocast(self.device.type, enabled=mixed_precision):
                rounds = self.trainer(batch)
                losses = loss.round_loss_terms(rounds, batch)
            predicted = rounds[-1]
            global_step = steps_before + progress.batch_count
            rate = self.schedule.at(global_step)
            clipped, skipped = self.optimise(losses.total, rate)
            loss_is_finite = progress.record(losses, clipped, skipped)
            progress.monitor_batch(predicted, batch)

            if progress.batch_count % LOG_EVERY_BATCHES == 0:
                heading = f"  batch {progress.batch_count}"
                scalars = progress.log_scalars(rate)
                self.report(heading, scalars, global_step + 1)

            seconds = time.perf_counter() - last_checkpoint
            if seconds >= self.arguments.checkpoint_every_seconds:
                if loss_is_finite:
                    recorded, destination = checkpoint_destination(
                        self.arguments.checkpoint_path, epoch_index, False
                    )
                    self.save(recorded, destination)
                last_checkpoint = time.perf_counter()
            # Every process stops at the same step count, so no process
            # waits for a gradient the others will never send.
            if progress.batch_count == self.plan.steps_per_epoch:
                break
            wait_start = time.perf_counter()
        return progress.epoch_scalars()

    def stop_message(self, prefix: str, elapsed: float, completed: int) -> str:
        arguments = self.arguments
        return (
            f"{prefix}: {elapsed / SECONDS_PER_HOUR:.2f} h elapsed against a"
            f" --stop-after-seconds budget of"
            f" {arguments.stop_after_seconds / SECONDS_PER_HOUR:.2f} h."
            f" {completed} of {self.planned_epochs()} epochs are complete and"
            f" {arguments.checkpoint_path} holds {self.saved_epochs}."
        )

    def begin_fall(self, epoch_index: int) -> None:
        """Fixes the fall's length in the plateau state as its first epoch
        starts, so a later session cannot change it part way.
        """
        state = self.plateau
        if state is None or state.flat_epochs is None:
            return
        if state.fall_epochs is None and epoch_index >= state.flat_epochs:
            fall = self.arguments.decay_epochs
            self.plateau = state._replace(fall_epochs=fall)

    def advance_plateau(self, epoch: int, holdout_ade: float) -> bool:
        """Folds a finished epoch's held-out minADE into the plateau state;
        returns whether the flat phase ended with it, in which case the
        schedule now holds the fall.
        """
        if self.plateau is None or self.plateau.flat_epochs is not None:
            return False
        self.plateau = self.rule.after_epoch(self.plateau, epoch, holdout_ade)
        if self.plateau.flat_epochs is None:
            return False
        self.schedule = self.learning_rate_schedule()
        return True

    def save_branch(self, completed: int) -> None:
        """Keeps a copy of the checkpoint as the flat phase ended, from which
        falls of other lengths can be resumed.
        """
        branch = branch_path(self.arguments.checkpoint_path)
        self.save(completed, branch)
        print(
            f"flat phase ended after epoch {completed}; {branch} keeps it."
            f" Resume to run the {self.arguments.decay_epochs}-epoch fall."
        )

    def run(self) -> None:
        """Trains the remaining epochs inside the time budget, checkpointing
        after each finished epoch. With --decay-on-plateau the session also
        ends where the flat phase does.
        """
        arguments = self.arguments
        budget = arguments.stop_after_seconds
        start = self.launch_time
        last_epoch_seconds = 0.0
        epoch_index = self.completed_epochs
        while epoch_index < self.planned_epochs():
            # Stop before an epoch that would not fit in the time budget,
            # since Kaggle ends a session without warning.
            elapsed = time.perf_counter() - start
            if self.main_decides(elapsed + last_epoch_seconds > budget):
                heading = f"STOPPING BEFORE EPOCH {epoch_index + 1}"
                print(self.stop_message(heading, elapsed, epoch_index))
                return

            self.begin_fall(epoch_index)
            epoch_start = time.perf_counter()
            settings = self.batch_settings(epoch_index)
            batches = pipeline.batches(self.paths, settings)
            scalars = self.train_epoch(batches, epoch_index)
            last_epoch_seconds = time.perf_counter() - epoch_start
            heading = f"epoch {epoch_index + 1}/{self.planned_epochs()}"
            self.report(heading, scalars, epoch_index + 1)
            holdout_ade = math.nan
            if self.holdout_paths and self.is_main:
                holdout = self.evaluate_holdout()
                holdout_ade = holdout.overall["holdout/ade_80step"]
                heading = f"held-out after epoch {epoch_index + 1}"
                self.report(heading, holdout.overall, epoch_index + 1)
                heading = f"held-out by type after epoch {epoch_index + 1}"
                self.report(heading, holdout.by_type, epoch_index + 1)

            total = scalars["epoch/loss_total"]
            flat_phase_ended = False
            if math.isfinite(total):
                # Only an epoch whose weights are kept moves the plateau.
                flat_phase_ended = self.advance_plateau(
                    epoch_index + 1, holdout_ade
                )
                recorded, destination = checkpoint_destination(
                    self.arguments.checkpoint_path, epoch_index, True
                )
                self.save(recorded, destination)
            else:
                print(
                    f"epoch {epoch_index + 1} mean total loss {total},"
                    f" checkpoint left as it was"
                )
            if flat_phase_ended:
                self.save_branch(epoch_index + 1)
                return

            elapsed = time.perf_counter() - start
            if self.main_decides(elapsed >= budget):
                completed = epoch_index + 1
                print(self.stop_message("STOPPING EARLY", elapsed, completed))
                return
            epoch_index += 1
        print(
            f"done: {arguments.checkpoint_path} holds"
            f" {self.saved_epochs} epochs"
        )


def epochs_recorded(epoch_index: int, finished: bool) -> int:
    """The completed-epoch count a checkpoint write stores for this epoch."""
    return epoch_index + int(finished)


def branch_path(checkpoint_path: Path) -> Path:
    """Where a plateau run keeps its checkpoint as the flat phase ended."""
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".flat")


def checkpoint_destination(path: Path, epoch_index: int, finished: bool):
    """The epoch count and file for this save. An unfinished epoch is a
    sidecar, so it cannot replace the finished checkpoint.
    """
    recorded = epochs_recorded(epoch_index, finished)
    if finished:
        destination = path
    else:
        destination = path.with_suffix(path.suffix + ".periodic")
    return recorded, destination


# ------------------------------------------------------------------
# THE LEARNING RATE, a function of the step count alone
#
#   peak |      ______________________
#        |     /                      `.
#        |    /                         `.
#        |   /                            `._
#      0 +--+---------------------------+-----+--> step
#         warm-up         hold          cosine fall, reaching
#                                       zero on the last step
#
#   The fall starts at --decay-from-epoch and is off without it. With
#   --decay-on-plateau the hold instead ends where held-out minADE stalls
#   (PlateauRule), the session stops there, and the resumed run falls
#   over --decay-epochs.
# ------------------------------------------------------------------

if __name__ == "__main__":
    faulthandler.dump_traceback_later(
        STACK_DUMP_SECONDS, repeat=True, file=sys.stderr
    )
    # The Kaggle launcher sends SIGUSR1 before it kills a run that passed
    # its deadline, so the log shows where the run was.
    faulthandler.register(signal.SIGUSR1, file=sys.stderr)
    # Kaggle pipes stdout to its log, so each line is flushed as printed.
    sys.stdout.reconfigure(line_buffering=True)
    # Only the main process prints; the others would repeat every line,
    # and print() writes nothing while sys.stdout is None.
    printing = contextlib.nullcontext()
    if os.environ.get("RANK", "0") != "0":
        printing = contextlib.redirect_stdout(None)
    with printing:
        TrainingRun(parse_arguments()).run()
    if distributed.is_initialized():
        distributed.destroy_process_group()
