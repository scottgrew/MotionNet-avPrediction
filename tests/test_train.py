"""Tests for the training loop, covering the learning-rate schedule, resuming
and the Kaggle kernel's command line.
"""

import ast
import contextlib
import io
import unittest.mock
from pathlib import Path

import numpy as np
import pytest
import torch
from reference_implementations import unit_anchor_offsets_per_type

import train
from womd import checkpoint, formats, layout, loss, model
from womd.model import MotionPredictor
from womd.views import TargetInputs

KERNEL_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "kaggle_upload"
    / "kernel"
    / "run.py"
)


def synthetic_batch():
    """Two targets, each with three agents (itself first) and two map chunks
    of five dots whose crossing columns hold valid codes.
    """
    history = layout.HISTORY_STEPS
    signals = layout.NUM_TRAFFIC_SIGNAL_STATES
    chunk_rows = torch.randn(2, 2, 5, layout.MAP_FEATURE_DIM)
    codes = torch.randint(0, layout.NUM_BOUNDARY_CROSSING_CODES, (2, 2, 5))
    chunk_rows[..., layout.MAP_LEFT_BOUNDARY_CROSSING] = codes.float()
    chunk_rows[..., layout.MAP_RIGHT_BOUNDARY_CROSSING] = codes.float()
    return TargetInputs(
        agent_history=torch.randn(2, 3, history, layout.AGENT_FEATURE_DIM),
        agent_history_mask=torch.ones(2, 3, history, dtype=torch.bool),
        chunk_rows=chunk_rows,
        chunk_dot_mask=torch.ones(2, 2, 5, dtype=torch.bool),
        chunk_signal_history=torch.zeros(2, 2, history, signals),
        future_positions=torch.randn(2, layout.FUTURE_STEPS, 2),
        future_mask=torch.ones(2, layout.FUTURE_STEPS, dtype=torch.bool),
    )


def test_warmup_rises_then_holds():
    """The rate climbs through warm-up, holds at the chosen rate, and reaches
    zero on the last step.
    """
    warmup_steps = 20
    schedule = train.Schedule(train.LEARNING_RATE, warmup_steps, None, None)
    rates = [schedule.at(step) for step in range(200)]
    assert rates[0] < rates[warmup_steps - 1]
    rising = zip(rates[:warmup_steps], rates[1:warmup_steps], strict=False)
    assert all(later >= earlier for earlier, later in rising)
    assert all(rate == train.LEARNING_RATE for rate in rates[warmup_steps:])
    lowered = train.Schedule(1e-4, warmup_steps, None, None)
    assert lowered.at(warmup_steps + 5) == 1e-4


def test_rate_holds_then_falls_to_zero():
    """Zero is reached only on the last step."""
    schedule = train.Schedule(1e-3, 20, 100, 200)
    rates = [schedule.at(step) for step in range(200)]
    assert all(rate == 1e-3 for rate in rates[20:100])
    falling = zip(rates[100:199], rates[101:200], strict=True)
    assert all(later < earlier for earlier, later in falling)
    assert rates[150] < 1e-3 / 2
    assert rates[198] > 0.0
    assert rates[199] == 0.0


def test_checkpoint_round_trips(tmp_path):
    """A checkpoint with a tampered version is rejected."""
    model_state = torch.nn.Linear(1, 1).state_dict()
    state = {
        "model_state": model_state,
        "code_version": layout.STAGING_CODE_VERSION,
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(state, checkpoint_path)
    reloaded_state = checkpoint.load_checkpoint(checkpoint_path)
    assert reloaded_state["code_version"] == layout.STAGING_CODE_VERSION
    assert torch.equal(
        reloaded_state["model_state"]["weight"], model_state["weight"]
    )
    tampered_state = dict(state)
    tampered_state["code_version"] = "not-a-real-version"
    tampered_path = tmp_path / "tampered.pt"
    torch.save(tampered_state, tampered_path)
    with pytest.raises(ValueError):
        checkpoint.load_checkpoint(tampered_path)


def test_one_training_step_runs_end_to_end():
    """Forward, loss, backward and optimiser step."""
    torch.manual_seed(0)
    predictor = MotionPredictor(unit_anchor_offsets_per_type())
    optimizer = torch.optim.AdamW(
        train.parameter_groups(predictor), lr=train.LEARNING_RATE
    )
    batch = synthetic_batch()
    total = loss.loss_terms(predictor(batch)[-1], batch).total
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    assert torch.isfinite(total)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0.0
        for parameter in predictor.parameters()
    )


def literal_run_arguments(call):
    """The literal arguments of one subprocess.run([...]) or
    subprocess.Popen([...]) call, or None.
    """
    is_attribute = isinstance(call.func, ast.Attribute)
    is_run = is_attribute and call.func.attr in ("run", "Popen")
    if not is_run or not call.args:
        return None
    argument_list = call.args[0]
    if not isinstance(argument_list, ast.List):
        return None
    literals = [
        element.value
        for element in argument_list.elts
        if isinstance(element, ast.Constant)
    ]
    return literals, len(argument_list.elts)


def kernel_train_arguments():
    """The literal train.py arguments in the Kaggle kernel script, and how many
    arguments the call has in total.
    """
    if not KERNEL_PATH.exists():
        pytest.skip(f"no Kaggle kernel script at {KERNEL_PATH}")
    kernel_tree = ast.parse(KERNEL_PATH.read_text())
    for node in ast.walk(kernel_tree):
        if not isinstance(node, ast.Call):
            continue
        found = literal_run_arguments(node)
        if found is None:
            continue
        literals, element_count = found
        if "train.py" in literals:
            launcher_count = literals.index("train.py") - 1
            return literals[launcher_count:], element_count - launcher_count
    raise AssertionError(f"no train.py invocation found in {KERNEL_PATH}")


def completed_flags(flag_arguments, value_of_flag):
    """The kernel's flags with the test's own values filled in for the flags
    in value_of_flag.
    """
    completed = []
    skip_next = False
    for position, argument in enumerate(flag_arguments):
        if skip_next:
            skip_next = False
            continue
        if argument in value_of_flag:
            completed.extend([argument, value_of_flag[argument]])
            has_value = position + 1 < len(flag_arguments)
            if has_value and not flag_arguments[position + 1].startswith("--"):
                skip_next = True
            continue
        completed.append(argument)
    return completed


def test_kernel_command_line_parses(tmp_path):
    """Parsed by train.py's real argument parser."""
    literal_arguments, element_count = kernel_train_arguments()
    flag_arguments = literal_arguments[2:]
    while flag_arguments and not flag_arguments[0].startswith("--"):
        flag_arguments = flag_arguments[1:]
    anchors_path = tmp_path / "training_anchors.npz"
    anchor_shape = (layout.NUM_OBJECT_TYPES, model.QUERY_COUNT, 2)
    np.savez(
        anchors_path,
        unit_anchors=np.zeros(anchor_shape, dtype=np.float32),
        provenance=formats.anchor_provenance("kernel-interface-test"),
    )
    value_of_flag = {
        "--anchors": str(anchors_path),
        "--stop-after-seconds": "1.0",
        "--holdout-directory": str(tmp_path),
    }
    completed_arguments = completed_flags(flag_arguments, value_of_flag)
    non_literal_count = element_count - len(literal_arguments)
    if "--anchors" not in flag_arguments:
        completed_arguments.extend(["--anchors", str(anchors_path)])
    if "--stop-after-seconds" not in flag_arguments and non_literal_count:
        completed_arguments.extend(["--stop-after-seconds", "1.0"])
    argv = [
        "train.py",
        str(tmp_path),
        str(tmp_path / "checkpoint.pt"),
    ] + completed_arguments

    patched_argv = unittest.mock.patch("sys.argv", argv)
    patched_planning = unittest.mock.patch.object(
        train.pipeline, "steps_per_epoch"
    )
    with patched_argv, patched_planning as blocked:
        blocked.side_effect = AssertionError(
            "parsing must fail before any work starts"
        )
        error_output = io.StringIO()
        try:
            with contextlib.redirect_stderr(error_output):
                train.TrainingRun(train.parse_arguments()).run()
        except AssertionError as expected_stop:
            assert "parsing must fail before any work starts" in str(
                expected_stop
            )
        except SystemExit as parse_failure:
            raise AssertionError(
                "the kernel's train.py command line does not parse: "
                f"{error_output.getvalue()}"
            ) from parse_failure


REPOSITORY = Path(__file__).resolve().parents[1]
STAGED_DIRECTORY = REPOSITORY / "data" / "staged"
ANCHOR_FILE = REPOSITORY / "data" / "training_anchors_54_8s.npz"


def training_run(staged, checkpoint_path, anchors=ANCHOR_FILE, *extra):
    """A TrainingRun built by train.py's own parser, on CPU, no workers."""
    command_line = [
        "train.py",
        str(staged),
        str(checkpoint_path),
        "--epochs",
        "2",
        "--batch-size",
        "4",
        "--workers",
        "0",
        "--anchors",
        str(anchors),
        "--checkpoint-every-seconds",
        "300",
        "--stop-after-seconds",
        "600",
        "--warmup-steps",
        "3",
        "--gradient-clip-norm",
        "250",
        "--resume",
        *extra,
    ]
    with unittest.mock.patch("sys.argv", command_line):
        return train.TrainingRun(train.parse_arguments())


def staged_copy(directory, first, count):
    directory.mkdir()
    for path in sorted(STAGED_DIRECTORY.glob("*.npz"))[first : first + count]:
        (directory / path.name).write_bytes(path.read_bytes())
    return directory


def test_non_finite_step_changes_no_weight(tmp_path):
    """Without an enabled gradient scaler, a NaN loss must be skipped, not
    stepped into every weight.
    """
    staged = staged_copy(tmp_path / "staged", 0, 2)
    run = training_run(staged, tmp_path / "checkpoint.pt")
    assert not run.scaler.is_enabled()
    before = [
        parameter.detach().clone() for parameter in run.predictor.parameters()
    ]
    batch = synthetic_batch()
    total = loss.loss_terms(run.predictor(batch)[-1], batch).total * float(
        "nan"
    )
    clipped, skipped = run.optimise(total, rate=1e-3)
    assert skipped and not clipped
    after = list(run.predictor.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, after))


def test_skipped_infinite_gradient_is_not_counted_as_clipped(tmp_path):
    """An infinite gradient norm exceeds the clip, but a step the scaler
    skips changed no weight and is not a clip.
    """
    staged = staged_copy(tmp_path / "staged", 0, 2)
    run = training_run(staged, tmp_path / "checkpoint.pt")

    class SkippingScaler:
        """Stands in for a CUDA gradient scaler: enabled, and a skip is a
        lowered scale with no optimiser step. This machine has no CUDA, so
        the real scaler disables itself.
        """

        def __init__(self) -> None:
            self.scale_value = 65536.0

        def is_enabled(self) -> bool:
            return True

        def scale(self, loss):
            return loss

        def unscale_(self, _optimizer) -> None:
            return None

        def step(self, _optimizer) -> None:
            return None

        def get_scale(self) -> float:
            return self.scale_value

        def update(self) -> None:
            self.scale_value *= 0.5

    run.scaler = SkippingScaler()
    before = [
        parameter.detach().clone() for parameter in run.predictor.parameters()
    ]
    batch = synthetic_batch()
    total = loss.loss_terms(run.predictor(batch)[-1], batch).total

    def infinite_norm(_parameters, _clip, *_args, **_kwargs):
        return torch.tensor(float("inf"))

    patched_norm = unittest.mock.patch.object(
        torch.nn.utils, "clip_grad_norm_", infinite_norm
    )
    with patched_norm:
        clipped, skipped = run.optimise(total, rate=1e-3)

    assert skipped and not clipped
    after = list(run.predictor.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, after))


def test_resume_checks_what_it_resumes(tmp_path):
    """A resumed run refuses other anchors, recounts for other scenes and
    restores a checkpoint left only as its fallback.
    """
    first = staged_copy(tmp_path / "first", 0, 3)
    other = staged_copy(tmp_path / "other", 3, 3)
    checkpoint_path = tmp_path / "checkpoint.pt"
    training_run(first, checkpoint_path).save(1)

    with np.load(ANCHOR_FILE) as stored:
        doubled = {name: stored[name] for name in stored.files}
    doubled["unit_anchors"] = doubled["unit_anchors"] * 2.0
    other_anchors = tmp_path / "other_anchors.npz"
    np.savez(other_anchors, **doubled)
    with pytest.raises(ValueError, match="other anchors"):
        training_run(first, checkpoint_path, other_anchors)

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        recounted = training_run(other, checkpoint_path)
    assert "read from the checkpoint" not in printed.getvalue()
    expected_steps = train.pipeline.steps_per_epoch(
        recounted.paths, recounted.batch_settings(0)
    )
    assert recounted.plan.steps_per_epoch == expected_steps

    checkpoint_path.replace(checkpoint.previous_path(checkpoint_path))
    restored = training_run(first, checkpoint_path)
    assert checkpoint_path.exists() and restored.completed_epochs == 1


def test_periodic_save_leaves_the_finished_epoch(tmp_path):
    """A save part way through an epoch goes to the sidecar, so the resume
    file keeps the last finished epoch's weights and count.
    """
    staged = staged_copy(tmp_path / "staged", 0, 2)
    checkpoint_path = tmp_path / "checkpoint.pt"
    run = training_run(staged, checkpoint_path)
    run.save(*train.checkpoint_destination(checkpoint_path, 0, True))
    finished_bytes = checkpoint_path.read_bytes()

    recorded, sidecar = train.checkpoint_destination(checkpoint_path, 1, False)
    run.save(recorded, sidecar)

    assert checkpoint_path.read_bytes() == finished_bytes
    assert run.saved_epochs == 1
    assert torch.load(sidecar)["completed_epochs"] == 1
    recorded, destination = train.checkpoint_destination(
        checkpoint_path, 1, True
    )
    assert (recorded, destination) == (2, checkpoint_path)


def test_resume_refuses_weights_trained_on_every_agent(tmp_path):
    """A checkpoint whose plan trained every eligible agent, most of them
    standing still, is not resumed into a designated-target run.
    """
    staged = staged_copy(tmp_path / "staged", 0, 2)
    checkpoint_path = tmp_path / "checkpoint.pt"
    run = training_run(staged, checkpoint_path)
    run.plan = run.plan._replace(designated_only=False)
    run.save(1)

    with pytest.raises(ValueError, match="every eligible agent"):
        training_run(staged, checkpoint_path)


def test_decay_from_the_first_epoch_starts_after_warm_up(tmp_path):
    """The cosine fall may not start before warm-up ends, or the rate drops
    from the peak in one step.
    """
    staged = staged_copy(tmp_path / "staged", 0, 2)
    run = training_run(
        staged,
        tmp_path / "checkpoint.pt",
        ANCHOR_FILE,
        "--decay-from-epoch",
        "1",
    )
    schedule = run.schedule
    peak = run.arguments.learning_rate
    end_of_warm_up = run.arguments.warmup_steps
    assert schedule.at(end_of_warm_up - 1) == pytest.approx(peak)
    assert schedule.at(end_of_warm_up) == pytest.approx(peak)


def test_every_epoch_is_a_full_pass(tmp_path):
    """Each epoch trains on every one of the run's files."""
    staged = staged_copy(tmp_path / "staged", 0, 3)
    run = training_run(staged, tmp_path / "checkpoint.pt")
    epoch_paths = []

    def recorded_batches(paths, _settings):
        epoch_paths.append(list(paths))
        return []

    finite_epoch = {"epoch/loss_total": 1.0}
    patched_batches = unittest.mock.patch.object(
        train.pipeline, "batches", recorded_batches
    )
    patched_epoch = unittest.mock.patch.object(
        run, "train_epoch", return_value=finite_epoch
    )
    with patched_batches, patched_epoch:
        run.run()

    assert epoch_paths == [run.paths, run.paths]


def test_holdout_scores_every_designated_target_and_trains_nothing(tmp_path):
    """The held-out pass sees each designated target in its directory once,
    changes no weight and leaves the model training; an empty directory is
    refused before any work starts.
    """
    staged = staged_copy(tmp_path / "staged", 0, 2)
    holdout = staged_copy(tmp_path / "holdout", 2, 2)
    checkpoint_path = tmp_path / "checkpoint.pt"
    run = training_run(
        staged,
        checkpoint_path,
        ANCHOR_FILE,
        "--holdout-directory",
        str(holdout),
    )
    before = [
        parameter.detach().clone() for parameter in run.predictor.parameters()
    ]

    scalars = run.evaluate_holdout()

    expected_targets = sum(
        train.pipeline.eligible_count(path, True)
        for path in sorted(holdout.glob("*.npz"))
    )
    assert scalars["holdout/targets"] == expected_targets
    assert np.isfinite(scalars["holdout/loss_total"])
    assert np.isfinite(scalars["holdout/ade_80step"])
    after = list(run.predictor.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, after))
    assert run.predictor.training

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="held-out"):
        training_run(
            staged,
            checkpoint_path,
            ANCHOR_FILE,
            "--holdout-directory",
            str(empty),
        )
