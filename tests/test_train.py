"""Tests for the training loop, covering the learning-rate schedule, resuming
and the Kaggle kernel's command line.
"""

import ast
import contextlib
import io
import math
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

    scalars, by_type = run.evaluate_holdout()

    expected_targets = sum(
        train.pipeline.eligible_count(path, True)
        for path in sorted(holdout.glob("*.npz"))
    )
    assert scalars["holdout/targets"] == expected_targets
    assert np.isfinite(scalars["holdout/loss_total"])
    assert np.isfinite(scalars["holdout/ade_80step"])
    # Every target is counted under exactly one type, and the types'
    # 8 s minADE, weighted by their targets, is the pooled 80-step one.
    type_names = ("vehicle", "pedestrian", "cyclist")
    counts = [by_type[f"holdout_{name}/targets"] for name in type_names]
    assert sum(counts) == expected_targets
    weighted = sum(
        count * by_type[f"holdout_{name}/ade_8s"]
        for name, count in zip(type_names, counts)
        if count
    )
    pooled = weighted / expected_targets
    assert pooled == pytest.approx(scalars["holdout/ade_80step"], rel=1e-5)
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


# Run 41's held-out minADE over its ten flat epochs at 2e-4.
RUN_41_FLAT_ADE = (
    1.904,
    1.674,
    1.659,
    1.523,
    1.578,
    1.503,
    1.447,
    1.438,
    1.427,
    1.430,
)


def replay(rule, ade_by_epoch):
    """The plateau state after each epoch of a sequence of held-out ADEs."""
    state = train.PlateauState.fresh()
    states = []
    for epoch, ade in enumerate(ade_by_epoch, start=1):
        state = rule.after_epoch(state, epoch, ade)
        states.append(state)
    return states


def test_plateau_survives_one_noisy_epoch_then_ends_the_flat_phase():
    """Replaying run 41: the bump at epoch 5 and the 9 mm step at epoch 8
    must not end the flat phase; two epochs without a 1 cm gain after epoch
    6 do. A stall before the minimum flat epochs is ignored, and the
    maximum ends the phase whatever the held-out numbers do.
    """
    rule = train.PlateauRule(
        patience=2, min_improvement=0.01, min_flat_epochs=6, max_flat_epochs=14
    )
    states = replay(rule, RUN_41_FLAT_ADE)
    assert all(state.flat_epochs is None for state in states)
    assert states[-1].best_ade == pytest.approx(1.427)

    ended = replay(rule, RUN_41_FLAT_ADE + (1.429,))[-1]
    assert ended.flat_epochs == 11

    early_stall = replay(rule, (2.0, 2.0, 2.0, 2.0, 2.0, 2.0))
    assert [state.flat_epochs for state in early_stall] == [None] * 5 + [6]

    still_improving = replay(rule, [2.0 - 0.05 * epoch for epoch in range(14)])
    assert still_improving[-1].flat_epochs == 14
    assert still_improving[-2].flat_epochs is None

    unmeasured = replay(rule, (2.0,) * 5 + (math.nan,))
    assert unmeasured[-1].flat_epochs == 6

    # 6 mm a epoch is a counted 1 cm every second epoch, so a slow but
    # steady gain holds the rate until the maximum.
    slow = replay(rule, [2.0 - 0.006 * epoch for epoch in range(14)])
    assert [state.flat_epochs for state in slow] == [None] * 13 + [14]


def plateau_run(staged, checkpoint_path, holdout, *extra):
    return training_run(
        staged,
        checkpoint_path,
        ANCHOR_FILE,
        "--epochs",
        "6",
        "--holdout-directory",
        str(holdout),
        "--decay-on-plateau",
        "--decay-epochs",
        "2",
        "--plateau-patience",
        "1",
        "--plateau-min-improvement",
        "0.01",
        "--min-flat-epochs",
        "2",
        *extra,
    )


def scripted_run(run, ade_by_epoch):
    """Runs run with each epoch's training replaced by a finite loss and
    each held-out pass by the next scripted ADE; returns the epochs run.
    """
    finished = []

    def epoch(_batches, epoch_index):
        finished.append(epoch_index + 1)
        return {"epoch/loss_total": 1.0}

    scores = iter(ade_by_epoch)

    def holdout():
        overall = {"holdout/ade_80step": next(scores)}
        return train.HoldoutScores(overall, {})

    patched_batches = unittest.mock.patch.object(
        train.pipeline, "batches", lambda *_: []
    )
    patched_epoch = unittest.mock.patch.object(
        run, "train_epoch", side_effect=epoch
    )
    patched_holdout = unittest.mock.patch.object(
        run, "evaluate_holdout", holdout
    )
    with patched_batches, patched_epoch, patched_holdout:
        run.run()
    return finished


def test_plateau_run_stops_at_the_trigger_and_resumes_into_the_fall(tmp_path):
    """A session ends when the flat phase does, leaving the checkpoint and a
    branch copy of it; the resumed session runs only the fall, whose rate
    holds at the peak through the flat epochs and reaches zero on its last
    step.
    """
    staged = staged_copy(tmp_path / "staged", 0, 2)
    holdout = staged_copy(tmp_path / "holdout", 2, 1)
    checkpoint_path = tmp_path / "checkpoint.pt"

    first = plateau_run(staged, checkpoint_path, holdout)
    assert scripted_run(first, [2.0, 2.0, 2.0]) == [1, 2]
    branch = train.branch_path(checkpoint_path)
    stored = torch.load(branch, weights_only=False)
    assert stored["completed_epochs"] == 2
    assert stored["plateau"]["flat_epochs"] == 2

    second = plateau_run(staged, checkpoint_path, holdout)
    assert scripted_run(second, [1.9, 1.8]) == [3, 4]
    steps = second.plan.steps_per_epoch
    peak = second.arguments.learning_rate
    assert second.schedule.at(2 * steps - 1) == pytest.approx(peak)
    assert second.schedule.at(4 * steps - 1) == pytest.approx(0.0)
    assert (
        torch.load(checkpoint_path, weights_only=False)["completed_epochs"] == 4
    )


def test_plateau_recipe_is_checked(tmp_path):
    """The plateau needs a held-out directory and excludes a fixed decay
    epoch; a resumed fall may not change its length, while a branch copy
    resumed before the fall may.
    """
    staged = staged_copy(tmp_path / "staged", 0, 2)
    holdout = staged_copy(tmp_path / "holdout", 2, 1)
    checkpoint_path = tmp_path / "checkpoint.pt"
    with pytest.raises(ValueError, match="holdout"):
        training_run(
            staged,
            checkpoint_path,
            ANCHOR_FILE,
            "--decay-on-plateau",
            "--decay-epochs",
            "2",
            "--plateau-patience",
            "1",
            "--plateau-min-improvement",
            "0.01",
            "--min-flat-epochs",
            "2",
        )
    with pytest.raises(ValueError, match="decay-from-epoch"):
        plateau_run(staged, checkpoint_path, holdout, "--decay-from-epoch", "3")

    scripted_run(plateau_run(staged, checkpoint_path, holdout), [2.0, 2.0])
    branch = train.branch_path(checkpoint_path)
    branch_copy = tmp_path / "branch.pt"
    branch_copy.write_bytes(branch.read_bytes())
    longer = plateau_run(staged, branch_copy, holdout, "--decay-epochs", "3")
    assert scripted_run(longer, [1.9, 1.8, 1.7]) == [3, 4, 5]
    steps = longer.plan.steps_per_epoch
    assert longer.schedule.at(5 * steps - 1) == pytest.approx(0.0)

    # One epoch into a two-epoch fall, its length is fixed.
    mid_fall = torch.load(branch, weights_only=False)
    mid_fall["completed_epochs"] = 3
    mid_fall["plateau"]["fall_epochs"] = 2
    mid_fall_path = tmp_path / "mid_fall.pt"
    torch.save(mid_fall, mid_fall_path)
    with pytest.raises(ValueError, match="decay-epochs"):
        plateau_run(staged, mid_fall_path, holdout, "--decay-epochs", "4")
