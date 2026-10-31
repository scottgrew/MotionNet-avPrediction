"""Tests for scene_at_step, rolling error metrics and MCAP recording."""

from collections import Counter

import numpy as np
import pytest
import torch
from test_invariants import two_lane_signal_scenario

import submit
from rolling import min_fde_metres, recording, truth_paths
from womd import layout, views
from womd.formats import StagedFile
from womd.pruning import SubmittedModes


def vehicle_track_rows(track_count):
    """Like two_lane_signal_scenario tracks but marked TYPE_VEHICLE."""
    rows = np.zeros(
        (track_count, layout.TOTAL_STEPS, layout.AGENT_FEATURE_DIM),
        dtype=np.float32,
    )
    vehicle_column = layout.AGENT_TYPE.start
    rows[:, :, vehicle_column] = 1.0
    rows[:, :, layout.AGENT_HEADING_COSINE] = 1.0
    rows[:, :, layout.AGENT_DIMENSIONS] = np.array([4.5, 2.0])
    return rows


def go_signal_history():
    signal_shape = (layout.HISTORY_STEPS, layout.NUM_TRAFFIC_SIGNAL_STATES)
    history = np.zeros(signal_shape, dtype=np.float32)
    go_column = layout.TRAFFIC_SIGNAL_STATES.index("LANE_STATE_GO")
    history[:, go_column] = 1.0
    return history


def test_scene_at_step_at_current_is_unchanged():
    history = go_signal_history()
    scenario = two_lane_signal_scenario(2, history)
    shifted = views.scene_at_step(scenario, layout.CURRENT_STEP_INDEX)
    for field in StagedFile._fields:
        left = getattr(scenario, field)
        right = getattr(shifted, field)
        if left is None:
            assert right is None
        else:
            assert np.array_equal(left, right)


def test_scene_at_step_shifts_tracks_and_leaves_input():
    history = go_signal_history()
    scenario = two_lane_signal_scenario(2, history)
    rows = vehicle_track_rows(2)
    steps = np.arange(layout.TOTAL_STEPS, dtype=np.float32)
    rows[0, :, layout.AGENT_POSITION.start] = steps
    scenario = scenario._replace(track_rows=rows)
    original_rows = scenario.track_rows.copy()
    shifted = views.scene_at_step(scenario, 25)
    for step_index in range(76):
        expected = 15.0 + step_index
        actual = shifted.track_rows[0, step_index, layout.AGENT_POSITION.start]
        assert actual == expected
    assert not shifted.track_valid[0, 76:].any()
    assert not shifted.track_rows[0, 76:].any()
    assert np.array_equal(scenario.track_rows, original_rows)


def test_scene_at_step_unknown_lights_after_staged_history():
    history = go_signal_history()
    scenario = two_lane_signal_scenario(2, history)
    shifted = views.scene_at_step(scenario, 13)
    unknown = layout.TRAFFIC_SIGNAL_STATES.index("LANE_STATE_UNKNOWN")
    go = layout.TRAFFIC_SIGNAL_STATES.index("LANE_STATE_GO")
    signalled = shifted.polyline_signal_histories[0]
    assert np.all(signalled[:8, go] == 1.0)
    assert np.all(signalled[8:, unknown] == 1.0)
    assert not shifted.polyline_signal_histories[1].any()


def test_scene_at_step_rejects_out_of_range():
    scenario = two_lane_signal_scenario(2, go_signal_history())
    with pytest.raises(ValueError, match="two-lane"):
        views.scene_at_step(scenario, 9)
    with pytest.raises(ValueError, match="two-lane"):
        views.scene_at_step(scenario, 91)


def test_min_fde_metres_averages_valid_tracks():
    history = go_signal_history()
    scenario = two_lane_signal_scenario(2, history)
    rows = vehicle_track_rows(2)
    rows[:, :, layout.AGENT_POSITION] = 0.0
    scenario = scenario._replace(track_rows=rows)
    shifted = views.scene_at_step(scenario, layout.CURRENT_STEP_INDEX)
    horizon = layout.CURRENT_STEP_INDEX + round(3.0 / layout.TIMESTEP_SECONDS)
    shifted = shifted._replace(track_rows=shifted.track_rows.copy())
    shifted.track_rows[0, horizon, layout.AGENT_POSITION] = np.array([0.0, 0.0])
    slot = layout.SUBMISSION_FUTURE_INDICES.index(29)
    mode_points = np.array(
        [[[5.0, 0.0], [8.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32
    )
    trajectories = np.zeros((2, 2, len(layout.SUBMISSION_FUTURE_INDICES), 2))
    trajectories[:, :, slot] = mode_points
    kept = SubmittedModes(torch.from_numpy(trajectories), torch.ones(2, 2))
    invalid = shifted.track_valid.copy()
    invalid[1, horizon] = False
    shifted = shifted._replace(track_valid=invalid)
    result = min_fde_metres(
        shifted, np.array([0, 1]), kept, round(3.0 / layout.TIMESTEP_SECONDS)
    )
    assert result == {"min_fde_metres": 5.0, "targets": 1}


def test_min_fde_metres_submission_slot_and_invalid_horizon():
    history = go_signal_history()
    scenario = two_lane_signal_scenario(2, history)
    rows = vehicle_track_rows(1)
    rows[:, :, layout.AGENT_POSITION] = 0.0
    scenario = scenario._replace(track_rows=rows)
    shifted = views.scene_at_step(scenario, 10)
    end_step = layout.TOTAL_STEPS - 1
    shifted = shifted._replace(track_rows=shifted.track_rows.copy())
    shifted.track_rows[0, end_step, layout.AGENT_POSITION] = np.array(
        [0.0, 0.0]
    )
    slot = layout.SUBMISSION_FUTURE_INDICES.index(79)
    trajectories = np.zeros((1, 2, len(layout.SUBMISSION_FUTURE_INDICES), 2))
    trajectories[0, :, slot] = np.array(
        [[3.0, 0.0], [9.0, 0.0]], dtype=np.float32
    )
    kept = SubmittedModes(torch.from_numpy(trajectories), torch.ones(1, 2))
    horizon_80 = layout.TOTAL_STEPS - 1 - 10
    assert horizon_80 == 80
    result = min_fde_metres(shifted, np.array([0]), kept, horizon_80)
    assert result == {"min_fde_metres": 3.0, "targets": 1}
    assert min_fde_metres(shifted, np.array([0]), kept, 7) is None


def test_recording_counts_and_zero_error_for_stationary():
    scenario = two_lane_signal_scenario(2, go_signal_history())
    rows = vehicle_track_rows(2)
    scenario = scenario._replace(track_rows=rows)
    messages = list(recording(scenario, submit.constant_velocity_modes))
    topics = Counter(topic for topic, _time, _message in messages)
    assert topics["/map"] == 1
    assert topics["/agents"] == layout.TOTAL_STEPS
    predict_steps = layout.TOTAL_STEPS - layout.CURRENT_STEP_INDEX
    assert topics["/futures"] == predict_steps
    assert topics["/truth"] == predict_steps
    assert topics["/error/3s"] == 51
    assert topics["/error/end"] == 16
    times = [log_time for _topic, log_time, _message in messages]
    assert times == sorted(times)
    for topic, _time, message in messages:
        if topic in ("/error/3s", "/error/end"):
            assert message["min_fde_metres"] == 0.0


def test_truth_uses_valid_steps_only():
    scenario = two_lane_signal_scenario(2, go_signal_history())
    rows = vehicle_track_rows(2)
    valid = scenario.track_valid.copy()
    gap = 25
    for step in range(layout.TOTAL_STEPS):
        rows[0, step, layout.AGENT_POSITION] = [float(step), 0.0]
    valid[0, gap] = False
    rows[0, gap, layout.AGENT_POSITION] = [999.0, 0.0]
    scenario = scenario._replace(track_rows=rows, track_valid=valid)
    (path,) = truth_paths(scenario, np.array([0]), layout.CURRENT_STEP_INDEX)
    expected = [step for step in range(10, layout.TOTAL_STEPS) if step != gap]
    assert path[:, 0].tolist() == expected
