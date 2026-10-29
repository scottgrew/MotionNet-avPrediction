"""Every change of frame in the project: points, directions, headings,
agent rows and map rows between the world, scene and agent frames.

A Frame is an origin and a heading measured in the frame the points are in
now. The three frames of the project are drawn at the end of this file.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy import ndarray

from womd import layout


class Frame(NamedTuple):
    """A coordinate frame as seen from its parent frame."""

    origin: np.ndarray  # (2,) metres
    heading: float  # radians


def rotation_matrix(heading: float, dtype: type = np.float64) -> np.ndarray:
    cosine = np.cos(heading)
    sine = np.sin(heading)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=dtype)


def positions_to_frame(positions: np.ndarray, frame: Frame) -> np.ndarray:
    """Moves points into the frame whose x axis points along its heading."""
    positions = np.asarray(positions)
    centred = positions - np.asarray(frame.origin, dtype=positions.dtype)
    # A row vector p lands at (p - origin) R(heading), which is the column
    # form R(heading)^T (p - origin), a rotation by -heading.
    return centred @ rotation_matrix(frame.heading, positions.dtype)


def positions_from_frame(positions: np.ndarray, frame: Frame) -> np.ndarray:
    frame_positions = np.asarray(positions, dtype=np.float64)
    rotated = frame_positions @ rotation_matrix(frame.heading).T
    return rotated + np.asarray(frame.origin, dtype=np.float64)


def directions_to_frame(directions: np.ndarray, heading: float) -> np.ndarray:
    """Rotates direction vectors into a frame; a direction has no position, so
    no origin is subtracted.
    """
    directions = np.asarray(directions)
    return directions @ rotation_matrix(heading, directions.dtype)


def directions_from_frame(directions: np.ndarray, heading: float) -> np.ndarray:
    frame_directions = np.asarray(directions, dtype=np.float64)
    return frame_directions @ rotation_matrix(heading).T


def headings_to_frame(headings: np.ndarray, heading: float) -> np.ndarray:
    """Headings relative to the frame heading, wrapped to [-pi, pi)."""
    return wrap_to_pi(np.asarray(headings, dtype=np.float64) - heading)


def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    # (a + pi) mod 2 pi - pi lands every angle in [-pi, pi).
    full_turn = 2.0 * np.pi
    shifted = np.asarray(angles, dtype=np.float64) + np.pi
    return shifted % full_turn - np.pi


# A heading's (cos, sin) columns, which rotate like a direction.
HEADING = slice(layout.AGENT_HEADING_COSINE, layout.AGENT_HEADING_SINE + 1)


def agent_frame(rows: ndarray) -> Frame:
    """A track's own frame at the current step, from its rows (91, 13) in the
    scene frame.
    """
    now = rows[layout.CURRENT_STEP_INDEX]
    sine = now[layout.AGENT_HEADING_SINE]
    cosine = now[layout.AGENT_HEADING_COSINE]
    return Frame(now[layout.AGENT_POSITION], np.arctan2(sine, cosine))


def rows_to_frame(rows: ndarray, frame: Frame) -> ndarray:
    """Re-expresses agent rows' position, velocity and heading columns in
    another agent's frame.
    """
    moved = rows.copy()
    moved[..., layout.AGENT_POSITION] = positions_to_frame(
        rows[..., layout.AGENT_POSITION], frame
    )
    moved[..., layout.AGENT_VELOCITY] = directions_to_frame(
        rows[..., layout.AGENT_VELOCITY], frame.heading
    )
    moved[..., HEADING] = directions_to_frame(rows[..., HEADING], frame.heading)
    return moved


def map_rows_to_frame(rows: ndarray, frame: Frame) -> ndarray:
    """Re-expresses map rows' position, direction and stop point columns in a
    frame. A dot with no stop point keeps zeros there.
    """
    moved = rows.copy()
    moved[..., layout.MAP_POSITION] = positions_to_frame(
        rows[..., layout.MAP_POSITION], frame
    )
    moved[..., layout.MAP_DIRECTION] = directions_to_frame(
        rows[..., layout.MAP_DIRECTION], frame.heading
    )
    stop_points = rows[..., layout.MAP_STOP_POINT]
    # Two comparisons, because .any over the strided last axis is slower.
    has_stop_point = (stop_points[..., 0] != 0.0) | (stop_points[..., 1] != 0.0)
    # Few dots carry a stop point, so only theirs are moved.
    moved[..., layout.MAP_STOP_POINT] = 0.0
    moved[has_stop_point, layout.MAP_STOP_POINT] = positions_to_frame(
        stop_points[has_stop_point], frame
    )
    return moved


def to_world(positions: ndarray, agent: Frame, scene: Frame) -> ndarray:
    """Agent-frame positions out to the world, through the scene frame."""
    scene_positions = positions_from_frame(positions, agent)
    return positions_from_frame(scene_positions, scene)


# ------------------------------------------------------------------
# THE THREE FRAMES
#
#   world            Waymo's own coordinates
#     |  origin and heading of the self-driving car, current step
#     v
#   scene            staged files are stored here (womd/store.py)
#     |  origin and heading of one predicted agent, current step
#     v
#   agent            the model reads and predicts here
#
#   Predictions travel back up, agent -> scene -> world (to_world).
# ------------------------------------------------------------------
