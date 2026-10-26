"""The fixed layout every step agrees on: sequence lengths, the columns of
an agent row and a map row, the vocabularies their one-hot blocks use and
the code version stamped into every artifact. Constants only.
"""

# A scenario is 91 steps at 10 Hz, 11 of history ending at the
# current step and then the 80 future steps to predict.
HISTORY_STEPS = 11
FUTURE_STEPS = 80
CURRENT_STEP_INDEX = 10
TOTAL_STEPS = HISTORY_STEPS + FUTURE_STEPS
TIMESTEP_SECONDS = 0.1
FUTURE_HORIZON_SECONDS = FUTURE_STEPS * TIMESTEP_SECONDS
# Bounds how far an agent can travel inside the horizon, which
# sets the scale its anchors are fitted at.
MAXIMUM_ACCELERATION_METRES_PER_SECOND_SQUARED = 3.56

# Agent type vocabulary the model predicts, plus an "other"
# catch-all used only for staged agent features.
PREDICTED_OBJECT_TYPES = ("TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST")
NUM_OBJECT_TYPES = len(PREDICTED_OBJECT_TYPES)
AGENT_TYPES = PREDICTED_OBJECT_TYPES + ("TYPE_OTHER",)
NUM_AGENT_TYPES = len(AGENT_TYPES)

# Staging and model sizing constants, and the version stamp
# checked against each staged artifact's provenance record.
STAGING_CROP_RADIUS_METRES = 400.0
MAP_POINT_SPACING_METRES = 1.0
MAP_CHUNK_DOTS = 20
NUM_PREDICTED_MODES = 6
STAGING_CODE_VERSION = "2026-08-19-a"
# The anchor fit's own stamp, changed whenever fit_anchors.py would give
# other anchors from the same staged files. 2026-10-26: designated targets
# only, tracks ending before 8 s left out.
ANCHOR_FIT_VERSION = "2026-10-26-8s-endpoints"

# Waymo scores 16 of the 80 future steps, every fifth one, the
# first being scenario step 15, future index 4.
SUBMISSION_STEPS = 16
SUBMISSION_STEP_STRIDE = FUTURE_STEPS // SUBMISSION_STEPS
SUBMISSION_FIRST_FUTURE_INDEX = 15 - (CURRENT_STEP_INDEX + 1)
SUBMISSION_FUTURE_INDICES = tuple(
    range(SUBMISSION_FIRST_FUTURE_INDEX, FUTURE_STEPS, SUBMISSION_STEP_STRIDE)
)

# Input scales. Each is the standard deviation of that quantity
# as the model reads it, measured on every 60th training scenario.
# All distances share the map-position scale, so geometry keeps
# its proportions.
DISTANCE_NORMALISER_METRES = 65.8
VELOCITY_NORMALISER_METRES_PER_SECOND = 3.9
DIMENSION_NORMALISER_METRES = 1.6
# Measured over lane dots that carry a posted limit above zero.
SPEED_LIMIT_NORMALISER_MILES_PER_HOUR = 12.9

# Column layout of one agent feature row, in order. Position, heading
# as cosine and sine, velocity, box size, one-hot type, is-SDC flag.
AGENT_POSITION = slice(0, 2)
AGENT_HEADING_COSINE = 2
AGENT_HEADING_SINE = 3
AGENT_VELOCITY = slice(4, 6)
AGENT_DIMENSIONS = slice(6, 8)
AGENT_TYPE = slice(8, 8 + NUM_AGENT_TYPES)
AGENT_IS_SDC = AGENT_TYPE.stop
AGENT_FEATURE_DIM = AGENT_IS_SDC + 1

# Traffic signal state vocabulary, one-hot per history step in a
# lane's signal history.
TRAFFIC_SIGNAL_STATES = (
    "LANE_STATE_UNKNOWN",
    "LANE_STATE_ARROW_STOP",
    "LANE_STATE_ARROW_CAUTION",
    "LANE_STATE_ARROW_GO",
    "LANE_STATE_STOP",
    "LANE_STATE_CAUTION",
    "LANE_STATE_GO",
    "LANE_STATE_FLASHING_STOP",
    "LANE_STATE_FLASHING_CAUTION",
)
NUM_TRAFFIC_SIGNAL_STATES = len(TRAFFIC_SIGNAL_STATES)
SIGNAL_HISTORY_SHAPE = (HISTORY_STEPS, NUM_TRAFFIC_SIGNAL_STATES)

# Vocabularies for the one-hot blocks of a map dot row, the kind of
# feature a dot belongs to and then the lane, line and edge types.
MAP_POLYLINE_KINDS = (
    "lane",
    "road_line",
    "road_edge",
    "stop_sign",
    "crosswalk",
    "speed_bump",
    "driveway",
)
NUM_MAP_POLYLINE_KINDS = len(MAP_POLYLINE_KINDS)

LANE_TYPES = (
    "TYPE_UNDEFINED",
    "TYPE_FREEWAY",
    "TYPE_SURFACE_STREET",
    "TYPE_BIKE_LANE",
)
NUM_LANE_TYPES = len(LANE_TYPES)

ROAD_LINE_TYPES = (
    "TYPE_UNKNOWN",
    "TYPE_BROKEN_SINGLE_WHITE",
    "TYPE_SOLID_SINGLE_WHITE",
    "TYPE_SOLID_DOUBLE_WHITE",
    "TYPE_BROKEN_SINGLE_YELLOW",
    "TYPE_BROKEN_DOUBLE_YELLOW",
    "TYPE_SOLID_SINGLE_YELLOW",
    "TYPE_SOLID_DOUBLE_YELLOW",
    "TYPE_PASSING_DOUBLE_YELLOW",
)

ROAD_EDGE_TYPES = (
    "TYPE_UNKNOWN",
    "TYPE_ROAD_EDGE_BOUNDARY",
    "TYPE_ROAD_EDGE_MEDIAN",
)
# Road lines and road edges share one boundary-type one-hot,
# lines first.
BOUNDARY_TYPES = ROAD_LINE_TYPES + ROAD_EDGE_TYPES
NUM_BOUNDARY_TYPES = len(BOUNDARY_TYPES)

# Column layout of one map dot row, 32 columns, in order. Columns
# that do not apply to a dot's kind stay zero.
MAP_POSITION = slice(0, 2)
MAP_DIRECTION = slice(2, 4)
MAP_KIND = slice(4, 4 + NUM_MAP_POLYLINE_KINDS)
MAP_LANE_TYPE = slice(MAP_KIND.stop, MAP_KIND.stop + NUM_LANE_TYPES)
MAP_SPEED_LIMIT = MAP_LANE_TYPE.stop
MAP_BOUNDARY_TYPE = slice(
    MAP_SPEED_LIMIT + 1, MAP_SPEED_LIMIT + 1 + NUM_BOUNDARY_TYPES
)
MAP_STOP_POINT = slice(MAP_BOUNDARY_TYPE.stop, MAP_BOUNDARY_TYPE.stop + 2)
# The two crossing codes stay last because the map encoder slices
# them off the end of the row and one-hot encodes them.
MAP_LEFT_BOUNDARY_CROSSING = MAP_STOP_POINT.stop
MAP_RIGHT_BOUNDARY_CROSSING = MAP_LEFT_BOUNDARY_CROSSING + 1
MAP_FEATURE_DIM = MAP_RIGHT_BOUNDARY_CROSSING + 1
# A crossing code is 0 for no boundary, else 1 + the road line
# type of the boundary on that side of the lane.
NUM_BOUNDARY_CROSSING_CODES = len(ROAD_LINE_TYPES) + 1

# One signal history per polyline, flattened to a one-hot signal
# state for each history step.
POLYLINE_SIGNAL_DIM = HISTORY_STEPS * NUM_TRAFFIC_SIGNAL_STATES

LANE_SIDES = ("left", "right")

# The history steps, ending at the current one, and the future steps
# of a 91-step track.
HISTORY = slice(0, HISTORY_STEPS)
FUTURE = slice(CURRENT_STEP_INDEX + 1, None)


# ------------------------------------------------------------------
# THE TWO ROW LAYOUTS the slices above describe.
#
# One agent row, per 0.1 s step.
#
#     0   1   2   3   4   5   6   7   8   9  10  11  12
#   +-------+---+---+-------+-------+---------------+---+
#   | x   y |cos|sin|vx  vy |len wid|veh ped cyc oth|sdc|
#   +-------+---+---+-------+-------+---------------+---+
#
# One map row, per metre of a map feature.
#
#     0-1   2-3    4-10    11-14   15    16-27   28-29  30  31
#   +-----+-----+--------+-------+-----+--------+------+---+---+
#   | x y |dx dy| kind   | lane  |speed|boundary| stop | L | R |
#   |     |     | 1-hot  | type  |limit| type   | x  y |   |   |
#   +-----+-----+--------+-------+-----+--------+------+---+---+
#
#   L and R are the marking on the lane's left and right, 0 for none.
# ------------------------------------------------------------------
