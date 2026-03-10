# PufferDrive type constants

# Agent types
VEHICLE = 1
PEDESTRIAN = 2
CYCLIST = 3
OTHER = 4

# Lane types (0-9 range)
LANE_UNKNOWN = 0
LANE_FREEWAY = 1
LANE_SURFACE_STREET = 2
LANE_BIKE_LANE = 3

# Road line types (10-19 range)
ROAD_LINE_UNKNOWN = 10
ROAD_LINE_BROKEN_SINGLE_WHITE = 11
ROAD_LINE_SOLID_SINGLE_WHITE = 12
ROAD_LINE_SOLID_DOUBLE_WHITE = 13
ROAD_LINE_BROKEN_SINGLE_YELLOW = 14
ROAD_LINE_BROKEN_DOUBLE_YELLOW = 15
ROAD_LINE_SOLID_SINGLE_YELLOW = 16
ROAD_LINE_SOLID_DOUBLE_YELLOW = 17
ROAD_LINE_PASSING_DOUBLE_YELLOW = 18

# Road edge types (20-29 range)
ROAD_EDGE_UNKNOWN = 20
ROAD_EDGE_BOUNDARY = 21
ROAD_EDGE_MEDIAN = 22
ROAD_EDGE_SIDEWALK = 23

# Misc road features (30+ range)
CROSSWALK = 31
SPEED_BUMP = 32
DRIVEWAY = 33

# Traffic light state values
TL_STATE_UNKNOWN = 0
TL_STATE_GREEN = 3
TL_STATE_YELLOW = 2
TL_STATE_RED = 4

# Traffic control types
TRAFFIC_LIGHT = 1
STOP_SIGN = 2
YIELD_SIGN = 3
SPEED_LIMIT_SIGN = 4

# Extension traffic control types (not yet in PufferDrive spec)
TRAFFIC_CONE = 5
TRAFFIC_BARRIER = 6
GUARDRAIL = 7


# Range check helpers
def is_road_lane(t):
    return 0 <= t <= 9


def is_road_line(t):
    return 10 <= t <= 19


def is_road_edge(t):
    return 20 <= t <= 29
