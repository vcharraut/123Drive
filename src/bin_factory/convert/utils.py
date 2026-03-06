import numpy as np
from py123d.datatypes.map_objects import LaneType


def extract_lane_centers(static_map_elements: dict) -> tuple[list, np.ndarray, dict, np.ndarray]:
    """
    Extract lane center information as numpy arrays for vectorized operations.

    Args:
        static_map_elements: Dict of static map elements

    Returns:
        Tuple of:
        - lane_ids: List of lane IDs (strings)
        - lane_polylines: Array of lane polylines, padded to max length (N_lanes, max_points, 2)
        - lane_metadata: Dict mapping lane_id to connectivity info
        - lane_lengths: Array of actual polyline lengths (N_lanes,) for each lane
    """
    lane_ids = []
    lane_polylines_list = []
    lane_lengths_list = []
    lane_metadata = {}
    max_points = 0

    # First pass: collect lanes and find max polyline length
    for element_id, element_data in static_map_elements.items():
        element_type = element_data["type"]

        # Only process lane centers
        if element_type in (LaneType.SURFACE_STREET, LaneType.FREEWAY):
            polyline = element_data["polyline"]

            if len(polyline) > 0:
                # Convert to 2D if needed
                polyline_2d = polyline[:, :2] if polyline.shape[1] == 3 else polyline

                lane_ids.append(element_id)
                lane_polylines_list.append(polyline_2d)
                lane_lengths_list.append(len(polyline_2d))
                max_points = max(max_points, len(polyline_2d))

                lane_metadata[element_id] = {
                    "entry_lanes": element_data["entry_lanes"],
                    "exit_lanes": element_data["exit_lanes"],
                }

    if not lane_ids:
        return [], np.array([]), {}, np.array([])

    # Second pass: create padded array
    n_lanes = len(lane_ids)
    lane_polylines = np.zeros((n_lanes, max_points, 2), dtype=np.float64)
    lane_lengths = np.array(lane_lengths_list, dtype=np.int64)

    for i, polyline_2d in enumerate(lane_polylines_list):
        lane_polylines[i, : len(polyline_2d), :] = polyline_2d

    return lane_ids, lane_polylines, lane_metadata, lane_lengths
