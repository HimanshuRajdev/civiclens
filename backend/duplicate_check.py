"""
CivicLens - Duplicate Detection
Checks if a new complaint is a duplicate of an existing one
using location proximity and issue type matching.
"""

from database import get_all_complaints
from math import radians, sin, cos, sqrt, atan2

# If same issue type reported within this radius (meters), flag as duplicate
DUPLICATE_RADIUS_METERS = 50


def haversine(lat1, lng1, lat2, lng2) -> float:
    """Returns distance in meters between two coordinates."""
    R = 6371000  # Earth radius in meters
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def check_duplicate(issue_type: str, lat: float, lng: float) -> dict:
    """
    Check if a similar complaint already exists nearby.
    Returns { is_duplicate, existing_id, distance_meters }
    """

    # If no location provided, skip duplicate check
    if lat == 0 and lng == 0:
        return {"is_duplicate": False, "existing_id": None, "distance_meters": None}

    existing = get_all_complaints()

    for complaint in existing:
        # Only compare same issue type
        if complaint.get("issue_type") != issue_type:
            continue

        # Skip resolved complaints
        if complaint.get("status") == "Resolved":
            continue

        clat = complaint.get("lat", 0)
        clng = complaint.get("lng", 0)

        if clat == 0 and clng == 0:
            continue

        distance = haversine(lat, lng, clat, clng)

        if distance <= DUPLICATE_RADIUS_METERS:
            return {
                "is_duplicate": True,
                "existing_id": complaint["id"],
                "distance_meters": round(distance, 1),
            }

    return {"is_duplicate": False, "existing_id": None, "distance_meters": None}