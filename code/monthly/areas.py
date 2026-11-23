"""Contributing areas used by the monthly zone and mainstream scripts."""

from math import isfinite

# Enter the actual contributing areas in km². 输入实际面积（km²）。
STATION_AREAS_KM2 = {
    "pingshan": None,
    "cuntan": None,
    "yichang": None,
    "hankou": None,
    "datong": None,
}

# Enter the actual independent zone areas in km². 输入实际面积（km²）。
ZONE_AREAS_KM2 = {
    "Z1": None,
    "Z2": None,
    "Z3": None,
    "Z4": None,
    "Z5": None,
}


def validate_areas(areas_km2: dict[str, int | float | None]) -> None:
    invalid = [
        name
        for name, area in areas_km2.items()
        if type(area) not in (int, float) or not isfinite(area) or area <= 0
    ]
    if invalid:
        raise ValueError(
            "Enter the actual positive areas in km² in areas.py for: " + ", ".join(invalid)
        )
