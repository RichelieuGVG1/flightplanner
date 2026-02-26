"""
filter_waypoints.py
Фильтрует waypoints, исключая те, что попадают внутрь запретных зон.
Граница зоны строится по прямым в проекции Меркатора.
"""

import json
from pathlib import Path

# Paths relative to this script
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent

WAYPOINTS_FILE      = PROJECT_DIR / "russia_waypoints.json"
ZONES_FILE          = BASE_DIR / "prohibited_zones.json"
OUTPUT_FILE         = BASE_DIR / "allowed_to_use_waypoints.json"


def point_in_polygon(lon: float, lat: float, points: list) -> bool:
    """
    Ray-casting алгоритм.
    points — список {"lat": ..., "lon": ...}, граница по прямым в Меркаторе.
    """
    n = len(points)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = points[i]["lon"], points[i]["lat"]
        xj, yj = points[j]["lon"], points[j]["lat"]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def filter_waypoints():
    waypoints = json.loads(Path(WAYPOINTS_FILE).read_text(encoding="utf-8"))
    zones_raw = json.loads(Path(ZONES_FILE).read_text(encoding="utf-8"))

    # Поддержка как {"zones": [...]} так и просто [...]
    if isinstance(zones_raw, dict):
        zones = zones_raw.get("zones", [])
    else:
        zones = zones_raw

    print(f"Waypoints: {len(waypoints)}")
    print(f"Запретных зон: {len(zones)}")

    allowed = []
    blocked = 0

    for wp in waypoints:
        lat, lon = wp["lat"], wp["lon"]
        in_zone = False
        for zone in zones:
            pts = zone.get("points") or zone.get("coordinates", [])
            # coordinates могут быть {"lon":..,"lat":..} или {"lat":..,"lon":..}
            if point_in_polygon(lon, lat, pts):
                in_zone = True
                break
        if in_zone:
            blocked += 1
        else:
            allowed.append(wp)

    Path(OUTPUT_FILE).write_text(
        json.dumps(allowed, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\nЗаблокировано: {blocked} точек")
    print(f"Разрешено:     {len(allowed)} точек")
    print(f"Сохранено → {OUTPUT_FILE}")


if __name__ == "__main__":
    filter_waypoints()
