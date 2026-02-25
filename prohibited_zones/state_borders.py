import requests
import json


def parse_russia_border(output_file="state_border.json"):
    url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    russia_geometry = None

    for feature in data["features"]:
        props = feature.get("properties", {})
        country_name = props.get("ADMIN") or props.get("name")

        if country_name == "Russia":
            russia_geometry = feature["geometry"]
            break

    if not russia_geometry:
        raise ValueError("Russia not found in dataset")

    border_polygons = []

    if russia_geometry["type"] == "Polygon":
        polygons = [russia_geometry["coordinates"]]
    elif russia_geometry["type"] == "MultiPolygon":
        polygons = russia_geometry["coordinates"]
    else:
        raise ValueError("Unexpected geometry type")

    for polygon in polygons:
        # Каждая запись - это список колец (внешнее + внутренние)
        for ring in polygon:
            ring_coords = [[lat, lon] for lon, lat in ring]
            border_polygons.append(ring_coords)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(border_polygons, f, ensure_ascii=False, indent=2)

    total_points = sum(len(r) for r in border_polygons)
    print(f"Saved {len(border_polygons)} rings with {total_points} points to {output_file}")


if __name__ == "__main__":
    parse_russia_border()