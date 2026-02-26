import requests
import csv
import json
from io import StringIO

AIRPORTS_URL = "https://ourairports.com/data/airports.csv"
RUNWAYS_URL = "https://ourairports.com/data/runways.csv"


def parse_russian_civil_airports():
    print("Загрузка данных аэропортов...")
    airports_response = requests.get(AIRPORTS_URL)
    runways_response = requests.get(RUNWAYS_URL)

    airports_data = list(csv.DictReader(StringIO(airports_response.text)))
    runways_data = list(csv.DictReader(StringIO(runways_response.text)))

    civil_types = {
        "large_airport",
        "medium_airport",
        "small_airport"
    }

    ru_airports = [
        a for a in airports_data
        if a["iso_country"] == "RU"
        and a["type"] in civil_types
        and a["scheduled_service"] == "yes"
    ]

    runways_by_airport = {}
    for r in runways_data:
        ident = r["airport_ident"]
        if ident not in runways_by_airport:
            runways_by_airport[ident] = []
        runways_by_airport[ident].append(r)

    result = []

    for airport in ru_airports:
        ident = airport["ident"]
        runways = runways_by_airport.get(ident, [])

        max_length = 0

        for r in runways:
            try:
                length = int(r["length_ft"])
                length_m = length * 0.3048
                if length_m > max_length:
                    max_length = length_m
            except:
                continue

        # Удаляем записи с ВПП < 100 м
        if max_length < 100:
            continue

        classification = "Long" if max_length >= 3000 else "Short"

        result.append({
            "name": airport["name"],
            "icao": airport["ident"],
            "iata": airport["iata_code"],
            "lat": float(airport["latitude_deg"]),
            "lon": float(airport["longitude_deg"]),
            "max_runway_m": round(max_length, 1),
            "class": classification
        })

    with open("russian_civil_airports.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("Файл russian_civil_airports.json сохранён")


if __name__ == "__main__":
    parse_russian_civil_airports()