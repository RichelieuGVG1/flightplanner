"""
Агентная среда маршрутизации самолётов с обучением с подкреплением (Q-learning).

Входные данные:
  simulation_result.json        — маршруты и встречные самолёты (5 бортов, 100 меток t)
  russian_civil_airports.json   — запасные аэропорты
  prohibited_zones.json         — запретные зоны (полигоны)
  allowed_to_use_waypoints.json — разрешённые waypoints
  weather_data.json             — погода: ветер, гроза, турбулентность, обледенение (5 эшелонов, 100 t)

Два самолёта-агента обучаются независимо методом Q-learning минимизировать
суммарный расход топлива при соблюдении всех ограничений.
"""

from __future__ import annotations

import json
import math
import random
import pickle
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# КОНСТАНТЫ ШТРАФОВ
# ---------------------------------------------------------------------------
PENALTY_PROHIBITED_ZONE          = 10_000  # пролёт через запретную зону
PENALTY_NOT_WAYPOINT             = 500     # следующий узел не из разрешённого списка
PENALTY_ALTITUDE_CHANGE_1LVL    = 100     # смена эшелона на 1 уровень вверх/вниз
PENALTY_ALTITUDE_CHANGE_2LVL    = 500     # смена эшелона на 2+ уровня (за каждый уровень)
PENALTY_WEATHER_TURBULENCE_BASE = 50      # турбулентность: × уровень turbulence
PENALTY_WEATHER_ICE_BASE        = 40      # обледенение: × уровень ice
PENALTY_WEATHER_STORM_BASE      = 80      # гроза: × storm_power
PENALTY_CONFLICT_LATERAL        = 200     # горизонтальный конфликт (<300 км, тот же эшелон)
PENALTY_CONFLICT_1LVL           = 300     # обход встречного через 1 эшелон
PENALTY_CONFLICT_2LVL           = 1_500   # обход встречного через 2+ эшелона (за каждый)
PENALTY_LOW_FUEL                = 5_000   # нехватка резервного топлива для ухода на запасной
PENALTY_DIVERT_KM               = 50      # лишний км отклонения от ортодромии
PENALTY_LOW_ALT_PER_STEP        = 300     # долгое нахождение на эшелоне 1 (за каждый шаг)
PENALTY_LOW_ALT_THRESHOLD       = 3       # через сколько шагов на эшелоне 1 начинается штраф

# ---------------------------------------------------------------------------
# ПАРАМЕТРЫ САМОЛЁТОВ (реальные ТТХ)
# ---------------------------------------------------------------------------
AIRCRAFT_PRESETS: Dict[str, Dict] = {

    # Airbus A350-900 | MTOW 280 000 кг | OEW 142 400 кг | топливо 127 033 кг
    # крейсер 903 км/ч, расход пустого ~6.98 кг/км
    "A350": {
        "full_name":            "Airbus A350-900",
        "max_passengers":       440,
        "empty_mass_kg":        142_400,
        "max_payload_kg":       53_000,
        "max_fuel_kg":          127_033,
        "fuel_per_km_empty_kg": 6.98,
        "fuel_climb_per_level": 480,
        "cruise_speed_kmh":     903,
        "min_runway_m":         2_600,
        "runway_class":         "Long",
        "max_altitude_level":   5,
    },

    # Boeing 737-800 | MTOW 79 016 кг | OEW 41 413 кг | топливо 20 810 кг
    # крейсер 842 км/ч, расход пустого ~2.9 кг/км
    "B738": {
        "full_name":            "Boeing 737-800",
        "max_passengers":       189,
        "empty_mass_kg":        41_413,
        "max_payload_kg":       20_800,
        "max_fuel_kg":          20_810,
        "fuel_per_km_empty_kg": 2.9,
        "fuel_climb_per_level": 150,
        "cruise_speed_kmh":     842,
        "min_runway_m":         2_090,
        "runway_class":         "Long",
        "max_altitude_level":   5,
    },

    # Sukhoi Superjet 100-95B | MTOW 49 450 кг | OEW 24 250 кг | топливо 12 700 кг
    # крейсер 841 км/ч, расход пустого ~2.2 кг/км
    "SSJ100": {
        "full_name":            "Sukhoi Superjet 100-95B",
        "max_passengers":       103,
        "empty_mass_kg":        24_250,
        "max_payload_kg":       12_245,
        "max_fuel_kg":          12_700,
        "fuel_per_km_empty_kg": 2.2,
        "fuel_climb_per_level": 100,
        "cruise_speed_kmh":     841,
        "min_runway_m":         1_731,
        "runway_class":         "Long",
        "max_altitude_level":   5,
    },

    # Airbus A321-200 | MTOW 93 500 кг | OEW 48 000 кг | топливо 24 000 кг
    # крейсер 833 км/ч, расход пустого ~3.6 кг/км
    "A321": {
        "full_name":            "Airbus A321-200",
        "max_passengers":       220,
        "empty_mass_kg":        48_000,
        "max_payload_kg":       25_300,
        "max_fuel_kg":          24_000,
        "fuel_per_km_empty_kg": 3.6,
        "fuel_climb_per_level": 160,
        "cruise_speed_kmh":     833,
        "min_runway_m":         2_200,
        "runway_class":         "Long",
        "max_altitude_level":   5,
    },

    # Boeing 777-300ER | MTOW 351 500 кг | OEW 167 800 кг | топливо 145 540 кг
    # крейсер 905 км/ч, расход пустого ~9.4 кг/км
    "B773ER": {
        "full_name":            "Boeing 777-300ER",
        "max_passengers":       550,
        "empty_mass_kg":        167_800,
        "max_payload_kg":       69_900,
        "max_fuel_kg":          145_540,
        "fuel_per_km_empty_kg": 9.4,
        "fuel_climb_per_level": 600,
        "cruise_speed_kmh":     905,
        "min_runway_m":         3_050,
        "runway_class":         "Long",
        "max_altitude_level":   5,
    },

    # ATR 72-600 | MTOW 23 000 кг | OEW 13 500 кг | топливо 5 000 кг
    # крейсер 510 км/ч, расход пустого ~0.85 кг/км
    "ATR72": {
        "full_name":            "ATR 72-600",
        "max_passengers":       78,
        "empty_mass_kg":        13_500,
        "max_payload_kg":       7_500,
        "max_fuel_kg":          5_000,
        "fuel_per_km_empty_kg": 0.85,
        "fuel_climb_per_level": 50,
        "cruise_speed_kmh":     510,
        "min_runway_m":         1_200,
        "runway_class":         "Short",
        "max_altitude_level":   5,
    },
}

AIRCRAFT_TYPE_ALIAS: Dict[str, str] = {
    "Airbus A350-900":          "A350",
    "Airbus A320-200":          "A350",   # заменён на A350 по заданию
    "Boeing 737-800":           "B738",
    "Sukhoi Superjet 100-95B":  "SSJ100",
    "Airbus A321-200":          "A321",
    "Boeing 777-300ER":         "B773ER",
    "ATR 72-600":               "ATR72",
    **{k: k for k in ["A350", "B738", "SSJ100", "A321", "B773ER", "ATR72"]},
}

# ---------------------------------------------------------------------------
# МАРШРУТЫ ДЛЯ ОБУЧЕНИЯ
# ---------------------------------------------------------------------------
# Москва (SVO) 55.9726N 37.4146E → Владивосток (VVO) 43.3989N 132.1478E ~6 430 км
# СПб (LED) 59.8003N 30.2625E → Петропавловск (PKC) 53.1679N 158.4539E ~7 720 км
# Waypoints взяты из allowed_to_use_waypoints.json; t — шаг из 100

TRAINING_SIMULATION = [
    {
        "plane_number": 1,
        "departure": {"name": "Moskva_SVO", "lat": 55.9726, "lon": 37.4146},
        "arrival":   {"name": "Vladivostok_VVO", "lat": 43.3989, "lon": 132.1478},
        "gc_distance_km": 6430.0,
        "max_deviation_km": 400.0,
        "corridor_km": 1000,
        "start_t": 1,
        "altitude_level": 1,
        "route_waypoints": [
            {"name": "Moskva_SVO", "lat": 55.9726, "lon":  37.4146, "t":  1},
            {"name": "ABALI",      "lat": 55.5454, "lon":  68.0205, "t": 10},
            {"name": "NOXOR",      "lat": 55.7500, "lon":  82.1400, "t": 22},
            {"name": "ABAGO",      "lat": 56.2920, "lon": 144.2381, "t": 65},
            {"name": "Vladivostok_VVO", "lat": 43.3989, "lon": 132.1478, "t": 85},
        ],
    },
    {
        "plane_number": 2,
        "departure": {"name": "SPb_LED", "lat": 59.8003, "lon":  30.2625},
        "arrival":   {"name": "PKC",      "lat": 53.1679, "lon": 158.4539},
        "gc_distance_km": 7720.0,
        "max_deviation_km": 450.0,
        "corridor_km": 1000,
        "start_t": 1,
        "altitude_level": 1,
        "route_waypoints": [
            {"name": "SPb_LED", "lat": 59.8003, "lon":  30.2625, "t":  1},
            {"name": "ABARA",   "lat": 61.1836, "lon":  50.8481, "t":  8},
            {"name": "NOXOR",   "lat": 55.7500, "lon":  82.1400, "t": 22},
            {"name": "ABAGO",   "lat": 56.2920, "lon": 144.2381, "t": 68},
            {"name": "PKC",     "lat": 53.1679, "lon": 158.4539, "t": 92},
        ],
    },
]

AGENT_CONFIGS = [
    # Агент 1 — Boeing 777-300ER | 80% загрузка: 317 пасс, багаж 317×23=7 291 кг
    {"aircraft_type": "B773ER", "passengers": 317, "baggage_kg": 7_291, "min_reserve_fuel_kg": 3_000},
    # Агент 2 — Airbus A350-900 | 80% загрузка: 352 пасс, багаж 352×23=8 096 кг
    {"aircraft_type": "A350",   "passengers": 352, "baggage_kg": 8_096, "min_reserve_fuel_kg": 3_000},
]

# ---------------------------------------------------------------------------
# ГЕОДЕЗИЯ
# ---------------------------------------------------------------------------
EARTH_R = 6_371.0

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние по большому кругу (км)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(max(0, min(1, a))))

def azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Истинный курс (0–360°)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return math.degrees(math.atan2(x, y)) % 360

def angle_diff(a: float, b: float) -> float:
    d = (a - b + 360) % 360
    return d - 360 if d > 180 else d

# ---------------------------------------------------------------------------
# ГЕОМЕТРИЯ ПОЛИГОНОВ
# ---------------------------------------------------------------------------
def point_in_polygon(lat: float, lon: float, polygon: List[Dict]) -> bool:
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]["lon"], polygon[i]["lat"]
        xj, yj = polygon[j]["lon"], polygon[j]["lat"]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside

def segment_crosses_polygon(lat1: float, lon1: float, lat2: float, lon2: float,
                             polygon: List[Dict], samples: int = 16) -> bool:
    for i in range(samples + 1):
        t = i / samples
        if point_in_polygon(lat1 + t * (lat2 - lat1), lon1 + t * (lon2 - lon1), polygon):
            return True
    return False

# ---------------------------------------------------------------------------
# ПОГОДНАЯ БД
# ---------------------------------------------------------------------------
class WeatherDB:
    def __init__(self, records: List[Dict]):
        self._data: Dict[Tuple, Dict] = {}
        self._wp_coords: Dict[str, Tuple[float, float]] = {}
        for r in records:
            self._data[(r["name"], r["z"], r["t"])] = r
            self._wp_coords[r["name"]] = (r["lat"], r["lon"])

    def get(self, lat: float, lon: float, z: int, t: int) -> Optional[Dict]:
        """Возвращает запись ближайшего погодного узла для (z, t)."""
        best_dist, best = float("inf"), None
        for (name, rz, rt), rec in self._data.items():
            if rz != z or rt != t:
                continue
            wlat, wlon = self._wp_coords[name]
            d = haversine(lat, lon, wlat, wlon)
            if d < best_dist:
                best_dist, best = d, rec
        return best

    def wind_effect(self, aircraft_az: float, wind_dir: float, wind_speed: float) -> float:
        """
        Коэффициент расхода топлива от ветра.
        Встречный ветер (diff≈180°) → >1, попутный (diff≈0°) → <1.
        """
        diff = angle_diff(aircraft_az, (wind_dir + 180) % 360)
        cos_val = math.cos(math.radians(diff))
        effect = 1.0 - cos_val * wind_speed * 0.005
        return max(0.5, min(2.0, effect))

# ---------------------------------------------------------------------------
# МОДЕЛЬ САМОЛЁТА
# ---------------------------------------------------------------------------
@dataclass
class Aircraft:
    plane_id:            int
    aircraft_type:       str
    departure_name:      str
    arrival_name:        str
    departure_lat:       float
    departure_lon:       float
    arrival_lat:         float
    arrival_lon:         float
    passengers:          int   = 180
    baggage_kg:          float = 2_000
    min_reserve_fuel_kg: float = 3_000
    altitude_level:      int   = 1
    preset: Dict = field(default_factory=dict)

    def __post_init__(self):
        key = AIRCRAFT_TYPE_ALIAS.get(self.aircraft_type, self.aircraft_type)
        if key not in AIRCRAFT_PRESETS:
            raise ValueError(f"Неизвестный тип ВС: '{self.aircraft_type}'")
        self.aircraft_type = key
        self.preset = deepcopy(AIRCRAFT_PRESETS[key])

    @property
    def passenger_mass_kg(self) -> float:
        return self.passengers * 80.0

    @property
    def total_payload_kg(self) -> float:
        return self.passenger_mass_kg + self.baggage_kg

    @property
    def max_fuel_kg(self) -> float:
        available = self.preset["max_payload_kg"] - self.total_payload_kg
        return min(self.preset["max_fuel_kg"], max(0.0, available + self.preset["max_fuel_kg"]))

    @property
    def total_mass_kg(self) -> float:
        return self.preset["empty_mass_kg"] + self.total_payload_kg + self.max_fuel_kg

    @property
    def mass_ratio(self) -> float:
        return self.total_mass_kg / self.preset["empty_mass_kg"]

    def fuel_per_km(self, wind_effect: float = 1.0) -> float:
        """Расход (кг/км) с учётом загрузки и ветра."""
        return self.preset["fuel_per_km_empty_kg"] * self.mass_ratio * wind_effect

    def fuel_for_level_change(self, levels: int) -> float:
        """Расход на смену n эшелонов (кг)."""
        n = abs(levels)
        if n == 0:
            return 0.0
        base = self.preset["fuel_climb_per_level"] * self.mass_ratio
        return base * n * (1.5 if n >= 2 else 1.0)

    def can_land(self, airport: Dict) -> bool:
        if self.preset["runway_class"] == "Short":
            return True
        return airport.get("max_runway_m", 0) >= self.preset["min_runway_m"]

# ---------------------------------------------------------------------------
# СОСТОЯНИЕ ПОЛЁТА
# ---------------------------------------------------------------------------
@dataclass
class FlightState:
    plane_id:          int
    current_lat:       float
    current_lon:       float
    current_t:         int
    altitude_level:    int
    fuel_remaining_kg: float
    total_penalty:     float = 0.0
    steps_at_level1:   int   = 0
    penalty_log:       List[Dict] = field(default_factory=list)
    path:              List[Dict] = field(default_factory=list)

    def add_penalty(self, reason: str, value: float):
        self.total_penalty += value
        self.penalty_log.append({
            "reason": reason, "value": value,
            "t": self.current_t,
            "lat": self.current_lat, "lon": self.current_lon,
        })

# ---------------------------------------------------------------------------
# СРЕДА (Environment)
# ---------------------------------------------------------------------------
class FlightEnvironment:
    """
    Среда для одного самолёта-агента.
    Поддерживает сброс (reset) и шаг (step) в стиле Gym.
    """

    # Действия: (delta_altitude, next_wp_offset)
    # delta_altitude: -1, 0, +1 (смена эшелона)
    # next_wp_offset: всегда 1 (следующий waypoint по маршруту) — агент управляет только эшелоном
    N_ACTIONS = 3  # 0: снизиться, 1: держать эшелон, 2: набрать высоту

    def __init__(self,
                 sim_data:       Dict,
                 aircraft:       Aircraft,
                 weather_db:     WeatherDB,
                 airports:       List[Dict],
                 prohibited:     List[Dict],
                 allowed_wp:     Dict[str, Dict],
                 other_planes:   List[Dict]):

        self.sim_data     = sim_data
        self.aircraft     = aircraft
        self.weather_db   = weather_db
        self.airports     = airports
        self.prohibited   = prohibited
        self.allowed_wp   = allowed_wp
        self.other_planes = other_planes

        self.waypoints: List[Dict] = sim_data["route_waypoints"]
        self.gc_dist = sim_data["gc_distance_km"]
        self._reset_state()

    def _reset_state(self):
        wp0 = self.waypoints[0]
        self.state = FlightState(
            plane_id=self.aircraft.plane_id,
            current_lat=wp0["lat"],
            current_lon=wp0["lon"],
            current_t=self.sim_data.get("start_t", 1),
            altitude_level=self.sim_data.get("altitude_level", 1),
            fuel_remaining_kg=self.aircraft.max_fuel_kg,
            path=[{
                "name":            wp0.get("name", "DEP"),
                "lat":             wp0["lat"],
                "lon":             wp0["lon"],
                "t":               self.sim_data.get("start_t", 1),
                "altitude_level":  self.sim_data.get("altitude_level", 1),
                "fuel_remaining_kg": round(self.aircraft.max_fuel_kg, 1),
                "step_penalty":    0.0,
                "dist_km":         0.0,
                "fuel_burned_step_kg": 0.0,
                "wind_effect":     1.0,
            }]
        )
        self.wp_idx = 1  # текущий целевой waypoint

    def reset(self) -> Tuple:
        """Сбрасывает среду и возвращает начальное состояние (tuple для Q-table)."""
        self._reset_state()
        return self._obs()

    def _obs(self) -> Tuple:
        """
        Дискретизированное наблюдение для Q-таблицы:
        (wp_idx, altitude_level, fuel_bucket, steps_at_level1)
        fuel_bucket — запас топлива, округлённый до 10 000 кг
        """
        fuel_bucket = int(self.state.fuel_remaining_kg // 10_000)
        return (
            self.wp_idx,
            self.state.altitude_level,
            max(0, min(fuel_bucket, 20)),
            min(self.state.steps_at_level1, 10),
        )

    def step(self, action: int) -> Tuple[Tuple, float, bool]:
        """
        Выполняет один шаг.
        action: 0=снизиться, 1=держать, 2=набрать высоту
        Возвращает (next_obs, reward, done).
        """
        if self.wp_idx >= len(self.waypoints):
            return self._obs(), 0.0, True

        # Целевой waypoint
        next_wp = self.waypoints[self.wp_idx]

        # Новый эшелон по действию
        max_lvl = self.aircraft.preset["max_altitude_level"]
        delta = action - 1  # 0→-1, 1→0, 2→+1
        new_alt = max(1, min(max_lvl, self.state.altitude_level + delta))

        # Расчёт шага
        reward, done = self._compute_step(next_wp, new_alt)

        self.wp_idx += 1
        if self.wp_idx >= len(self.waypoints):
            done = True

        return self._obs(), reward, done

    def _compute_step(self, next_wp: Dict, new_alt: int) -> Tuple[float, bool]:
        """Полный расчёт одного шага: топливо + штрафы → reward."""
        s = self.state
        s.current_t += 1

        next_lat = next_wp["lat"]
        next_lon = next_wp["lon"]
        next_name = next_wp.get("name", "")
        penalty = 0.0

        # --- 1. Смена эшелона ---
        level_diff = abs(new_alt - s.altitude_level)
        fuel_climb = self.aircraft.fuel_for_level_change(level_diff)
        s.fuel_remaining_kg -= fuel_climb
        s.altitude_level = new_alt

        if level_diff == 1:
            penalty += PENALTY_ALTITUDE_CHANGE_1LVL
            s.add_penalty("Смена эшелона 1 уровень", PENALTY_ALTITUDE_CHANGE_1LVL)
        elif level_diff >= 2:
            p = PENALTY_ALTITUDE_CHANGE_2LVL * level_diff
            penalty += p
            s.add_penalty(f"Смена эшелона {level_diff} уровня", p)

        # --- 2. Штраф за долгое пребывание на эшелоне 1 ---
        if new_alt == 1:
            s.steps_at_level1 += 1
            if s.steps_at_level1 > PENALTY_LOW_ALT_THRESHOLD:
                extra = s.steps_at_level1 - PENALTY_LOW_ALT_THRESHOLD
                p = PENALTY_LOW_ALT_PER_STEP * (1 + extra * 0.5)
                penalty += p
                s.add_penalty(f"Долгое нахождение на эшелоне 1 (шаг #{s.steps_at_level1})", p)
        else:
            s.steps_at_level1 = 0

        # --- 3. Разрешённые waypoints ---
        if next_name and next_name not in self.allowed_wp:
            # Аэропорты отправления/прилёта исключаем из штрафа
            if next_name not in (self.aircraft.departure_name, self.aircraft.arrival_name):
                penalty += PENALTY_NOT_WAYPOINT
                s.add_penalty("Waypoint не из разрешённого списка", PENALTY_NOT_WAYPOINT)

        # --- 4. Запретные зоны ---
        for zone in self.prohibited:
            if segment_crosses_polygon(s.current_lat, s.current_lon,
                                        next_lat, next_lon, zone["points"]):
                penalty += PENALTY_PROHIBITED_ZONE
                s.add_penalty(f"Пролёт через запретную зону #{zone['id']}", PENALTY_PROHIBITED_ZONE)

        # --- 5. Погода ---
        weather = self.weather_db.get(s.current_lat, s.current_lon, new_alt, s.current_t)
        wind_eff = 1.0
        if weather:
            az = azimuth(s.current_lat, s.current_lon, next_lat, next_lon)
            wind_eff = self.weather_db.wind_effect(az, weather["wind_dir"], weather["wind_speed"])

            turb = weather.get("turbulence", 0)
            if turb > 0:
                p = PENALTY_WEATHER_TURBULENCE_BASE * turb
                penalty += p
                s.add_penalty(f"Турбулентность уровень {turb}", p)

            ice = weather.get("ice", 0)
            if ice > 0:
                p = PENALTY_WEATHER_ICE_BASE * ice
                penalty += p
                s.add_penalty(f"Обледенение уровень {ice}", p)

            storm = weather.get("storm_power", 0.0)
            if storm > 0:
                p = PENALTY_WEATHER_STORM_BASE * storm
                penalty += p
                s.add_penalty(f"Гроза мощность {storm:.1f}", p)

        # --- 6. Расход топлива на перелёт ---
        dist_km = haversine(s.current_lat, s.current_lon, next_lat, next_lon)
        fuel_flight = dist_km * self.aircraft.fuel_per_km(wind_eff)
        s.fuel_remaining_kg -= fuel_flight

        # --- 7. Конфликты с другими ВС (из simulation_result.json) ---
        # Для каждого встречного борта ищем его позицию на текущем шаге t.
        # Позиция интерполируется между ближайшими waypoints с t_prev <= t <= t_next.
        # Эшелон встречного: берём из его ближайшего waypoint по t (altitude_level борта
        # не меняется в simulation_result, но при расширении структуры поддержим wp-уровень).
        for other in self.other_planes:
            if other.get("plane_number") == s.plane_id:
                continue

            # Найти позицию встречного на шаге s.current_t (интерполяция)
            other_wps = other.get("route_waypoints", [])
            if not other_wps:
                continue

            # Найти пару waypoints, между которыми находится текущий t
            other_lat, other_lon, other_alt = None, None, other.get("altitude_level", 1)
            for k in range(len(other_wps)):
                wp_k = other_wps[k]
                if wp_k["t"] == s.current_t:
                    # Точное совпадение по t
                    other_lat, other_lon = wp_k["lat"], wp_k["lon"]
                    other_alt = wp_k.get("altitude_level", other.get("altitude_level", 1))
                    break
                if k + 1 < len(other_wps):
                    wp_next = other_wps[k + 1]
                    if wp_k["t"] < s.current_t < wp_next["t"]:
                        # Линейная интерполяция координат между wp_k и wp_next
                        frac = (s.current_t - wp_k["t"]) / max(1, wp_next["t"] - wp_k["t"])
                        other_lat = wp_k["lat"] + frac * (wp_next["lat"] - wp_k["lat"])
                        other_lon = wp_k["lon"] + frac * (wp_next["lon"] - wp_k["lon"])
                        other_alt = wp_k.get("altitude_level", other.get("altitude_level", 1))
                        break

            # Если борт ещё не вылетел или уже прилетел — пропускаем
            if other_lat is None:
                t_start = other_wps[0]["t"]
                t_end   = other_wps[-1]["t"]
                if s.current_t < t_start or s.current_t > t_end:
                    continue
                # Борт находится в крайней точке
                if s.current_t <= t_start:
                    other_lat, other_lon = other_wps[0]["lat"],  other_wps[0]["lon"]
                else:
                    other_lat, other_lon = other_wps[-1]["lat"], other_wps[-1]["lon"]
                other_alt = other.get("altitude_level", 1)

            d = haversine(s.current_lat, s.current_lon, other_lat, other_lon)

            if d >= 300:
                # Безопасное расстояние — штрафа нет
                continue

            # Нарушение нормы 300 км: штраф зависит от разницы эшелонов
            alt_diff = abs(new_alt - other_alt)
            if alt_diff == 0:
                # На одном эшелоне и ближе 300 км — горизонтальный конфликт
                penalty += PENALTY_CONFLICT_LATERAL
                s.add_penalty(
                    f"Горизонт. конфликт с ВС #{other['plane_number']} "
                    f"(d={d:.0f} км, эш.{new_alt})",
                    PENALTY_CONFLICT_LATERAL
                )
            elif alt_diff == 1:
                # Разошлись на 1 эшелон — допустимый обход, но со штрафом
                penalty += PENALTY_CONFLICT_1LVL
                s.add_penalty(
                    f"Обход ВС #{other['plane_number']} через 1 эшелон "
                    f"(d={d:.0f} км)",
                    PENALTY_CONFLICT_1LVL
                )
            else:
                # Разошлись на 2+ эшелона — высокий штраф за каждый лишний уровень
                p = PENALTY_CONFLICT_2LVL * alt_diff
                penalty += p
                s.add_penalty(
                    f"Обход ВС #{other['plane_number']} через {alt_diff} эшелона "
                    f"(d={d:.0f} км, штраф×{alt_diff})",
                    p
                )

        # --- 8. Контроль резервного запаса топлива ---
        alt_ap, dist_ap = self._nearest_alternate(s)
        fuel_needed = dist_ap * self.aircraft.fuel_per_km() * 1.1
        fuel_needed += self.aircraft.fuel_for_level_change(max(0, new_alt - 1))
        fuel_needed += self.aircraft.min_reserve_fuel_kg
        if s.fuel_remaining_kg < fuel_needed and dist_ap < float("inf"):
            shortage = fuel_needed - s.fuel_remaining_kg
            penalty += PENALTY_LOW_FUEL
            s.add_penalty(f"Мало топлива для ухода на запасной (не хватает {shortage:.0f} кг)", PENALTY_LOW_FUEL)

        # --- 9. Штраф за отклонение от ортодромии ---
        total_dist = sum(p["dist_km"] for p in s.path) + dist_km
        excess_km = max(0.0, total_dist - self.gc_dist)
        if excess_km > 0:
            p = PENALTY_DIVERT_KM * excess_km / max(1.0, self.gc_dist) * 10
            penalty += p

        # --- Обновляем координаты ---
        s.total_penalty += penalty
        s.current_lat = next_lat
        s.current_lon = next_lon
        s.path.append({
            "name":                next_name,
            "lat":                 next_lat,
            "lon":                 next_lon,
            "t":                   s.current_t,
            "altitude_level":      new_alt,
            "fuel_remaining_kg":   round(s.fuel_remaining_kg, 1),
            "step_penalty":        round(penalty, 1),
            "dist_km":             round(dist_km, 2),
            "fuel_burned_step_kg": round(fuel_flight + fuel_climb, 1),
            "wind_effect":         round(wind_eff, 3),
        })

        # Reward = −(топливо + штрафы) → максимизируем (т.е. минимизируем расход)
        fuel_total_step = fuel_flight + fuel_climb
        reward = -(fuel_total_step + penalty)

        done = s.fuel_remaining_kg <= 0
        return reward, done

    def _nearest_alternate(self, s: FlightState) -> Tuple[Optional[Dict], float]:
        best_ap, best_d = None, float("inf")
        for ap in self.airports:
            if not self.aircraft.can_land(ap):
                continue
            d = haversine(s.current_lat, s.current_lon, ap["lat"], ap["lon"])
            if d < best_d:
                best_d, best_ap = d, ap
        return best_ap, best_d

    def get_result(self) -> Dict:
        s = self.state
        total_dist = sum(p["dist_km"] for p in s.path)
        total_fuel = sum(p["fuel_burned_step_kg"] for p in s.path)
        return {
            "plane_id":             s.plane_id,
            "aircraft_type":        self.aircraft.aircraft_type,
            "departure":            self.aircraft.departure_name,
            "arrival":              self.aircraft.arrival_name,
            "total_distance_km":    round(total_dist, 2),
            "total_fuel_burned_kg": round(total_fuel, 1),
            "fuel_remaining_kg":    round(s.fuel_remaining_kg, 1),
            "total_penalty":        round(s.total_penalty, 1),
            "penalties":            s.penalty_log,
            "path":                 s.path,
        }

# ---------------------------------------------------------------------------
# Q-LEARNING АГЕНТ
# ---------------------------------------------------------------------------
class QLearningAgent:
    """
    Табличный Q-learning агент.
    Состояние: (wp_idx, altitude_level, fuel_bucket, steps_at_level1)
    Действия: 0=снизиться, 1=держать эшелон, 2=набрать высоту
    """

    def __init__(self,
                 n_actions:   int   = 3,
                 alpha:       float = 0.1,    # скорость обучения
                 gamma:       float = 0.95,   # дисконт будущих наград
                 epsilon:     float = 1.0,    # начальная ε-жадность
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995):

        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q: Dict[Tuple, np.ndarray] = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, obs: Tuple) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[obs]))

    def update(self, obs: Tuple, action: int, reward: float,
               next_obs: Tuple, done: bool):
        q_curr = self.Q[obs][action]
        q_next = 0.0 if done else float(np.max(self.Q[next_obs]))
        self.Q[obs][action] = q_curr + self.alpha * (reward + self.gamma * q_next - q_curr)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q = defaultdict(lambda: np.zeros(self.n_actions), data)

# ---------------------------------------------------------------------------
# ЗАГРУЗКА ДАННЫХ
# ---------------------------------------------------------------------------
def _load_json_tolerant(path: str) -> list:
    """Загружает усечённый или полный JSON-массив."""
    raw = Path(path).read_text(encoding="utf-8").strip()
    raw = raw.rstrip(",").strip()
    if not raw.endswith("]"):
        raw += "]"
    return json.loads(raw)

def load_environment_data(base_dir: Path):
    """Загрузка всех необходимых данных для среды."""
    
    airports_path = base_dir / "airports" / "russian_civil_airports.json"
    prohibited_path = base_dir / "prohibited_zones" / "prohibited_zones.json"
    waypoints_path = base_dir / "prohibited_zones" / "allowed_to_use_waypoints.json"
    weather_path = base_dir / "weather" / "weather_data.json"
    simulation_path = base_dir / "plane_simulation" / "simulation_result.json"

    with open(airports_path, "r", encoding="utf-8") as f:
        airports = json.load(f)

    with open(prohibited_path, "r", encoding="utf-8") as f:
        prohibited = json.load(f)

    with open(waypoints_path, "r", encoding="utf-8") as f:
        wp_list = json.load(f)

    with open(weather_path, "r", encoding="utf-8") as f:
        weather = json.load(f)

    with open(simulation_path, "r", encoding="utf-8") as f:
        oncoming = json.load(f)

    allowed_wp = {w["name"]: w for w in wp_list}
    weather_db = WeatherDB(weather)
    return airports, prohibited, allowed_wp, weather_db, oncoming

def make_aircraft(sim_plane: Dict, cfg: Dict) -> Aircraft:
    return Aircraft(
        plane_id=sim_plane["plane_number"],
        aircraft_type=cfg["aircraft_type"],
        departure_name=sim_plane["departure"]["name"],
        arrival_name=sim_plane["arrival"]["name"],
        departure_lat=sim_plane["departure"]["lat"],
        departure_lon=sim_plane["departure"]["lon"],
        arrival_lat=sim_plane["arrival"]["lat"],
        arrival_lon=sim_plane["arrival"]["lon"],
        passengers=cfg.get("passengers", 150),
        baggage_kg=cfg.get("baggage_kg", 2000),
        min_reserve_fuel_kg=cfg.get("min_reserve_fuel_kg", 3000),
        altitude_level=sim_plane.get("altitude_level", 1),
    )

# ---------------------------------------------------------------------------
# ОБУЧЕНИЕ
# ---------------------------------------------------------------------------
def train(n_episodes: int = 2000,
          base_path: str = "",
          save_dir: str  = "") -> List[Dict]:
    """
    Обучает двух Q-learning агентов независимо.
    После обучения запускает финальный эпизод (greedy) и возвращает результаты.
    """
    base = Path(base_path) if base_path else Path(__file__).parent
    airports, prohibited, allowed_wp, weather_db, oncoming = load_environment_data(base)

    # Создаём объекты самолётов
    aircrafts = [
        make_aircraft(TRAINING_SIMULATION[i], AGENT_CONFIGS[i])
        for i in range(2)
    ]

    # Создаём агентов
    agents = [QLearningAgent(n_actions=FlightEnvironment.N_ACTIONS) for _ in range(2)]

    # Создаём среды
    def make_env(agent_idx: int) -> FlightEnvironment:
        # other_planes = второй агент + все 5 встречных из simulation_result.json
        other_agent = [TRAINING_SIMULATION[j] for j in range(2) if j != agent_idx]
        other_planes = other_agent + oncoming
        return FlightEnvironment(
            sim_data=TRAINING_SIMULATION[agent_idx],
            aircraft=make_aircraft(TRAINING_SIMULATION[agent_idx], AGENT_CONFIGS[agent_idx]),
            weather_db=weather_db,
            airports=airports,
            prohibited=prohibited,
            allowed_wp=allowed_wp,
            other_planes=other_planes,
        )

    print(f"\n{'='*60}")
    print(f"  ОБУЧЕНИЕ С ПОДКРЕПЛЕНИЕМ — Q-LEARNING")
    print(f"  Агентов: 2 | Эпизодов: {n_episodes}")
    print(f"{'='*60}")
    for i, ac in enumerate(aircrafts):
        print(f"  Агент {i+1}: {ac.preset['full_name']} | "
              f"{ac.passengers} пасс | {ac.baggage_kg:.0f} кг багажа | "
              f"топливо {ac.max_fuel_kg:.0f} кг")
    print(f"{'='*60}\n")

    # Журнал обучения
    reward_history = [[], []]

    for ep in range(1, n_episodes + 1):
        ep_rewards = []
        for agent_idx in range(2):
            env = make_env(agent_idx)
            obs = env.reset()
            agent = agents[agent_idx]
            ep_reward = 0.0
            done = False

            while not done:
                action = agent.select_action(obs)
                next_obs, reward, done = env.step(action)
                agent.update(obs, action, reward, next_obs, done)
                obs = next_obs
                ep_reward += reward

            agent.decay_epsilon()
            reward_history[agent_idx].append(ep_reward)
            ep_rewards.append(ep_reward)

        if ep % 200 == 0 or ep == 1:
            avg1 = np.mean(reward_history[0][-100:])
            avg2 = np.mean(reward_history[1][-100:])
            e1   = agents[0].epsilon
            e2   = agents[1].epsilon
            print(f"  Эп {ep:>5}/{n_episodes} | "
                  f"Агент1: avg_r={avg1:>10.0f} eps={e1:.3f} | "
                  f"Агент2: avg_r={avg2:>10.0f} eps={e2:.3f}")

    # Сохраняем Q-таблицы
    save = Path(save_dir)
    for i, agent in enumerate(agents):
        qpath = str(save / f"q_table_agent{i+1}.pkl")
        agent.save(qpath)
        print(f"\n  Q-таблица агента {i+1} сохранена: {qpath}"
              f" ({len(agent.Q)} состояний)")

    # -----------------------------------------------------------------------
    # ФИНАЛЬНЫЙ GREEDY ЭПИЗОД (без исследования)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ФИНАЛЬНЫЙ ЗАПУСК (greedy, ε=0)")
    print(f"{'='*60}")

    results = []
    for agent_idx in range(2):
        env = make_env(agent_idx)
        obs = env.reset()
        agents[agent_idx].epsilon = 0.0  # только эксплуатация
        done = False

        while not done:
            action = agents[agent_idx].select_action(obs)
            obs, _, done = env.step(action)

        res = env.get_result()
        results.append(res)

        print(f"\n  [ВС #{res['plane_id']}] "
              f"{res['departure']} → {res['arrival']}")
        print(f"  Тип: {res['aircraft_type']} | "
              f"Расстояние: {res['total_distance_km']:.0f} км")
        print(f"  Сожжено топлива: {res['total_fuel_burned_kg']:.0f} кг | "
              f"Остаток: {res['fuel_remaining_kg']:.0f} кг")
        print(f"  Суммарный штраф: {res['total_penalty']:.0f}")

        if res["penalties"]:
            print(f"  Штрафы ({len(res['penalties'])} событий):")
            for p in res["penalties"]:
                print(f"    t={p['t']:>3} | {p['reason']:55s} | {p['value']:>8.0f}")

        print(f"\n  Маршрут по эшелонам:")
        for step in res["path"]:
            print(f"    t={step['t']:>3} | {step['name']:20s} | "
                  f"эшелон {step['altitude_level']} | "
                  f"топл.сожж. {step['fuel_burned_step_kg']:>8.0f} кг | "
                  f"ветер ×{step['wind_effect']:.2f} | "
                  f"штраф {step['step_penalty']:>8.0f}")

    # Сохраняем результаты
    out_path = save / "two_planes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "training_episodes": n_episodes,
            "agents": results,
            "reward_history": {
                "agent1": reward_history[0],
                "agent2": reward_history[1],
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f" Результаты сохранены: {out_path}")
    print(f"{'='*60}\n")

    return results


# ---------------------------------------------------------------------------
# ТОЧКА ВХОДА
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train(n_episodes=2000)