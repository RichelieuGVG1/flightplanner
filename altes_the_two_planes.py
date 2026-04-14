"""
Агентная среда маршрутизации самолётов — Q-learning, два агента.

Входные файлы (кладутся рядом со скриптом):
  simulation_result.json        — встречные борты: plane_number, altitude_level,
                                  route_waypoints[{lat, lon, t}]
  russian_civil_airports.json   — запасные аэропорты
  prohibited_zones.json         — запретные зоны (полигоны, действуют на все эшелоны)
  allowed_to_use_waypoints.json — разрешённые waypoints
  weather_data.json             — погода: 5 эшелонов z=1..5, 100 шагов t=1..100

ДВИЖЕНИЕ:
  - t=1 … 100 — единая временная ось для погоды, встречных ВС и агентов.
  - Каждый шаг: агент выбирает одну из k=8 ближайших wp (в сторону цели,
    ещё не посещённых в этом эпизоде) И новый эшелон 1..5.
  - Запрещено возвращаться в уже посещённые wp — предотвращает зацикливание.
  - Запретные зоны блокируют все эшелоны.

ДИСТАНЦИИ БЕЗОПАСНОСТИ:
  - Между двумя агентами: минимум 200 км (AGENT_MIN_DISTANCE_KM).
  - До встречных ВС:      минимум 300 км (ONCOMING_MIN_DISTANCE_KM).

ACTION SPACE: 5 эшелонов × 8 wp = 40 действий
  action = altitude_idx * K_NEAREST_WAYPOINTS + wp_idx
  altitude_idx = 0..4 → эшелон 1..5
  wp_idx       = 0..7 → индекс в отсортированном списке кандидатов

OBSERVATION: (progress_bucket, altitude_level, fuel_bucket, steps_at_lvl1, conflict_bucket)
  progress_bucket — доля пройденного расстояния до цели × 10 (0..10)

REWARD:
  + REWARD_PROGRESS_PER_KM × прогресс_км  (главный сигнал, очень большой)
  + REWARD_ARRIVAL при прилёте            (перекрывает все штрафы маршрута)
  - расход топлива
  - штрафы (маленькие относительно прогресса — агент обходит опасности,
    только если это не сильно замедляет движение к цели)

Запуск:
  python flight_env.py          — обучение с нуля (n_episodes из TRAIN_CONFIG)
  python flight_env.py resume   — дообучение с сохранённых весов
  python flight_env.py inference — только полёт по загруженным весам
"""

from __future__ import annotations

import json
import math
import random
import pickle
import sys
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

# ===========================================================================
# РАЗДЕЛ 1: ШТРАФЫ И НАГРАДЫ
# ---------------------------------------------------------------------------
# Принцип балансировки:
#   REWARD_PROGRESS_PER_KM  — главный движущий сигнал; намеренно очень большой,
#                              чтобы агент всегда предпочитал двигаться к цели.
#   REWARD_ARRIVAL          — разовый бонус за прилёт; намного больше суммарных
#                              штрафов за весь маршрут, чтобы долететь всегда
#                              выгоднее, чем стоять на месте избегая штрафов.
#   Жёсткие запреты (зоны, не-wp) — большие, но не мешают прогрессу.
#   Погода, эшелоны, конфликты — маленькие; агент обходит их только если это
#                              не замедляет продвижение к цели.
# ===========================================================================

# ── Запреты (абсолютные) ──────────────────────────────────────────────────────
PENALTY_PROHIBITED_ZONE          = 500000 # пролёт через запретную зону (теперь физически невозможно)
PENALTY_NOT_WAYPOINT             = 0      # (убрано)
PENALTY_NOT_IN_K_NEAREST         = 0      # (убрано)
PENALTY_OUT_OF_FUEL              = 1000000 # огромный штраф за окончание топлива до цели
PENALTY_STEP                     = 500    # штраф за каждый сделанный шаг (чтобы не гулял)

# ── Эшелоны ──────────────────────────────────────────────────────────────────
PENALTY_ALTITUDE_CHANGE_1LVL     = 50     # смена эшелона на 1 уровень
PENALTY_ALTITUDE_CHANGE_2LVL     = 150    # смена эшелона на 2 уровня (за каждый)
PENALTY_ALTITUDE_CHANGE_3PLUS    = 300    # смена эшелона на 3+ уровня (за каждый)
PENALTY_LOW_ALT_PER_STEP         = 80     # долгое нахождение на эшелоне 1 (за шаг)
PENALTY_LOW_ALT_THRESHOLD        = 3      # шагов на эшелоне 1 до начала штрафа

# ── Погода ───────────────────────────────────────────────────────────────────
PENALTY_WEATHER_TURBULENCE_BASE  = 30     # турбулентность: × уровень turbulence
PENALTY_WEATHER_ICE_BASE         = 25     # обледенение: × уровень ice
PENALTY_WEATHER_STORM_BASE       = 60     # гроза: × storm_power

# ── Конфликты с чужими ВС ────────────────────────────────────────────────────
# Минимальная дистанция между агентами-самолётами — 200 км.
# Минимальная дистанция до встречных ВС — 300 км.
AGENT_MIN_DISTANCE_KM            = 200    # дистанция между двумя агентами
PENALTY_AGENT_CONFLICT           = 3000   # оба агента нарушают 200 км — штраф каждому
ONCOMING_MIN_DISTANCE_KM         = 300    # дистанция до встречных ВС
PENALTY_CONFLICT_LATERAL         = 800    # сближение < ONCOMING_MIN_DISTANCE_KM (одинак. эшелон)
PENALTY_CONFLICT_1LVL            = 400    # обход встречного через 1 эшелон
PENALTY_CONFLICT_2LVL            = 1000   # обход встречного через 2+ эшелона (за каждый)

# ── Топливо ──────────────────────────────────────────────────────────────────
PENALTY_LOW_FUEL                 = 200   # нехватка резервного топлива для запасного

# ── Отсутствие прогресса ─────────────────────────────────────────────────────
PENALTY_NO_PROGRESS              = 500    # штраф за шаг без приближения к цели

# ── Награды ──────────────────────────────────────────────────────────────────
REWARD_PROGRESS_PER_KM           = 10000  # было 5000
REWARD_ARRIVAL                   = 2000000 # было 200000

# ===========================================================================
# РАЗДЕЛ 2: ПАРАМЕТРЫ ОБУЧЕНИЯ И СРЕДЫ
# ===========================================================================
K_NEAREST_WAYPOINTS  = 8     # сколько ближайших wp предлагать агенту на каждом шаге
MAX_T                = 100   # максимум временных шагов на эпизод
ARRIVAL_RADIUS_KM    = 100   # км — считается что самолёт прилетел
FORWARD_CONE_DEG     = 130   # градусов от курса на цель — конус допустимых wp

TRAIN_CONFIG = {
    "n_episodes":    1000,   # эпизодов обучения
    "alpha":         0.15,   # скорость обучения Q-learning
    "gamma":         0.97,   # дисконт будущих наград
    "epsilon":       1.0,    # начальная ε для ε-greedy
    "epsilon_min":   0.05,
    "epsilon_decay": 0.997,  # множитель после каждого эпизода
    "log_every":     10,    # печатать прогресс каждые N эпизодов
}

# ===========================================================================
# РАЗДЕЛ 3: ТТХ САМОЛЁТОВ
# ===========================================================================
AIRCRAFT_PRESETS: Dict[str, Dict] = {

    # Airbus A350-900 | MTOW 280000 кг | OEW 142400 кг | топливо 127033 кг
    "A350": {
        "full_name":            "Airbus A350-900",
        "max_passengers":       440,
        "empty_mass_kg":        142400,
        "max_payload_kg":       53000,
        "max_fuel_kg":          127033,
        "fuel_per_km_empty_kg": 6.98,
        "fuel_climb_per_level": 480,
        "cruise_speed_kmh":     903,
        "min_runway_m":         2600,
        "runway_class":         "Long",
        "max_altitude_level":   5,
    },

    # Boeing 737-800 | MTOW 79016 кг | OEW 41413 кг | топливо 20810 кг
    "B738": {
        "full_name":            "Boeing 737-800",
        "max_passengers":       189,
        "empty_mass_kg":        41413,
        "max_payload_kg":       20800,
        "max_fuel_kg":          20810,
        "fuel_per_km_empty_kg": 2.9,
        "fuel_climb_per_level": 150,
        "cruise_speed_kmh":     842,
        "min_runway_m":         2090,
        "runway_class":         "Long",
        "max_altitude_level":   5,
    },

    # Sukhoi Superjet 100-95B | MTOW 49450 кг | OEW 29150 кг | топливо 11000 кг
    "SSJ100": {
        "full_name":            "Sukhoi Superjet 100-95B",
        "max_passengers":       98,
        "empty_mass_kg":        29150,
        "max_payload_kg":       12245,
        "max_fuel_kg":          11000,
        "fuel_per_km_empty_kg": 2.2,
        "fuel_climb_per_level": 100,
        "cruise_speed_kmh":     787,
        "min_runway_m":         1731,
        "runway_class":         "Long",
        "max_altitude_level":   5,
    },

    # Airbus A321-200 | MTOW 93500 кг | OEW 48500 кг | топливо 21385 кг
    "A321": {
        "full_name":            "Airbus A321-200",
        "max_passengers":       220,
        "empty_mass_kg":        48500,
        "max_payload_kg":       23400,
        "max_fuel_kg":          21385,
        "fuel_per_km_empty_kg": 3.6,
        "fuel_climb_per_level": 160,
        "cruise_speed_kmh":     833,
        "min_runway_m":         2200,
        "runway_class":         "Long",
        "max_altitude_level":   5,
    },

    # Boeing 777-300ER | MTOW 351500 кг | OEW 167829 кг | топливо 145027 кг
    "B773ER": {
        "full_name":            "Boeing 777-300ER",
        "max_passengers":       396,
        "empty_mass_kg":        167829,
        "max_payload_kg":       69600,
        "max_fuel_kg":          145027,
        "fuel_per_km_empty_kg": 9.4,
        "fuel_climb_per_level": 600,
        "cruise_speed_kmh":     905,
        "min_runway_m":         3050,
        "runway_class":         "Long",
        "max_altitude_level":   5,
    },

    # ATR 72-600 | MTOW 23000 кг | OEW 13010 кг | топливо 6370 кг
    "ATR72": {
        "full_name":            "ATR 72-600",
        "max_passengers":       70,
        "empty_mass_kg":        13010,
        "max_payload_kg":       7500,
        "max_fuel_kg":          6370,
        "fuel_per_km_empty_kg": 0.85,
        "fuel_climb_per_level": 50,
        "cruise_speed_kmh":     510,
        "min_runway_m":         1200,
        "runway_class":         "Short",
        "max_altitude_level":   5,
    },
}

AIRCRAFT_TYPE_ALIAS: Dict[str, str] = {
    "Airbus A350-900":          "A350",
    "Airbus A320-200":          "A350",
    "Boeing 737-800":           "B738",
    "Sukhoi Superjet 100-95B":  "SSJ100",
    "Airbus A321-200":          "A321",
    "Boeing 777-300ER":         "B773ER",
    "ATR 72-600":               "ATR72",
    **{k: k for k in ["A350", "B738", "SSJ100", "A321", "B773ER", "ATR72"]},
}

# ===========================================================================
# РАЗДЕЛ 4: КОНФИГУРАЦИИ АГЕНТОВ И МАРШРУТЫ
# (все настройки двух самолётов-агентов собраны здесь)
# ===========================================================================

# ── Маршруты агентов ────────────────────────────────────────────────────────
# Москва (SVO) → Владивосток (VVO) ~6430 км
# Санкт-Петербург (LED) → Петропавловск-Камчатский (PKC) ~7720 км
AGENT_ROUTES: List[Dict] = [
    {
        "plane_number":    1,
        "departure":       {"name": "Moskva_SVO",      "lat": 55.9726, "lon":  37.4146},
        "arrival":         {"name": "Vladivostok_VVO", "lat": 43.3989, "lon": 132.1478},
        "gc_distance_km":  6430.0,
        "corridor_km":     1000,
        "start_t":         1,
        "start_altitude":  3,   # начальный эшелон
    },
    {
        "plane_number":    2,
        "departure":       {"name": "SPb_LED", "lat": 59.8003, "lon":  30.2625},
        "arrival":         {"name": "PKC",     "lat": 53.1679, "lon": 158.4539},
        "gc_distance_km":  7720.0,
        "corridor_km":     1000,
        "start_t":         1,
        "start_altitude":  3,
    },
]

# ── Параметры воздушных судов агентов ───────────────────────────────────────
# Топливо: ВСЕГДА полный бак (max_fuel_kg определяется из ТТХ с учётом загрузки).
AGENT_AIRCRAFT: List[Dict] = [
    {
        "aircraft_type":       "B773ER",   # Boeing 777-300ER
        "passengers":          317,        # ~80% от 396 мест
        "baggage_kg":          7291,       # 317 пасс × 23 кг
        "min_reserve_fuel_kg": 5000,       # минимальный резерв для ухода на запасной
        # fuel_kg не задаётся — всегда грузится полный бак (Aircraft.max_fuel_kg)
    },
    {
        "aircraft_type":       "A350",     # Airbus A350-900
        "passengers":          352,        # ~80% от 440 мест
        "baggage_kg":          8096,       # 352 пасс × 23 кг
        "min_reserve_fuel_kg": 5000,
    },
]

# Встречные борты — загружаются из simulation_result.json; здесь только структура.
# Формат каждой записи:
#   { "plane_number": int, "altitude_level": int,
#     "route_waypoints": [{"lat":…, "lon":…, "t":…}, …] }


# ===========================================================================
# ГЕОДЕЗИЯ
# ===========================================================================
EARTH_R = 6371.0

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(max(0.0, min(1.0, a))))

def azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return math.degrees(math.atan2(x, y)) % 360

def angle_diff(a: float, b: float) -> float:
    """Минимальная разница азимутов: −180 … +180°."""
    d = (a - b + 360) % 360
    return d - 360 if d > 180 else d


# ===========================================================================
# ГЕОМЕТРИЯ ПОЛИГОНОВ
# ===========================================================================
def point_in_polygon(lat: float, lon: float, polygon: List[Dict]) -> bool:
    """Ray-casting. Запретные зоны действуют на все эшелоны."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]["lon"], polygon[i]["lat"]
        xj, yj = polygon[j]["lon"], polygon[j]["lat"]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside

def segment_crosses_polygon(lat1: float, lon1: float,
                             lat2: float, lon2: float,
                             polygon: List[Dict],
                             samples: int = 20) -> bool:
    for i in range(samples + 1):
        t = i / samples
        if point_in_polygon(lat1 + t * (lat2 - lat1),
                             lon1 + t * (lon2 - lon1), polygon):
            return True
    return False

def wp_in_any_prohibited(wp: Dict, prohibited: List[Dict]) -> bool:
    for zone in prohibited:
        if point_in_polygon(wp["lat"], wp["lon"], zone["points"]):
            return True
    return False


# ===========================================================================
# ПОГОДНАЯ БД
# ===========================================================================
class WeatherDB:
    """Хранит погоду по (name, z, t) и ищет ближайший узел по (lat, lon, z, t)."""

    def __init__(self, records: List[Dict]):
        self._data: Dict[Tuple, Dict] = {}
        self._wp_coords: Dict[str, Tuple[float, float]] = {}
        for r in records:
            self._data[(r["name"], r["z"], r["t"])] = r
            self._wp_coords[r["name"]] = (r["lat"], r["lon"])

    def get(self, lat: float, lon: float, z: int, t: int) -> Optional[Dict]:
        """Ближайший погодный узел для эшелона z и шага t."""
        t_clamped = max(1, min(MAX_T, t))
        best_dist, best = float("inf"), None
        for (name, rz, rt), rec in self._data.items():
            if rz != z or rt != t_clamped:
                continue
            wlat, wlon = self._wp_coords[name]
            d = haversine(lat, lon, wlat, wlon)
            if d < best_dist:
                best_dist, best = d, rec
        return best

    def wind_effect(self, aircraft_az: float,
                    wind_dir: float, wind_speed: float) -> float:
        """
        Коэффициент расхода топлива от ветра (0.5 … 2.0).
        Встречный (azimuth ≈ wind_dir + 180°) → >1.0.
        Попутный (azimuth ≈ wind_dir)          → <1.0.
        """
        diff    = angle_diff(aircraft_az, (wind_dir + 180) % 360)
        cos_val = math.cos(math.radians(diff))
        return max(0.5, min(2.0, 1.0 - cos_val * wind_speed * 0.005))


# ===========================================================================
# МОДЕЛЬ САМОЛЁТА
# ===========================================================================
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
    baggage_kg:          float = 2000
    min_reserve_fuel_kg: float = 5000
    start_altitude:      int   = 3
    preset: Dict = field(default_factory=dict)

    def __post_init__(self):
        key = AIRCRAFT_TYPE_ALIAS.get(self.aircraft_type, self.aircraft_type)
        if key not in AIRCRAFT_PRESETS:
            raise ValueError(f"Неизвестный тип ВС: '{self.aircraft_type}'. "
                             f"Доступные: {list(AIRCRAFT_PRESETS)}")
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
        """Полный бак с учётом загрузки."""
        slack = self.preset["max_payload_kg"] - self.total_payload_kg
        return min(self.preset["max_fuel_kg"],
                   max(0.0, self.preset["max_fuel_kg"] + slack))

    @property
    def total_mass_kg(self) -> float:
        return self.preset["empty_mass_kg"] + self.total_payload_kg + self.max_fuel_kg

    @property
    def mass_ratio(self) -> float:
        return self.total_mass_kg / self.preset["empty_mass_kg"]

    def fuel_per_km(self, wind_effect: float = 1.0) -> float:
        """Расход топлива кг/км с учётом загрузки и ветра."""
        return self.preset["fuel_per_km_empty_kg"] * self.mass_ratio * wind_effect

    def fuel_for_level_change(self, levels: int) -> float:
        """Расход на смену |levels| эшелонов. 2+ уровня — множитель растёт."""
        n = abs(levels)
        if n == 0:
            return 0.0
        base = self.preset["fuel_climb_per_level"] * self.mass_ratio
        if n == 1:
            return base
        elif n == 2:
            return base * n * 1.5
        else:
            return base * n * 2.0

    def can_land(self, airport: Dict) -> bool:
        if self.preset["runway_class"] == "Short":
            return True
        return airport.get("max_runway_m", 0) >= self.preset["min_runway_m"]


# ===========================================================================
# СОСТОЯНИЕ ПОЛЁТА
# ===========================================================================
@dataclass
class FlightState:
    plane_id:          int
    current_lat:       float
    current_lon:       float
    current_t:         int
    altitude_level:    int
    fuel_remaining_kg: float
    dist_to_goal_km:   float          # расстояние до цели — для расчёта прогресса
    total_penalty:     float = 0.0
    steps_at_level1:   int   = 0
    visited_wp:        set   = field(default_factory=set)   # имена посещённых wp
    penalty_log:       List[Dict] = field(default_factory=list)
    path:              List[Dict] = field(default_factory=list)

    def add_penalty(self, reason: str, value: float):
        self.total_penalty += value
        self.penalty_log.append({
            "reason": reason,
            "value":  round(value, 1),
            "t":      self.current_t,
            "lat":    round(self.current_lat, 5),
            "lon":    round(self.current_lon, 5),
        })


# ===========================================================================
# СРЕДА
# ===========================================================================
class FlightEnvironment:
    """
    Среда Q-learning для одного самолёта-агента.

    Ключевые отличия от предыдущей версии:
      1. Нет зацикливания — посещённые wp исключаются из кандидатов.
      2. Есть reward за прогресс — агент получает + за каждый км к цели.
      3. Большой бонус за прилёт — самолёт мотивирован долетать.
      4. Наблюдение включает progress_bucket вместо t_bucket.
    """

    N_ALTITUDES = 5
    N_ACTIONS   = N_ALTITUDES * K_NEAREST_WAYPOINTS   # 5 × 8 = 40

    def __init__(self,
                 route:        Dict,
                 aircraft:     Aircraft,
                 weather_db:   WeatherDB,
                 airports:     List[Dict],
                 prohibited:   List[Dict],
                 allowed_wp:   Dict[str, Dict],
                 other_planes: List[Dict]):

        self.route        = route
        self.aircraft     = aircraft
        self.weather_db   = weather_db
        self.airports     = airports
        self.prohibited   = prohibited
        self.other_planes = other_planes

        # Разрешённые wp вне запретных зон
        self.allowed_wp: Dict[str, Dict] = {
            name: wp for name, wp in allowed_wp.items()
            if not wp_in_any_prohibited(wp, prohibited)
        }
        self._wp_list: List[Dict] = list(self.allowed_wp.values())

        self.arrival_lat  = route["arrival"]["lat"]
        self.arrival_lon  = route["arrival"]["lon"]
        self.arrival_name = route["arrival"]["name"]
        self.gc_dist      = route["gc_distance_km"]

        self._reset_state()

    # ──────────────────────────────────────────────────────────────────
    # GYM-интерфейс
    # ──────────────────────────────────────────────────────────────────
    def reset(self) -> Tuple:
        self._reset_state()
        return self._obs()

    def step(self, action: int) -> Tuple[Tuple, float, bool]:
        """
        Декодируем action → (new_alt, выбор wp из кандидатов).
        Возвращает (obs, reward, done).
        """
        altitude_idx = action // K_NEAREST_WAYPOINTS   # 0..4
        wp_idx       = action % K_NEAREST_WAYPOINTS    # 0..5
        new_alt      = altitude_idx + 1                # эшелон 1..5

        candidates = self._get_candidates(
            self.state.current_lat,
            self.state.current_lon,
            self.state.visited_wp,
        )

        if not candidates:
            # Все соседние wp посещены или заблокированы — эпизод завершён
            reward = -(PENALTY_NO_PROGRESS * 5)
            return self._obs(), reward, True

        wp_idx  = min(wp_idx, len(candidates) - 1)
        next_wp = candidates[wp_idx]

        reward, done = self._compute_step(next_wp, new_alt, candidates)

        # Достижение аэропорта прилёта
        dist_arr = haversine(self.state.current_lat, self.state.current_lon,
                             self.arrival_lat, self.arrival_lon)
        if dist_arr < ARRIVAL_RADIUS_KM:
            reward += REWARD_ARRIVAL
            done    = True

        if self.state.current_t >= MAX_T:
            done = True

        return self._obs(), reward, done

    # ──────────────────────────────────────────────────────────────────
    # Внутренние методы
    # ──────────────────────────────────────────────────────────────────
    def _reset_state(self):
        dep = self.route["departure"]
        d0  = haversine(dep["lat"], dep["lon"], self.arrival_lat, self.arrival_lon)
        self.state = FlightState(
            plane_id=self.aircraft.plane_id,
            current_lat=dep["lat"],
            current_lon=dep["lon"],
            current_t=self.route.get("start_t", 1),
            altitude_level=self.route.get("start_altitude", 3),
            fuel_remaining_kg=self.aircraft.max_fuel_kg,   # ПОЛНЫЙ БАК
            dist_to_goal_km=d0,
            visited_wp={dep["name"]},
            path=[{
                "name":                dep["name"],
                "lat":                 dep["lat"],
                "lon":                 dep["lon"],
                "t":                   self.route.get("start_t", 1),
                "altitude_level":      self.route.get("start_altitude", 3),
                "fuel_remaining_kg":   round(self.aircraft.max_fuel_kg, 1),
                "step_penalty":        0.0,
                "dist_km":             0.0,
                "fuel_burned_step_kg": 0.0,
                "wind_effect":         1.0,
                "progress_km":         0.0,
            }]
        )

    def _obs(self) -> Tuple:
        """
        (progress_bucket, altitude_level, fuel_bucket, steps_at_lvl1, conflict_bucket)

        progress_bucket  — доля пройденного пути к цели × 10 → 0..10
        fuel_bucket      — остаток топлива // 10000 → 0..20
        conflict_bucket  — дистанция до ближайшего встречного // 200 → 0..10
        """
        progress = 1.0 - self.state.dist_to_goal_km / max(1.0, self.gc_dist)
        progress_bucket = max(0, min(10, int(progress * 10)))
        fuel_bucket     = max(0, min(20, int(self.state.fuel_remaining_kg // 10000)))

        min_d = float("inf")
        for other in self.other_planes:
            pos = self._interp_other_pos(other, self.state.current_t)
            if pos:
                d = haversine(self.state.current_lat, self.state.current_lon,
                              pos[0], pos[1])
                if d < min_d:
                    min_d = d
        # conflict_bucket: 0 = ближе 200 км (опасно), 10 = дальше 2000 км (безопасно)
        conflict_bucket = (max(0, min(10, int(min_d // 200)))
                           if min_d < float("inf") else 10)

        return (
            progress_bucket,
            self.state.altitude_level,
            fuel_bucket,
            min(self.state.steps_at_level1, 10),
            conflict_bucket,
        )

    def _get_candidates(self, lat: float, lon: float,
                        visited: set) -> List[Dict]:
        """
        k=6 ближайших разрешённых wp, удовлетворяющих условиям:
          - не посещались в этом эпизоде (нет зацикливания)
          - находятся в конусе ±FORWARD_CONE_DEG° от курса на цель
          - расстояние > 10 км (не "уже там")

        Если ни один wp не проходит фильтр — возвращаем k=6 ближайших
        непосещённых без ограничения конуса (fallback).
        """
        if not self._wp_list:
            return []

        goal_az = azimuth(lat, lon, self.arrival_lat, self.arrival_lon)

        forward: List[Tuple[float, Dict]] = []
        any_unvisited: List[Tuple[float, Dict]] = []

        for wp in self._wp_list:
            if wp.get("name", "") in visited:
                continue
            d = haversine(lat, lon, wp["lat"], wp["lon"])
            if d < 10:
                continue
            any_unvisited.append((d, wp))
            wp_az = azimuth(lat, lon, wp["lat"], wp["lon"])
            if abs(angle_diff(wp_az, goal_az)) <= FORWARD_CONE_DEG:
                forward.append((d, wp))

        # Добавляем аэропорт прилёта, если он близко и не посещён
        arr_name = self.arrival_name
        arr_d    = haversine(lat, lon, self.arrival_lat, self.arrival_lon)
        if arr_name not in visited and arr_d < 2000:
            arr_wp = {"name": arr_name, "lat": self.arrival_lat, "lon": self.arrival_lon}
            forward.append((arr_d, arr_wp))
            any_unvisited.append((arr_d, arr_wp))

        pool = forward if forward else any_unvisited
        
        # СТРОГАЯ ПРОВЕРКА ЗАПРЕТНЫХ ЗОН: исключаем точки, путь к которым пересекает зону
        filtered_pool = []
        for d, wp in pool:
            crosses = False
            for zone in self.prohibited:
                if segment_crosses_polygon(lat, lon, wp["lat"], wp["lon"], zone["points"]):
                    crosses = True
                    break
            if not crosses:
                filtered_pool.append((d, wp))
        
        # СОРТИРОВКА: выбираем те точки, которые максимально ПРИБЛИЖАЮТ к цели (progress)
        # а не просто ближайшие к текущей позиции
        def progress_score(item):
            d_curr_to_goal = haversine(lat, lon, self.arrival_lat, self.arrival_lon)
            d_next_to_goal = haversine(item[1]["lat"], item[1]["lon"], self.arrival_lat, self.arrival_lon)
            return d_next_to_goal # чем меньше, тем лучше

        filtered_pool.sort(key=progress_score)
        return [wp for _, wp in filtered_pool[:K_NEAREST_WAYPOINTS]]

    def _compute_step(self, next_wp: Dict, new_alt: int,
                      candidates: List[Dict]) -> Tuple[float, bool]:
        """Полный расчёт одного шага."""
        s         = self.state
        s.current_t += 1
        penalty   = 0.0

        next_lat  = next_wp["lat"]
        next_lon  = next_wp["lon"]
        next_name = next_wp.get("name", "")

        # ── 1. Смена эшелона ─────────────────────────────────────────
        level_diff = abs(new_alt - s.altitude_level)
        fuel_climb = self.aircraft.fuel_for_level_change(level_diff)
        s.fuel_remaining_kg -= fuel_climb
        s.altitude_level     = new_alt

        if level_diff == 1:
            penalty += PENALTY_ALTITUDE_CHANGE_1LVL
            s.add_penalty("Altitude change 1 level", PENALTY_ALTITUDE_CHANGE_1LVL)
        elif level_diff == 2:
            p = PENALTY_ALTITUDE_CHANGE_2LVL * level_diff
            penalty += p
            s.add_penalty(f"Altitude change {level_diff} levels", p)
        elif level_diff >= 3:
            p = PENALTY_ALTITUDE_CHANGE_3PLUS * level_diff
            penalty += p
            s.add_penalty(f"Altitude change {level_diff} levels", p)

        # ── 2. Долгое нахождение на эшелоне 1 ────────────────────────
        if new_alt == 1:
            s.steps_at_level1 += 1
            if s.steps_at_level1 > PENALTY_LOW_ALT_THRESHOLD:
                extra = s.steps_at_level1 - PENALTY_LOW_ALT_THRESHOLD
                p = PENALTY_LOW_ALT_PER_STEP * (1 + extra * 0.5)
                penalty += p
                s.add_penalty(f"Long stay at level 1 (step #{s.steps_at_level1})", p)
        else:
            s.steps_at_level1 = 0

        # ── 3. Проверка wp из разрешённого списка / k ближайших ───────
        is_arrival       = (next_name == self.arrival_name)
        candidate_names  = {wp.get("name", "") for wp in candidates}
        if not is_arrival:
            if next_name not in self.allowed_wp:
                penalty += PENALTY_NOT_WAYPOINT
                s.add_penalty(f"WP '{next_name}' not in allowed list",
                              PENALTY_NOT_WAYPOINT)
            elif next_name not in candidate_names:
                penalty += PENALTY_NOT_IN_K_NEAREST
                s.add_penalty(f"WP '{next_name}' not in {K_NEAREST_WAYPOINTS} nearest",
                              PENALTY_NOT_IN_K_NEAREST)

        # ── 4. Запретные зоны (все эшелоны) ──────────────────────────
        for zone in self.prohibited:
            if segment_crosses_polygon(s.current_lat, s.current_lon,
                                       next_lat, next_lon, zone["points"]):
                penalty += PENALTY_PROHIBITED_ZONE
                s.add_penalty(f"Prohibited zone #{zone['id']} (lvl.{new_alt})",
                              PENALTY_PROHIBITED_ZONE)

        # ── 5. Погода (текущий эшелон, текущий t) ────────────────────
        weather  = self.weather_db.get(s.current_lat, s.current_lon, new_alt, s.current_t)
        wind_eff = 1.0
        if weather:
            az       = azimuth(s.current_lat, s.current_lon, next_lat, next_lon)
            wind_eff = self.weather_db.wind_effect(
                az, weather["wind_dir"], weather["wind_speed"])

            turb = weather.get("turbulence", 0)
            if turb > 0:
                p = PENALTY_WEATHER_TURBULENCE_BASE * turb
                penalty += p
                s.add_penalty(f"Turbulence {turb} (lvl.{new_alt}, t={s.current_t})", p)

            ice = weather.get("ice", 0)
            if ice > 0:
                p = PENALTY_WEATHER_ICE_BASE * ice
                penalty += p
                s.add_penalty(f"Icing {ice} (lvl.{new_alt}, t={s.current_t})", p)

            storm = weather.get("storm_power", 0.0)
            if storm > 0:
                p = PENALTY_WEATHER_STORM_BASE * storm
                penalty += p
                s.add_penalty(f"Storm {storm:.1f} (lvl.{new_alt}, t={s.current_t})", p)

        # ── 6. Расход топлива на перелёт ──────────────────────────────
        dist_km     = haversine(s.current_lat, s.current_lon, next_lat, next_lon)
        fuel_flight = dist_km * self.aircraft.fuel_per_km(wind_eff)
        s.fuel_remaining_kg -= fuel_flight

        # ── 7а. Дистанция между двумя агентами (≥ AGENT_MIN_DISTANCE_KM = 200 км) ──
        # other_planes содержит маршруты второго агента как элементы AGENT_ROUTES,
        # их отличаем по наличию поля "plane_number" и отсутствию "altitude_level"
        # как фиксированного эшелона (у агентов эшелон меняется — проверяем только дистанцию).
        for other in self.other_planes:
            num = other.get("plane_number", "?")
            if num == s.plane_id:
                continue
            # Второй агент — у него нет фиксированного altitude_level в корне словаря,
            # но есть departure/arrival; отличаем по "departure" key
            if "departure" not in other:
                continue  # это встречный ВС, обработаем ниже
            pos = self._interp_other_pos(other, s.current_t)
            if pos is None:
                continue
            other_lat, other_lon, _ = pos
            d = haversine(next_lat, next_lon, other_lat, other_lon)
            if d < AGENT_MIN_DISTANCE_KM:
                penalty += PENALTY_AGENT_CONFLICT
                s.add_penalty(
                    f"Distance to agent #{num} = {d:.0f} km < {AGENT_MIN_DISTANCE_KM} km",
                    PENALTY_AGENT_CONFLICT)

        # ── 7б. Конфликты со встречными ВС (≥ ONCOMING_MIN_DISTANCE_KM = 300 км) ─
        for other in self.other_planes:
            if other.get("plane_number") == s.plane_id:
                continue
            if "departure" in other:
                continue  # это второй агент, уже обработан выше
            pos = self._interp_other_pos(other, s.current_t)
            if pos is None:
                continue
            other_lat, other_lon, other_alt = pos
            d = haversine(next_lat, next_lon, other_lat, other_lon)
            if d >= ONCOMING_MIN_DISTANCE_KM:
                continue
            alt_diff = abs(new_alt - other_alt)
            num      = other.get("plane_number", "?")
            if alt_diff == 0:
                penalty += PENALTY_CONFLICT_LATERAL
                s.add_penalty(
                    f"Oncoming conflict with AC #{num} (d={d:.0f} km, lvl.{new_alt})",
                    PENALTY_CONFLICT_LATERAL)
            elif alt_diff == 1:
                penalty += PENALTY_CONFLICT_1LVL
                s.add_penalty(f"Conflict bypass AC #{num} via 1 level (d={d:.0f} km)",
                              PENALTY_CONFLICT_1LVL)
            else:
                p = PENALTY_CONFLICT_2LVL * alt_diff
                penalty += p
                s.add_penalty(f"Conflict bypass AC #{num} via {alt_diff} levels (d={d:.0f} km)", p)

        # ── 8. Резервное топливо на запасной аэродром ─────────────────
        _, dist_ap = self._nearest_alternate(s)
        if dist_ap < float("inf"):
            fuel_needed = (dist_ap * self.aircraft.fuel_per_km() * 1.1
                           + self.aircraft.fuel_for_level_change(max(0, new_alt - 1))
                           + self.aircraft.min_reserve_fuel_kg)
            if s.fuel_remaining_kg < fuel_needed:
                shortage = fuel_needed - s.fuel_remaining_kg
                penalty += PENALTY_LOW_FUEL
                s.add_penalty(f"Мало топлива для запасного (не хватает {shortage:.0f} кг)",
                              PENALTY_LOW_FUEL)

        # ── 9. Штраф за отсутствие прогресса ─────────────────────────
        new_dist_to_goal = haversine(next_lat, next_lon, self.arrival_lat, self.arrival_lon)
        progress_km      = s.dist_to_goal_km - new_dist_to_goal  # + если приближаемся

        if progress_km <= 0:
            penalty += PENALTY_NO_PROGRESS
            s.add_penalty("Нет прогресса к цели на этом шаге", PENALTY_NO_PROGRESS)

        # ── 10. Обновляем состояние ───────────────────────────────────
        s.total_penalty  += penalty
        s.current_lat     = next_lat
        s.current_lon     = next_lon
        s.dist_to_goal_km = new_dist_to_goal
        s.visited_wp.add(next_name)

        s.path.append({
            "name":                next_name,
            "lat":                 round(next_lat, 6),
            "lon":                 round(next_lon, 6),
            "t":                   s.current_t,
            "altitude_level":      new_alt,
            "fuel_remaining_kg":   round(s.fuel_remaining_kg, 1),
            "step_penalty":        round(penalty, 1),
            "dist_km":             round(dist_km, 2),
            "fuel_burned_step_kg": round(fuel_flight + fuel_climb, 1),
            "wind_effect":         round(wind_eff, 3),
            "progress_km":         round(progress_km, 1),
        })

        # reward = прогресс к цели − топливо − штрафы − штраф за шаг
        reward = (REWARD_PROGRESS_PER_KM * max(0.0, progress_km)
                  - fuel_flight - fuel_climb - penalty - PENALTY_STEP)
        
        done = False
        
        # Достижение аэропорта прилёта (проверяем ДО топлива)
        if haversine(s.current_lat, s.current_lon, self.arrival_lat, self.arrival_lon) < ARRIVAL_RADIUS_KM:
            reward += REWARD_ARRIVAL
            done = True
        
        if s.fuel_remaining_kg <= 0:
            if not done: # если не прилетел, но топливо кончилось
                reward -= PENALTY_OUT_OF_FUEL
                s.add_penalty("ТОПЛИВО ЗАКОНЧИЛОСЬ", PENALTY_OUT_OF_FUEL)
            done = True

        return reward, done

    # ──────────────────────────────────────────────────────────────────
    # Вспомогательные методы
    # ──────────────────────────────────────────────────────────────────
    def _interp_other_pos(self, other: Dict,
                           t: int) -> Optional[Tuple[float, float, int]]:
        """(lat, lon, alt) встречного борта на шаге t. None если не летит."""
        wps = other.get("route_waypoints", [])
        if not wps:
            return None
        other_alt = int(other.get("altitude_level", 1))
        t_start, t_end = wps[0]["t"], wps[-1]["t"]
        if t < t_start or t > t_end:
            return None
        for wp in wps:
            if wp["t"] == t:
                return wp["lat"], wp["lon"], other_alt
        for k in range(len(wps) - 1):
            a, b = wps[k], wps[k + 1]
            if a["t"] < t < b["t"]:
                frac = (t - a["t"]) / max(1, b["t"] - a["t"])
                lat  = a["lat"] + frac * (b["lat"] - a["lat"])
                lon  = a["lon"] + frac * (b["lon"] - a["lon"])
                return lat, lon, other_alt
        return None

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
        s          = self.state
        total_dist = sum(p["dist_km"]             for p in s.path)
        total_fuel = sum(p["fuel_burned_step_kg"] for p in s.path)
        arrived    = s.dist_to_goal_km < ARRIVAL_RADIUS_KM
        return {
            "plane_id":             s.plane_id,
            "aircraft_type":        self.aircraft.aircraft_type,
            "departure":            self.aircraft.departure_name,
            "arrival":              self.aircraft.arrival_name,
            "arrived":              arrived,
            "total_distance_km":    round(total_dist, 2),
            "total_fuel_burned_kg": round(total_fuel, 1),
            "fuel_remaining_kg":    round(s.fuel_remaining_kg, 1),
            "total_penalty":        round(s.total_penalty, 1),
            "steps_taken":          s.current_t - self.route.get("start_t", 1),
            "dist_to_goal_final_km": round(s.dist_to_goal_km, 1),
            "penalties":            s.penalty_log,
            "path":                 s.path,
        }


# ===========================================================================
# Q-LEARNING АГЕНТ
# ===========================================================================
class QLearningAgent:
    """
    Табличный Q-learning агент.

    Observation: (progress_bucket, altitude, fuel_bucket, steps_at_lvl1, conflict_bucket)
    Actions: 40 — (5 эшелонов) × (8 ближайших wp)
    Веса хранятся в .pkl, при загрузке ε = 0 (режим inference).
    """

    def __init__(self,
                 n_actions:     int   = FlightEnvironment.N_ACTIONS,
                 alpha:         float = TRAIN_CONFIG["alpha"],
                 gamma:         float = TRAIN_CONFIG["gamma"],
                 epsilon:       float = TRAIN_CONFIG["epsilon"],
                 epsilon_min:   float = TRAIN_CONFIG["epsilon_min"],
                 epsilon_decay: float = TRAIN_CONFIG["epsilon_decay"]):

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
        self.Q[obs][action] += self.alpha * (reward + self.gamma * q_next - q_curr)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "Q":             dict(self.Q),
                "n_actions":     self.n_actions,
                "alpha":         self.alpha,
                "gamma":         self.gamma,
                "epsilon":       self.epsilon,
                "epsilon_min":   self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            }, f)
        print(f"    Веса сохранены: {path} ({len(self.Q)} состояний)")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        n = data.get("n_actions", self.n_actions)
        self.Q             = defaultdict(lambda: np.zeros(n), data["Q"])
        self.n_actions     = n
        self.alpha         = data.get("alpha",         self.alpha)
        self.gamma         = data.get("gamma",         self.gamma)
        self.epsilon       = data.get("epsilon",       self.epsilon)
        self.epsilon_min   = data.get("epsilon_min",   self.epsilon_min)
        self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)

    @classmethod
    def from_file(cls, path: str) -> "QLearningAgent":
        agent = cls()
        agent.load(path)
        agent.epsilon = 0.0
        return agent


# ===========================================================================
# ЗАГРУЗКА ДАННЫХ
# ===========================================================================
def load_environment_data():
    with open("russian_civil_airports.json", "r", encoding="utf-8") as f:
        airports = json.load(f)
    with open("prohibited_zones.json", "r", encoding="utf-8") as f:
        prohibited = json.load(f)
    with open("allowed_to_use_waypoints.json", "r", encoding="utf-8") as f:
        wp_list = json.load(f)
    with open("weather_data.json", "r", encoding="utf-8") as f:
        weather = json.load(f)
    with open("simulation_result.json", "r", encoding="utf-8") as f:
        oncoming = json.load(f)

    allowed_wp = {w["name"]: w for w in wp_list}
    weather_db = WeatherDB(weather)
    return airports, prohibited, allowed_wp, weather_db, oncoming


def _make_aircraft(agent_idx: int) -> Aircraft:
    """Создаёт Aircraft из AGENT_ROUTES + AGENT_AIRCRAFT."""
    route  = AGENT_ROUTES[agent_idx]
    cfg    = AGENT_AIRCRAFT[agent_idx]
    return Aircraft(
        plane_id             = route["plane_number"],
        aircraft_type        = cfg["aircraft_type"],
        departure_name       = route["departure"]["name"],
        arrival_name         = route["arrival"]["name"],
        departure_lat        = route["departure"]["lat"],
        departure_lon        = route["departure"]["lon"],
        arrival_lat          = route["arrival"]["lat"],
        arrival_lon          = route["arrival"]["lon"],
        passengers           = cfg.get("passengers", 150),
        baggage_kg           = cfg.get("baggage_kg", 2000),
        min_reserve_fuel_kg  = cfg.get("min_reserve_fuel_kg", 5000),
        start_altitude       = route.get("start_altitude", 3),
    )


def _make_env(agent_idx: int,
              airports, prohibited, allowed_wp,
              weather_db, oncoming) -> FlightEnvironment:
    """Создаёт среду. other_planes = второй агент-маршрут + встречные из simulation_result."""
    other_routes = [AGENT_ROUTES[j] for j in range(len(AGENT_ROUTES)) if j != agent_idx]
    other_planes = other_routes + oncoming
    return FlightEnvironment(
        route        = AGENT_ROUTES[agent_idx],
        aircraft     = _make_aircraft(agent_idx),
        weather_db   = weather_db,
        airports     = airports,
        prohibited   = prohibited,
        allowed_wp   = allowed_wp,
        other_planes = other_planes,
    )


# ===========================================================================
# ОБУЧЕНИЕ
# ===========================================================================
def train(n_episodes: int  = TRAIN_CONFIG["n_episodes"],
          save_dir:   str  = "",
          resume:     bool = False) -> List[Dict]:
    """
    Обучает двух Q-learning агентов независимо.
    Сохраняет q_table_agent1.pkl / q_table_agent2.pkl и two_planes.json.
    """
    save = Path(save_dir)
    save.mkdir(parents=True, exist_ok=True)

    airports, prohibited, allowed_wp, weather_db, oncoming = load_environment_data()
    agents = [QLearningAgent() for _ in range(len(AGENT_ROUTES))]

    for i, agent in enumerate(agents):
        pkl = save / f"q_table_agent{i + 1}.pkl"
        if resume and pkl.exists():
            agent.load(str(pkl))
            print(f"  Агент {i+1}: загружены веса ({len(agent.Q)} состояний), ε={agent.epsilon:.3f}")

    print(f"\n{'='*65}")
    print(f"  ОБУЧЕНИЕ Q-LEARNING | Агентов: {len(agents)} | Эпизодов: {n_episodes}")
    print(f"  Действий: {FlightEnvironment.N_ACTIONS} "
          f"({FlightEnvironment.N_ALTITUDES} эшелонов x {K_NEAREST_WAYPOINTS} wp)")
    print(f"  REWARD_PROGRESS_PER_KM={REWARD_PROGRESS_PER_KM}  "
          f"REWARD_ARRIVAL={REWARD_ARRIVAL}")
    print(f"{'='*65}")
    for i in range(len(AGENT_ROUTES)):
        ac = _make_aircraft(i)
        r  = AGENT_ROUTES[i]
        print(f"  Agent {i+1}: {ac.preset['full_name']} | "
              f"{r['departure']['name']} -> {r['arrival']['name']} "
              f"({r['gc_distance_km']:.0f} km) | "
              f"Fuel (full): {ac.max_fuel_kg:.0f} kg")
    print(f"  Встречных ВС: {len(oncoming)}")
    print(f"{'='*65}\n")

    reward_hist: List[List[float]] = [[] for _ in agents]
    arrived_hist: List[List[int]] = [[] for _ in agents]

    for ep in range(1, n_episodes + 1):
        for ai in range(len(agents)):
            env       = _make_env(ai, airports, prohibited, allowed_wp, weather_db, oncoming)
            obs       = env.reset()
            ep_reward = 0.0
            done      = False

            while not done:
                action          = agents[ai].select_action(obs)
                next_obs, r, done = env.step(action)
                agents[ai].update(obs, action, r, next_obs, done)
                obs        = next_obs
                ep_reward += r

            agents[ai].decay_epsilon()
            reward_hist[ai].append(ep_reward)
            res = env.get_result()
            arrived_hist[ai].append(1 if res["arrived"] else 0)

        if ep % TRAIN_CONFIG["log_every"] == 0 or ep == 1:
            parts = []
            for ai in range(len(agents)):
                avg_r   = np.mean(reward_hist[ai][-100:])
                arr_pct = np.mean(arrived_hist[ai][-100:]) * 100
                eps     = agents[ai].epsilon
                parts.append(f"Ag{ai+1}: avg_r={avg_r:>9.0f} arr={arr_pct:>5.1f}% eps={eps:.3f}")
            print(f"  Ep {ep:>5}/{n_episodes} | " + " | ".join(parts))

    for i, agent in enumerate(agents):
        agent.save(str(save / f"q_table_agent{i + 1}.pkl"))

    return _run_greedy(agents, airports, prohibited, allowed_wp, weather_db, oncoming, save)


# ===========================================================================
# INFERENCE
# ===========================================================================
def inference(save_dir: str = "") -> List[Dict]:
    """Запускает финальный полёт с загруженными весами. Обучение не нужно."""
    save = Path(save_dir)
    for i in range(1, len(AGENT_ROUTES) + 1):
        pkl = save / f"q_table_agent{i}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(
                f"Файл весов не найден: {pkl}\n"
                f"Сначала запустите: python flight_env.py")

    airports, prohibited, allowed_wp, weather_db, oncoming = load_environment_data()
    agents = [QLearningAgent.from_file(str(save / f"q_table_agent{i + 1}.pkl"))
              for i in range(len(AGENT_ROUTES))]

    print(f"\n{'='*65}")
    print(f"  INFERENCE")
    for i, ag in enumerate(agents):
        print(f"  Agent {i+1}: {len(ag.Q)} states, eps=0")
    print(f"{'='*65}\n")

    return _run_greedy(agents, airports, prohibited, allowed_wp, weather_db, oncoming, save)


# ===========================================================================
# ФИНАЛЬНЫЙ GREEDY-ПОЛЁТ
# ===========================================================================
def _run_greedy(agents, airports, prohibited, allowed_wp,
                weather_db, oncoming, save: Path) -> List[Dict]:
    print(f"\n{'='*65}")
    print(f"  FINAL FLIGHT (greedy, eps=0)")
    print(f"{'='*65}")

    results = []
    for ai in range(len(AGENT_ROUTES)):
        env = _make_env(ai, airports, prohibited, allowed_wp, weather_db, oncoming)
        obs = env.reset()
        agents[ai].epsilon = 0.0
        done = False

        while not done:
            action       = agents[ai].select_action(obs)
            obs, _, done = env.step(action)

        res = env.get_result()
        results.append(res)

        status = "ARRIVED" if res["arrived"] else f"missed ({res['dist_to_goal_final_km']:.0f} km to goal)"
        print(f"\n  [AC #{res['plane_id']}] {res['departure']} -> {res['arrival']}  [{status}]")
        print(f"  Type: {res['aircraft_type']} | "
              f"Distance: {res['total_distance_km']:.0f} km | "
              f"Steps: {res['steps_taken']}")
        print(f"  Fuel burned: {res['total_fuel_burned_kg']:.0f} kg | "
              f"Remaining: {res['fuel_remaining_kg']:.0f} kg")
        print(f"  Total penalty: {res['total_penalty']:.0f}")

        if res["penalties"]:
            print(f"  Penalties ({len(res['penalties'])} events):")
            for p in res["penalties"]:
                print(f"    t={p['t']:>3} | {p['reason']:<60} | {p['value']:>8.0f}")

        print(f"\n  Route:")
        for step in res["path"]:
            print(f"    t={step['t']:>3} | {step['name']:<22} | lvl.{step['altitude_level']} | "
                  f"fuel.burned {step['fuel_burned_step_kg']:>7.0f} kg | "
                  f"wind x{step['wind_effect']:.2f} | "
                  f"progress {step['progress_km']:>7.1f} km | "
                  f"penalty {step['step_penalty']:>7.0f}")

    out_path = save / "two_planes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Результаты: {out_path}")
    print(f"{'='*65}\n")
    return results


# ===========================================================================
# ТОЧКА ВХОДА
# ===========================================================================
if __name__ == "__main__":
    import time
    t0   = time.time()
    # По умолчанию запускаем обучение
    train()

    print(f"Время: {time.time() - t0:.2f} сек.")