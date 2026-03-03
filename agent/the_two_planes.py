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
# РАЗДЕЛ 0: СОВМЕСТИМОСТЬ (Numpy 2.0 -> 1.x)
# ===========================================================================
class CompatibleUnpickler(pickle.Unpickler):
    """Обеспечивает загрузку весов, сохраненных в numpy 2.0, на старых версиях."""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

# ===========================================================================
# РАЗДЕЛ 1: ШТРАФЫ И НАГРАДЫ
# ---------------------------------------------------------------------------
# Масштаб: прогресс за шаг ~300-600 км × 10.0 = 3000-6000.
# Штраф за зигзаг 60° = 2.5 × (60-15) = 112 — заметный, но не блокирующий.
# REWARD_ARRIVAL >> суммарных штрафов за весь маршрут.
# ===========================================================================

# ── Погодные штрафы (масштабируются с интенсивностью) ─────────────────────
# Логика: макс. штраф за непогоду >> стоимость смены 2-3 эшелонов.
# Смена 1 эшелона = ~50, 2 = ~150, 3 = ~300 очков штрафа.
# Сильный ветер 35 м/с = 35*12 = 420, гроза 1.0 = 500, турб. 3 = 450 >> смены.
PENALTY_WEATHER_WIND_PER_MS      = 12    # за каждый м/с скорости ветра
PENALTY_WEATHER_STORM_PER_UNIT   = 500   # за единицу storm_power (0..1)
PENALTY_WEATHER_TURB_PER_LEVEL   = 150   # за уровень турбулентности (0..3)
PENALTY_WEATHER_ICE_PER_LEVEL    = 120   # за уровень обледенения (0..3)
WIND_FREE_THRESHOLD_MS           = 10    # ветер ≤10 м/с — бесплатно

# ── Конфликты ─────────────────────────────────────────────────────────────
PENALTY_AGENT_CONFLICT           = 300
ONCOMING_MIN_DISTANCE_KM         = 300
PENALTY_CONFLICT_LATERAL         = 200

# ── Смена эшелона ─────────────────────────────────────────────────────────
PENALTY_ALT_CHANGE_PER_LEVEL     = 50    # за каждый уровень изменения (разовый)
REWARD_ALT_CHANGE                = 30    # небольшое поощрение за смену эшелона
PENALTY_ALT_STAGNATION_STEPS    = 10    # начинать штрафовать после N шагов на одном эшелоне
PENALTY_ALT_STAGNATION_PER_STEP = 15    # штраф за каждый шаг сверх лимита

# ── Курс и прогресс ───────────────────────────────────────────────────────
PENALTY_PER_DEG_HEADING_CHANGE   = 2.5
PENALTY_NO_PROGRESS              = 500

REWARD_PROGRESS_PER_KM           = 10.0
REWARD_ARRIVAL                   = 50000

# ===========================================================================
# РАЗДЕЛ 2: ПАРАМЕТРЫ ОБУЧЕНИЯ И СРЕДЫ
# ===========================================================================
AGENT_MIN_DISTANCE_KM = 200
K_NEAREST_WAYPOINTS   = 8
MAX_T                 = 150
ARRIVAL_RADIUS_KM     = 100
FORWARD_CONE_DEG      = 80    # строже — ±80° от курса на цель

TRAIN_CONFIG = {
    "n_episodes":    1660,
    "alpha":         0.2,
    "gamma":         0.99,
    "epsilon":       1.0,
    "epsilon_min":   0.05,
    "epsilon_decay": 0.998,   # к эп.1000 epsilon≈0.14, к эп.3000 epsilon≈0.05 — плавный переход
    "log_every":     1,
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
        "arrival":         {"name": "Magadan_GDX", "lat": 59.9103, "lon": 150.7203},
        "gc_distance_km":  5663.0,
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
    """Хранит погоду по (name, z, t). Lookup O(1) через индекс по (z, t)."""

    def __init__(self, records: List[Dict]):
        # Индекс: (z, t) → список (dist_sq_fn, rec) — строим при init
        self._by_zt: Dict[Tuple[int,int], List[Dict]] = {}
        for r in records:
            key = (int(r["z"]), int(r["t"]))
            self._by_zt.setdefault(key, []).append(r)

    def get(self, lat: float, lon: float, z: int, t: int) -> Optional[Dict]:
        """Ближайший погодный узел для эшелона z и шага t. O(|узлов на z,t|)."""
        t_clamped = max(1, min(MAX_T, t))
        recs = self._by_zt.get((z, t_clamped))
        if not recs:
            return None
        best_dist, best = float("inf"), None
        for rec in recs:
            dlat = lat - rec["lat"]
            dlon = lon - rec["lon"]
            d2 = dlat * dlat + dlon * dlon   # квадрат угловой дистанции, точно для сравнения
            if d2 < best_dist:
                best_dist, best = d2, rec
        return best

    def wind_effect(self, aircraft_az: float,
                    wind_dir: float, wind_speed: float) -> float:
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
        """Расход топлива кг/км с учётом начальной загрузки и ветра."""
        return self.preset["fuel_per_km_empty_kg"] * self.mass_ratio * wind_effect

    def fuel_per_km_dynamic(self, current_mass_kg: float, wind_effect: float = 1.0) -> float:
        """Расход топлива кг/км с учётом ТЕКУЩЕЙ массы (уменьшается по мере выгорания топлива)."""
        dynamic_ratio = max(1.0, current_mass_kg / self.preset["empty_mass_kg"])
        return self.preset["fuel_per_km_empty_kg"] * dynamic_ratio * wind_effect

    def fuel_for_level_change(self, levels: int, current_mass_kg: float = 0.0) -> float:
        """Расход на смену |levels| эшелонов. Учитывает текущую массу самолёта."""
        n = abs(levels)
        if n == 0:
            return 0.0
        mass = current_mass_kg if current_mass_kg > 0 else self.total_mass_kg
        dyn_ratio = max(1.0, mass / self.preset["empty_mass_kg"])
        base = self.preset["fuel_climb_per_level"] * dyn_ratio
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
    plane_id:             int
    current_lat:          float
    current_lon:          float
    current_t:            int
    altitude_level:       int
    fuel_remaining_kg:    float
    current_mass_kg:      float            # уменьшается с каждым шагом по мере выгорания топлива
    dist_to_goal_km:      float
    total_penalty:        float = 0.0
    steps_at_current_alt: int   = 0        # счётчик шагов на текущем эшелоне
    prev_azimuth:         float = -1.0    # курс предыдущего шага; -1 = нет данных
    visited_wp:           set   = field(default_factory=set)
    penalty_log:          List[Dict] = field(default_factory=list)
    path:                 List[Dict] = field(default_factory=list)

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

    Жёсткие правила (физически нарушить невозможно):
      - Агент выбирает ТОЛЬКО из _get_candidates → только разрешённые wp,
        только те, чей путь не пересекает запретные зоны, только непосещённые.
      - Топливо не уходит в минус: при нехватке шаг завершает эпизод.
      - k=8 ближайших — агент физически не может прыгнуть дальше.

    Мягкие ограничения (штрафы, обучает агента их избегать):
      - Конфликты с другими ВС, погода, эшелон 1, смена эшелонов.

    Reward: REWARD_PROGRESS_PER_KM × км_прогресса - топливо - штрафы
      + REWARD_ARRIVAL при прилёте (всегда выгоднее долететь).
    """

    N_ALTITUDES = 5
    N_ACTIONS   = N_ALTITUDES * K_NEAREST_WAYPOINTS  # эшелон (1..5) + wp = 5×8 = 40 действий

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

        # Разрешённые wp вне запретных зон — фильтруем один раз при инициализации
        self.allowed_wp: Dict[str, Dict] = {
            name: wp for name, wp in allowed_wp.items()
            if not wp_in_any_prohibited(wp, prohibited)
        }

        self.arrival_lat  = route["arrival"]["lat"]
        self.arrival_lon  = route["arrival"]["lon"]
        self.arrival_name = route["arrival"]["name"]
        self.gc_dist      = route["gc_distance_km"]

        # Предвычисляем список wp, безопасных от запретных зон, для быстрого _get_candidates.
        # Для каждого wp сохраняем: пересекает ли прямой путь от аэропорта вылета запрет.
        # Во время эпизода только проверяем дистанцию и visited — без polygon-тестов.
        dep = route["departure"]
        safe: List[Dict] = []
        for wp in self.allowed_wp.values():
            # wp сам не в зоне (гарантировано allowed_wp выше)
            # Не проверяем сегмент здесь — он зависит от текущей позиции агента,
            # которая меняется. Проверка сегментов перенесена в _get_candidates,
            # но только для топ-K кандидатов по расстоянию (а не для всего списка).
            safe.append(wp)
        self._wp_list: List[Dict] = safe

        self._reset_state()

    # ──────────────────────────────────────────────────────────────────
    # GYM-интерфейс
    # ──────────────────────────────────────────────────────────────────
    def reset(self) -> Tuple:
        self._reset_state()
        return self._obs()

    def step(self, action: int) -> Tuple[Tuple, float, bool]:
        """
        action = alt_idx * K_NEAREST_WAYPOINTS + wp_idx
          alt_idx ∈ [0..4] → эшелон 1..5
          wp_idx  ∈ [0..7] → кандидат из _get_candidates
        Агент управляет ОДНОВРЕМЕННО эшелоном и выбором waypoint.
        Прилёт проверяется по позиции ПОСЛЕ хода.
        """
        action   = int(action)
        alt_idx  = action // K_NEAREST_WAYPOINTS
        wp_idx_a = action % K_NEAREST_WAYPOINTS
        new_alt  = max(1, min(self.N_ALTITUDES, alt_idx + 1))  # 1..5

        candidates = self._get_candidates(
            self.state.current_lat,
            self.state.current_lon,
            self.state.visited_wp,
        )

        if not candidates:
            return self._obs(), -500.0, True

        wp_idx  = min(wp_idx_a, len(candidates) - 1)
        next_wp = candidates[wp_idx]

        reward, done = self._compute_step(next_wp, new_alt, candidates)

        # Проверка прилёта — по позиции ПОСЛЕ хода (state уже обновлён в _compute_step)
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
        dep        = self.route["departure"]
        d0         = haversine(dep["lat"], dep["lon"], self.arrival_lat, self.arrival_lon)
        init_az    = azimuth(dep["lat"], dep["lon"], self.arrival_lat, self.arrival_lon)
        start_fuel = self.aircraft.max_fuel_kg
        start_mass = self.aircraft.total_mass_kg
        start_alt  = self.route.get("start_altitude", 3)
        start_t    = self.route.get("start_t", 1)
        self.state = FlightState(
            plane_id=self.aircraft.plane_id,
            current_lat=dep["lat"],
            current_lon=dep["lon"],
            current_t=start_t,
            altitude_level=start_alt,
            fuel_remaining_kg=start_fuel,
            current_mass_kg=start_mass,
            dist_to_goal_km=d0,
            prev_azimuth=init_az,
            steps_at_current_alt=0,
            visited_wp={dep["name"]},
            path=[{
                "name":                dep["name"],
                "lat":                 dep["lat"],
                "lon":                 dep["lon"],
                "t":                   start_t,
                "altitude_level":      start_alt,
                "fuel_remaining_kg":   round(start_fuel, 1),
                "current_mass_kg":     round(start_mass, 1),
                "step_penalty":        0.0,
                "dist_km":             0.0,
                "fuel_burned_step_kg": 0.0,
                "wind_effect":         1.0,
                "progress_km":         0.0,
            }]
        )

    def _obs(self) -> Tuple:
        """
        (progress_bucket, altitude_level, heading_bucket, conflict, stagnation_bucket)

        progress_bucket   — прогресс × 20 → 0..20
        altitude_level    — текущий эшелон 1..5
        heading_bucket    — отклонение курса от прямой к цели // 15 → 0..6
        conflict          — 0/1: нарушение 200 км между агентами
        stagnation_bucket — 0/1: агент засиделся на одном эшелоне
        """
        progress = 1.0 - self.state.dist_to_goal_km / max(1.0, self.gc_dist)
        progress_bucket = max(0, min(20, int(progress * 20)))

        alt_bucket = self.state.altitude_level  # 1..5

        # Отклонение текущего курса от прямой к цели
        goal_az  = azimuth(self.state.current_lat, self.state.current_lon,
                           self.arrival_lat, self.arrival_lon)
        cur_az   = self.state.prev_azimuth if self.state.prev_azimuth >= 0 else goal_az
        dev      = abs(angle_diff(cur_az, goal_az))
        heading_bucket = min(6, int(dev / 15))

        conflict = 0
        for other in self.other_planes:
            if "departure" not in other:
                continue
            if other.get("plane_number") == self.state.plane_id:
                continue
            pos = self._interp_other_pos(other, self.state.current_t)
            if pos:
                d = haversine(self.state.current_lat, self.state.current_lon,
                              pos[0], pos[1])
                if d < AGENT_MIN_DISTANCE_KM:
                    conflict = 1
                    break

        stagnation = 1 if self.state.steps_at_current_alt > PENALTY_ALT_STAGNATION_STEPS else 0

        return (progress_bucket, alt_bucket, heading_bucket, conflict, stagnation)

    def _get_candidates(self, lat: float, lon: float,
                        visited: set) -> List[Dict]:
        """
        K wp, отсортированных по "прямолинейности" курса к цели.

        Ключевое: сортировка по убыванию проекции wp на вектор (текущая→цель).
        wp[0] = самый "прямой" к цели, wp[7] = самый боковой.
        Это значит что action=0 при greedy-политике всегда ведёт прямо.

        Жёсткие фильтры:
          1. Не посещался.
          2. wp ближе к цели чем текущая позиция (прогресс гарантирован).
          3. Угол отклонения от курса ≤ FORWARD_CONE_DEG.
          4. Путь не пересекает запретную зону (для топ-2K).

        Fallback (обход зоны): если нет wp с прогрессом в конусе — ищем шире.
        """
        if not self._wp_list:
            return []

        goal_az     = azimuth(lat, lon, self.arrival_lat, self.arrival_lon)
        cur_to_goal = haversine(lat, lon, self.arrival_lat, self.arrival_lon)

        # Быстрая предсортировка по квадрату угловой дистанции
        scored: List[Tuple[float, Dict]] = []
        for wp in self._wp_list:
            if wp.get("name", "") in visited:
                continue
            dlat = lat - wp["lat"]
            dlon = lon - wp["lon"]
            d2 = dlat * dlat + dlon * dlon
            if d2 < 1e-4:
                continue
            scored.append((d2, wp))
        scored.sort(key=lambda x: x[0])

        # Проверяем топ-2K на прогресс, конус, зоны
        candidates_scored: List[Tuple[float, Dict]] = []  # (score, wp); score: меньше = прямее

        for d2, wp in scored[:K_NEAREST_WAYPOINTS * 3]:
            wp_to_goal = haversine(wp["lat"], wp["lon"], self.arrival_lat, self.arrival_lon)
            if wp_to_goal >= cur_to_goal:          # нет прогресса — пропускаем
                continue

            wp_az     = azimuth(lat, lon, wp["lat"], wp["lon"])
            deviation = abs(angle_diff(wp_az, goal_az))
            if deviation > FORWARD_CONE_DEG:       # вне конуса — пропускаем
                continue

            crosses = any(
                segment_crosses_polygon(lat, lon, wp["lat"], wp["lon"], zone["points"])
                for zone in self.prohibited
            )
            if crosses:
                continue

            # Ключевой score: угловое отклонение от курса к цели (меньше = прямее)
            candidates_scored.append((deviation, wp))

        # Аэропорт прилёта — нулевое отклонение (он и есть цель)
        arr_name = self.arrival_name
        arr_d    = haversine(lat, lon, self.arrival_lat, self.arrival_lon)
        if arr_name not in visited and arr_d < cur_to_goal and arr_d < 3000:
            arr_cross = any(
                segment_crosses_polygon(lat, lon, self.arrival_lat, self.arrival_lon,
                                        zone["points"])
                for zone in self.prohibited
            )
            if not arr_cross:
                arr_wp = {"name": arr_name, "lat": self.arrival_lat, "lon": self.arrival_lon}
                candidates_scored.append((0.0, arr_wp))

        # Fallback: нет кандидатов в конусе — берём любые wp с прогрессом (обход зоны)
        if not candidates_scored:
            for d2, wp in scored[:K_NEAREST_WAYPOINTS * 3]:
                wp_to_goal = haversine(wp["lat"], wp["lon"], self.arrival_lat, self.arrival_lon)
                if wp_to_goal >= cur_to_goal:
                    continue
                crosses = any(
                    segment_crosses_polygon(lat, lon, wp["lat"], wp["lon"], zone["points"])
                    for zone in self.prohibited
                )
                if not crosses:
                    wp_az     = azimuth(lat, lon, wp["lat"], wp["lon"])
                    deviation = abs(angle_diff(wp_az, goal_az))
                    candidates_scored.append((deviation, wp))
                if len(candidates_scored) >= K_NEAREST_WAYPOINTS:
                    break

        # Второй fallback: тупик у зоны — берём ближайшие БЕЗ фильтра прогресса,
        # но строго в конусе FORWARD_CONE_DEG (назад не летим никогда)
        if not candidates_scored:
            for d2, wp in scored[:K_NEAREST_WAYPOINTS * 4]:
                wp_az     = azimuth(lat, lon, wp["lat"], wp["lon"])
                deviation = abs(angle_diff(wp_az, goal_az))
                if deviation > FORWARD_CONE_DEG:
                    continue
                crosses = any(
                    segment_crosses_polygon(lat, lon, wp["lat"], wp["lon"], zone["points"])
                    for zone in self.prohibited
                )
                if not crosses:
                    candidates_scored.append((deviation, wp))
                if len(candidates_scored) >= K_NEAREST_WAYPOINTS:
                    break

        # Сортируем: wp[0] = минимальное отклонение от прямого курса (самый "прямой")
        candidates_scored.sort(key=lambda x: x[0])
        return [wp for _, wp in candidates_scored[:K_NEAREST_WAYPOINTS]]

    def _compute_step(self, next_wp: Dict, new_alt: int,
                      candidates: List[Dict]) -> Tuple[float, bool]:
        """
        Расчёт одного шага симуляции.
        Агент управляет эшелоном и выбором waypoint.
        Топливо учитывается: снижение массы → меньший расход на следующих шагах.
        Reward = REWARD_PROGRESS_PER_KM × progress_km + бонусы - штрафы.
        """
        s         = self.state
        s.current_t += 1
        penalty   = 0.0
        reward_bonus = 0.0

        next_lat   = next_wp["lat"]
        next_lon   = next_wp["lon"]
        next_name  = next_wp.get("name", "")
        prev_alt   = s.altitude_level
        level_delta = abs(new_alt - prev_alt)

        # ── Смена эшелона: штраф + топливо на набор/снижение ─────────
        if level_delta > 0:
            alt_fuel = self.aircraft.fuel_for_level_change(level_delta, s.current_mass_kg)
            alt_penalty = PENALTY_ALT_CHANGE_PER_LEVEL * level_delta * (1.5 if level_delta >= 2 else 1.0)
            penalty += alt_penalty
            s.add_penalty(f"Смена эшелона FL{prev_alt*100}→FL{new_alt*100}", alt_penalty)
            # Топливо на манёвр
            s.current_mass_kg     = max(self.aircraft.preset["empty_mass_kg"], s.current_mass_kg - alt_fuel)
            s.fuel_remaining_kg   = max(0.0, s.fuel_remaining_kg - alt_fuel)
            # Поощрение за смену эшелона (стимулирует исследование)
            reward_bonus += REWARD_ALT_CHANGE
            s.steps_at_current_alt = 0
        else:
            s.steps_at_current_alt += 1
            # Штраф за долгое засиживание на одном эшелоне
            if s.steps_at_current_alt > PENALTY_ALT_STAGNATION_STEPS:
                stag_p = PENALTY_ALT_STAGNATION_PER_STEP
                penalty += stag_p
                s.add_penalty(f"Стагнация на FL{new_alt*100} ({s.steps_at_current_alt} шагов)", stag_p)

        s.altitude_level = new_alt

        # ── Погода: штраф масштабируется с интенсивностью ─────────────
        weather  = self.weather_db.get(s.current_lat, s.current_lon, new_alt, s.current_t)
        wind_eff = 1.0
        w_wind, w_storm, w_turb, w_ice = 0.0, 0.0, 0.0, 0.0
        if weather:
            az       = azimuth(s.current_lat, s.current_lon, next_lat, next_lon)
            wind_eff = self.weather_db.wind_effect(
                az, weather["wind_dir"], weather["wind_speed"])
            w_speed = weather.get("wind_speed", 0.0)
            w_turb  = weather.get("turbulence", 0)
            w_ice   = weather.get("ice", 0)
            w_storm = weather.get("storm_power", 0.0)

            # Ветер — бесплатен до порога, далее линейно
            wind_excess = max(0.0, w_speed - WIND_FREE_THRESHOLD_MS)
            w_wind = PENALTY_WEATHER_WIND_PER_MS * wind_excess

            weather_penalty = (w_wind
                               + PENALTY_WEATHER_STORM_PER_UNIT * w_storm
                               + PENALTY_WEATHER_TURB_PER_LEVEL * w_turb
                               + PENALTY_WEATHER_ICE_PER_LEVEL  * w_ice)
            if weather_penalty > 0:
                penalty += weather_penalty
                if w_wind > 0:
                    s.add_penalty(f"Ветер {w_speed:.1f} м/с (эффект ×{wind_eff:.2f})", w_wind)
                if w_storm > 0:
                    s.add_penalty(f"Гроза {w_storm:.2f}", PENALTY_WEATHER_STORM_PER_UNIT * w_storm)
                if w_turb > 0:
                    s.add_penalty(f"Турбулентность ур.{w_turb}", PENALTY_WEATHER_TURB_PER_LEVEL * w_turb)
                if w_ice > 0:
                    s.add_penalty(f"Обледенение ур.{w_ice}", PENALTY_WEATHER_ICE_PER_LEVEL * w_ice)

        # ── Топливо: используем ТЕКУЩУЮ массу (снижается каждый шаг) ──
        dist_km   = haversine(s.current_lat, s.current_lon, next_lat, next_lon)
        fuel_step = dist_km * self.aircraft.fuel_per_km_dynamic(s.current_mass_kg, wind_eff)
        s.current_mass_kg   = max(self.aircraft.preset["empty_mass_kg"], s.current_mass_kg - fuel_step)
        s.fuel_remaining_kg = max(0.0, s.fuel_remaining_kg - fuel_step)

        # ── Дистанция между агентами (штраф за нарушение 200 км) ──────
        min_other_d = float("inf")
        for other in self.other_planes:
            num = other.get("plane_number", "?")
            if num == s.plane_id or "departure" not in other:
                continue
            pos = self._interp_other_pos(other, s.current_t)
            if pos is None:
                continue
            d = haversine(next_lat, next_lon, pos[0], pos[1])
            min_other_d = min(min_other_d, d)
            if d < AGENT_MIN_DISTANCE_KM:
                penalty += PENALTY_AGENT_CONFLICT
                s.add_penalty(f"Агент #{num} < {AGENT_MIN_DISTANCE_KM} км (d={d:.0f})",
                              PENALTY_AGENT_CONFLICT)

        # ── Конфликты со встречными ВС ────────────────────────────────
        min_oncoming_d = float("inf")
        for other in self.other_planes:
            if other.get("plane_number") == s.plane_id or "departure" in other:
                continue
            pos = self._interp_other_pos(other, s.current_t)
            if pos is None:
                continue
            d = haversine(next_lat, next_lon, pos[0], pos[1])
            min_oncoming_d = min(min_oncoming_d, d)
            if d < ONCOMING_MIN_DISTANCE_KM:
                penalty += PENALTY_CONFLICT_LATERAL
                s.add_penalty(f"ВС #{other.get('plane_number','?')} < 300 км (d={d:.0f})",
                              PENALTY_CONFLICT_LATERAL)

        # ── Штраф за резкую смену курса (зигзаги) ────────────────────
        cur_az = azimuth(s.current_lat, s.current_lon, next_lat, next_lon)
        if s.prev_azimuth >= 0:
            heading_change = abs(angle_diff(cur_az, s.prev_azimuth))
            if heading_change > 15:
                p = PENALTY_PER_DEG_HEADING_CHANGE * (heading_change - 15)
                penalty += p
                s.add_penalty(f"Смена курса {heading_change:.0f}°", p)

        # ── Прогресс к цели ───────────────────────────────────────────
        new_dist_to_goal = haversine(next_lat, next_lon, self.arrival_lat, self.arrival_lon)
        progress_km      = s.dist_to_goal_km - new_dist_to_goal

        if progress_km <= 0:
            penalty += PENALTY_NO_PROGRESS
            s.add_penalty("Нет прогресса", PENALTY_NO_PROGRESS)

        # ── Обновляем состояние ───────────────────────────────────────
        s.total_penalty  += penalty
        s.current_lat     = next_lat
        s.current_lon     = next_lon
        s.dist_to_goal_km = new_dist_to_goal
        s.prev_azimuth    = cur_az
        s.visited_wp.add(next_name)

        s.path.append({
            "name":                next_name,
            "lat":                 round(next_lat, 6),
            "lon":                 round(next_lon, 6),
            "t":                   s.current_t,
            "altitude_level":      new_alt,
            "fuel_remaining_kg":   round(s.fuel_remaining_kg, 1),
            "current_mass_kg":     round(s.current_mass_kg, 1),
            "step_penalty":        round(penalty, 1),
            "dist_km":             round(dist_km, 2),
            "fuel_burned_step_kg": round(fuel_step, 1),
            "wind_effect":         round(wind_eff, 3),
            "progress_km":         round(progress_km, 1),
            "weather": {
                "wind_speed":  round(weather.get("wind_speed", 0), 1)  if weather else 0,
                "storm":       round(weather.get("storm_power", 0), 2) if weather else 0,
                "turbulence":  weather.get("turbulence", 0)            if weather else 0,
                "ice":         weather.get("ice", 0)                   if weather else 0,
            },
            "dist_to_agent_km":   round(min_other_d,    1) if min_other_d    < float("inf") else -1,
            "dist_to_oncoming_km":round(min_oncoming_d, 1) if min_oncoming_d < float("inf") else -1,
        })

        # reward = прогресс + бонус за смену эшелона - штрафы
        reward = REWARD_PROGRESS_PER_KM * max(0.0, progress_km) + reward_bonus - penalty
        return reward, False

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
            "final_mass_kg":        round(s.current_mass_kg, 1),
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

    Observation: (progress_bucket, conflict)  — 21 × 2 = 42 состояния
    Actions: 8 — индекс wp из _get_candidates (эшелон фиксирован = FIXED_ALTITUDE)
    Reward: REWARD_PROGRESS_PER_KM × км_прогресса - штрафы (топливо не вычитается)
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
            data = CompatibleUnpickler(f).load()
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


def _make_aircraft(agent_idx: int, agent_routes: List[Dict] = None, agent_aircraft: List[Dict] = None) -> Aircraft:
    """Создаёт Aircraft из переданных данных или AGENT_ROUTES + AGENT_AIRCRAFT."""
    routes = agent_routes if agent_routes is not None else AGENT_ROUTES
    aircraft_configs = agent_aircraft if agent_aircraft is not None else AGENT_AIRCRAFT
    
    route  = routes[agent_idx]
    cfg    = aircraft_configs[agent_idx]
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
              weather_db, oncoming,
              agent_routes: List[Dict] = None,
              agent_aircraft: List[Dict] = None) -> FlightEnvironment:
    """Создаёт среду. other_planes = второй агент-маршрут + встречные из simulation_result."""
    routes = agent_routes if agent_routes is not None else AGENT_ROUTES
    other_routes = [routes[j] for j in range(len(routes)) if j != agent_idx]
    other_planes = other_routes + oncoming
    return FlightEnvironment(
        route        = routes[agent_idx],
        aircraft     = _make_aircraft(agent_idx, agent_routes, agent_aircraft),
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
            print(f"  Агент {i+1}: загружены веса ({len(agent.Q)} состояний), epsilon={agent.epsilon:.3f}")

    print(f"\n{'='*65}")
    print(f"  ОБУЧЕНИЕ Q-LEARNING | Агентов: {len(agents)} | Эпизодов: {n_episodes}")
    print(f"  Действий: {FlightEnvironment.N_ACTIONS} "
          f"({FlightEnvironment.N_ALTITUDES} эшелонов × {K_NEAREST_WAYPOINTS} wp)")
    print(f"  REWARD_PROGRESS_PER_KM={REWARD_PROGRESS_PER_KM}  "
          f"REWARD_ARRIVAL={REWARD_ARRIVAL}")
    print(f"{'='*65}")
    for i in range(len(AGENT_ROUTES)):
        ac = _make_aircraft(i)
        r  = AGENT_ROUTES[i]
        print(f"  Агент {i+1}: {ac.preset['full_name']} | "
              f"{r['departure']['name']} → {r['arrival']['name']} "
              f"({r['gc_distance_km']:.0f} км) | "
              f"Топливо (полный бак): {ac.max_fuel_kg:.0f} кг")
    print(f"  Встречных ВС: {len(oncoming)}")
    print(f"{'='*65}\n")

    reward_hist: List[List[float]] = [[] for _ in agents]
    arrived_hist: List[List[int]] = [[] for _ in agents]

    # Среды создаются ОДИН РАЗ — дорогая инициализация (фильтрация wp, зон) вне цикла
    envs = [_make_env(ai, airports, prohibited, allowed_wp, weather_db, oncoming)
            for ai in range(len(agents))]

    for ep in range(1, n_episodes + 1):
        for ai in range(len(agents)):
            obs       = envs[ai].reset()   # только сбрасывает state, не пересоздаёт среду
            ep_reward = 0.0
            done      = False

            while not done:
                action            = agents[ai].select_action(obs)
                next_obs, r, done = envs[ai].step(action)
                agents[ai].update(obs, action, r, next_obs, done)
                obs        = next_obs
                ep_reward += r

            agents[ai].decay_epsilon()
            reward_hist[ai].append(ep_reward)
            res = envs[ai].get_result()
            arrived_hist[ai].append(1 if res["arrived"] else 0)

        if ep % TRAIN_CONFIG["log_every"] == 0 or ep == 1:
            parts = []
            for ai in range(len(agents)):
                avg_r   = np.mean(reward_hist[ai][-100:])
                arr_pct = np.mean(arrived_hist[ai][-100:]) * 100
                eps     = agents[ai].epsilon
                parts.append(f"Аг{ai+1}: avg_r={avg_r:>9.0f} arr={arr_pct:>5.1f}% epsilon={eps:.3f}")
            print(f"  Эп {ep:>5}/{n_episodes} | " + " | ".join(parts))

    for i, agent in enumerate(agents):
        agent.save(str(save / f"q_table_agent{i + 1}.pkl"))

    return _run_greedy(agents, airports, prohibited, allowed_wp, weather_db, oncoming, save)


# ===========================================================================
# INFERENCE
# ===========================================================================
def inference(save_dir: str = "", agent_routes: List[Dict] = None, agent_aircraft: List[Dict] = None) -> List[Dict]:
    """Запускает финальный полёт с загруженными весами. Обучение не нужно."""
    save = Path(save_dir)
    routes = agent_routes if agent_routes is not None else AGENT_ROUTES
    
    for i in range(1, len(routes) + 1):
        pkl = save / f"q_table_agent{i}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(
                f"Файл весов не найден: {pkl}\n"
                f"Сначала запустите: python flight_env.py")

    airports, prohibited, allowed_wp, weather_db, oncoming = load_environment_data()
    agents = [QLearningAgent.from_file(str(save / f"q_table_agent{i + 1}.pkl"))
              for i in range(len(routes))]

    print(f"\n{'='*65}")
    print(f"  INFERENCE — обучение не требуется")
    for i, ag in enumerate(agents):
        print(f"  Агент {i+1}: {len(ag.Q)} состояний, epsilon=0")
    print(f"{'='*65}\n")

    return _run_greedy(agents, airports, prohibited, allowed_wp, weather_db, oncoming, save, agent_routes, agent_aircraft)


# ===========================================================================
# ФИНАЛЬНЫЙ GREEDY-ПОЛЁТ
# ===========================================================================
def _run_greedy(agents, airports, prohibited, allowed_wp,
                weather_db, oncoming, save: Path,
                agent_routes: List[Dict] = None,
                agent_aircraft: List[Dict] = None) -> List[Dict]:
    print(f"\n{'='*65}")
    print(f"  ФИНАЛЬНЫЙ ПОЛЁТ (greedy, epsilon=0)")
    print(f"{'='*65}")

    results = []
    routes = agent_routes if agent_routes is not None else AGENT_ROUTES
    for ai in range(len(routes)):
        env = _make_env(ai, airports, prohibited, allowed_wp, weather_db, oncoming, agent_routes, agent_aircraft)
        obs = env.reset()
        agents[ai].epsilon = 0.0
        done = False

        while not done:
            action       = agents[ai].select_action(obs)
            obs, _, done = env.step(action)

        res = env.get_result()
        results.append(res)

        status = "ПРИЛЕТЕЛ" if res["arrived"] else f"не долетел ({res['dist_to_goal_final_km']:.0f} км до цели)"
        print(f"\n  [ВС #{res['plane_id']}] {res['departure']} → {res['arrival']}  [{status}]")
        print(f"  Тип: {res['aircraft_type']} | "
              f"Пройдено: {res['total_distance_km']:.0f} км | "
              f"Шагов: {res['steps_taken']}")
        print(f"  Топливо сожжено: {res['total_fuel_burned_kg']:.0f} кг | "
              f"Остаток: {res['fuel_remaining_kg']:.0f} кг")
        print(f"  Суммарный штраф: {res['total_penalty']:.0f}")

        if res["penalties"]:
            print(f"  Штрафы ({len(res['penalties'])} событий):")
            for p in res["penalties"]:
                print(f"    t={p['t']:>3} | {p['reason']:<60} | {p['value']:>8.0f}")

        print(f"\n  Детальный маршрут (каждый шаг):")
        hdr = (f"  {'t':>3} | {'Waypoint':<22} | {'Эш':>3} | "
               f"{'Топл.сожж':>10} | {'Топл.ост':>9} | {'Масса':>9} | "
               f"{'Ветер':>7} | {'Гроза':>6} | {'Турб':>4} | {'Лёд':>4} | "
               f"{'Д.до агента':>12} | {'Д.до встреч':>12} | "
               f"{'Прогр.':>8} | {'Штраф':>7}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for step in res["path"]:
            w = step.get("weather", {})
            d_ag = step.get("dist_to_agent_km", -1)
            d_on = step.get("dist_to_oncoming_km", -1)
            d_ag_str  = f"{d_ag:>9.0f} км" if d_ag >= 0 else f"{'N/A':>12}"
            d_on_str  = f"{d_on:>9.0f} км" if d_on >= 0 else f"{'N/A':>12}"
            print(f"  {step['t']:>3} | {step['name']:<22} | FL{step['altitude_level']*100:<3} | "
                  f"{step['fuel_burned_step_kg']:>8.0f} кг | "
                  f"{step['fuel_remaining_kg']:>7.0f} кг | "
                  f"{step.get('current_mass_kg', 0):>7.0f} кг | "
                  f"{w.get('wind_speed', 0):>5.1f}м/с | "
                  f"{w.get('storm', 0):>5.2f} | "
                  f"{w.get('turbulence', 0):>4} | "
                  f"{w.get('ice', 0):>4} | "
                  f"{d_ag_str} | {d_on_str} | "
                  f"{step['progress_km']:>6.1f} км | "
                  f"{step['step_penalty']:>7.0f}")

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
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "inference":
        inference()
    elif mode == "resume":
        train(resume=True)
    else:
        train()

    print(f"Время: {time.time() - t0:.2f} сек.")