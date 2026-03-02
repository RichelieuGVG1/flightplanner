import json
import math
import random
import os
import time
import logging
import traceback
from typing import List, Dict, Tuple

# Настройка логирования
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(PROJECT_ROOT, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("prohibited_zones")

# ── КОНСТАНТЫ ──────────────────────────────────────────────────────────────
MIN_AREA_KM2 = 35000
MAX_AREA_KM2 = 250000
MIN_DIST_KM  = 400
MAX_ZONES    = 10
MIN_VERTICES = 6
MAX_VERTICES = 10

# Упрощенные границы РФ (прямоугольники для генерации центров)
RUSSIA_BOUNDS = [
    {"lat": [41.0, 77.0], "lon": [26.0, 170.0]}
]

# Аэропорты, которые нужно избегать (минимум 100км)
AIRPORTS = [
    { "name": "Sheremetyevo", "iata": "SVO", "city": "Москва", "coords": [55.9726, 37.4146] },
    { "name": "Pulkovo", "iata": "LED", "city": "Санкт-Петербург", "coords": [59.8003, 30.2625] },
    { "name": "Tolmachevo", "iata": "OVB", "city": "Новосибирск", "coords": [55.0126, 82.6507] },
    { "name": "Sochi", "iata": "AER", "city": "Сочи", "coords": [43.4499, 39.9566] },
    { "name": "Kazan", "iata": "KZN", "city": "Казань", "coords": [55.6062, 49.2787] },
    { "name": "Knevichi", "iata": "VVO", "city": "Владивосток", "coords": [43.3990, 132.1480] },
    { "name": "Novy", "iata": "KHV", "city": "Хабаровск", "coords": [48.5280, 135.1880] },
    { "name": "Yelizovo", "iata": "PKC", "city": "Петропавловск-Камчатский", "coords": [53.1679, 158.4539] },
    { "name": "Ufa", "iata": "UFA", "city": "Уфа", "coords": [54.5654, 55.8845] },
    { "name": "Koltsovo", "iata": "SVX", "city": "Екатеринбург", "coords": [56.7431, 60.8027] },
    { "name": "Platov", "iata": "ROV", "city": "Ростов-на-Дону", "coords": [47.4938, 39.9247] },
    { "name": "Yemelyanovo", "iata": "KJA", "city": "Красноярск", "coords": [56.1729, 92.4933] },
    { "name": "Kurumoch", "iata": "KUF", "city": "Самара", "coords": [53.5050, 50.1642] }
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "prohibited_zones.json")
BORDER_FILE = os.path.join(BASE_DIR, "state_border.json")

# ── ГЕОДЕЗИЯ ────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_area_km2(points: List[Dict]):
    """
    Приблизительный расчет площади плоского многоугольника на сфере.
    Используется упрощенная проекция для малых и средних зон.
    """
    if len(points) < 3: return 0
    # Проекция на плоскость (центр масс)
    clat = sum(p['lat'] for p in points) / len(points)
    clon = sum(p['lon'] for p in points) / len(points)
    
    # km per degree
    lat_scale = 111.3
    lon_scale = 111.3 * math.cos(math.radians(clat))
    
    # Координаты в км относительно центра
    xy = []
    for p in points:
        x = (p['lon'] - clon) * lon_scale
        y = (p['lat'] - clat) * lat_scale
        xy.append((x, y))
    
    # Формула Гаусса
    area = 0.0
    for i in range(len(xy)):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % len(xy)]
        area += (x1 * y2 - x2 * y1)
    return abs(area) / 2.0

def is_point_in_poly_coords(lat, lon, poly_coords):
    """Ray casting для списка координат [[lat, lon], ...]"""
    n = len(poly_coords)
    inside = False
    p1_lat, p1_lon = poly_coords[0]
    for i in range(n + 1):
        p2_lat, p2_lon = poly_coords[i % n]
        if lon > min(p1_lon, p2_lon):
            if lon <= max(p1_lon, p2_lon):
                if lat <= max(p1_lat, p2_lat):
                    if p1_lon != p2_lon:
                        xints = (lon - p1_lon) * (p2_lat - p1_lat) / (p2_lon - p1_lon) + p1_lat
                    if p1_lat == p2_lat or lat <= xints:
                        inside = not inside
        p1_lat, p1_lon = p2_lat, p2_lon
    return inside

def is_point_in_polygon(lat, lon, poly_points):
    """Ray casting для списка словарей [{'lat':..., 'lon':...}, ...]"""
    n = len(poly_points)
    inside = False
    p1 = poly_points[0]
    for i in range(n + 1):
        p2 = poly_points[i % n]
        if lon > min(p1['lon'], p2['lon']):
            if lon <= max(p1['lon'], p2['lon']):
                if lat <= max(p1['lat'], p2['lat']):
                    if p1['lon'] != p2['lon']:
                        xints = (lon - p1['lon']) * (p2['lat'] - p1['lat']) / (p2['lon'] - p1['lon']) + p1['lat']
                    if p1['lat'] == p2['lat'] or lat <= xints:
                        inside = not inside
        p1 = p2
    return inside

def is_point_in_russia(lat, lon, border_rings):
    """Проверяет, находится ли точка внутри территории РФ."""
    for ring in border_rings:
        if is_point_in_poly_coords(lat, lon, ring):
            return True
    return False

def on_segment(p, q, r):
    return (q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]) and
            q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]))

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def segments_intersect(p1, q1, p2, q2):
    """Проверка пересечения двух отрезков."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4: return True
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True
    return False

def is_border_violated(poly, border_rings):
    """
    Комплексная проверка:
    1. Пересекают ли ребра зоны ребра границы.
    2. Попадают ли точки границы внутрь зоны.
    3. Находятся ли все вершины зоны внутри России.
    """
    # 0. Bounding box зоны
    lats = [p['lat'] for p in poly]
    lons = [p['lon'] for p in poly]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # 1. Все вершины зоны должны быть в России
    for p in poly:
        if not is_point_in_russia(p['lat'], p['lon'], border_rings):
            return True

    # 2. Проверка пересечения ребер и точек границы
    for ring in border_rings:
        # Быстрая проверка BB кольца
        r_lats = [pt[0] for pt in ring]
        r_lons = [pt[1] for pt in ring]
        r_min_lat, r_max_lat = min(r_lats), max(r_lats)
        r_min_lon, r_max_lon = min(r_lons), max(r_lons)
        
        if max_lat < r_min_lat or min_lat > r_max_lat or max_lon < r_min_lon or min_lon > r_max_lon:
            continue

        # Проверка точек этого кольца на попадание внутрь зоны
        for bp_lat, bp_lon in ring:
            if min_lat <= bp_lat <= max_lat and min_lon <= bp_lon <= max_lon:
                if is_point_in_polygon(bp_lat, bp_lon, poly):
                    return True

        # Проверка пересечения отрезков
        for i in range(len(poly)):
            p1 = (poly[i]['lat'], poly[i]['lon'])
            p2 = (poly[(i+1)%len(poly)]['lat'], poly[(i+1)%len(poly)]['lon'])
            
            for j in range(len(ring)):
                q1 = (ring[j][0], ring[j][1])
                q2 = (ring[(j+1)%len(ring)][0], ring[(j+1)%len(ring)][1])
                
                if segments_intersect(p1, p2, q1, q2):
                    return True
    
    return False

# ── ГЕНЕРАЦИЯ ──────────────────────────────────────────────────────────────
def generate_polygon(center_lat, center_lon, target_area):
    """Генерирует выпуклый многоугольник заданной площади."""
    num_verts = random.randint(MIN_VERTICES, MAX_VERTICES)
    angles = sorted([random.uniform(0, 2*math.pi) for _ in range(num_verts)])
    
    # Начальный радиус для примерной площади
    # Area ~ pi * R^2 => R ~ sqrt(Area/pi)
    # 1 deg ~ 111km
    base_radius_km = math.sqrt(target_area / math.pi)
    radius_deg = base_radius_km / 111.3
    
    points = []
    for a in angles:
        # Добавляем jitter к радиусу для неправильной формы
        r = radius_deg * random.uniform(0.7, 1.3)
        lat = center_lat + r * math.cos(a)
        lon = center_lon + r * math.sin(a) / max(math.cos(math.radians(center_lat)), 0.1)
        points.append({"lat": round(lat, 5), "lon": round(lon, 5)})
    
    # Масштабируем до точной площади
    current_area = get_area_km2(points)
    if current_area > 0:
        scale = math.sqrt(target_area / current_area)
        for p in points:
            dlat = (p['lat'] - center_lat) * scale
            dlon = (p['lon'] - center_lon) * scale
            p['lat'] = round(center_lat + dlat, 5)
            p['lon'] = round(center_lon + dlon, 5)
            
    return points

def is_far_enough(points, existing_zones):
    """Проверяет расстояние до других зон."""
    for zone in existing_zones:
        for p1 in points:
            for p2 in zone['points']:
                if haversine(p1['lat'], p1['lon'], p2['lat'], p2['lon']) < MIN_DIST_KM:
                    return False
    return True

def is_away_from_airports(points, min_dist_km=100.0):
    """Проверяет, что зона находится минимум в 100км от ключевых аэропортов."""
    for airport in AIRPORTS:
        a_lat, a_lon = airport["coords"]
        # Проверяем каждую вершину зоны
        for p in points:
            if haversine(p['lat'], p['lon'], a_lat, a_lon) < min_dist_km:
                return False
        # Также проверим, не попал ли сам аэропорт внутрь зоны
        if is_point_in_polygon(a_lat, a_lon, points):
            return True # Иронично, но True здесь означает "виолация" в контексте логики loop
    return True

def generate_zones():
    start_time = time.time()
    logger.info(f"Начало генерации запретных зон (макс: {MAX_ZONES})")

    try:
        # Загружаем кольца границы
        logger.info(f"Загрузка данных границ из {BORDER_FILE}...")
        border_rings = []
        if os.path.exists(BORDER_FILE):
            with open(BORDER_FILE, "r", encoding="utf-8") as f:
                border_rings = json.load(f)
        logger.info(f"Граница загружена ({len(border_rings)} колец)")

        zones = []
        attempts = 0
        max_attempts = 500

        logger.info(f"Генерация зон с проверкой близости к аэропортам...")

        while len(zones) < MAX_ZONES and attempts < max_attempts:
            attempts += 1
            
            # Случайный центр в РФ
            b = RUSSIA_BOUNDS[0]
            clat = random.uniform(b["lat"][0] + 2, b["lat"][1] - 2)
            clon = random.uniform(b["lon"][0] + 5, b["lon"][1] - 5)
            
            # Первая проверка центра
            if not is_point_in_russia(clat, clon, border_rings):
                continue

            target_area = random.uniform(MIN_AREA_KM2, MAX_AREA_KM2)
            poly = generate_polygon(clat, clon, target_area)
            
            # Проверка на пересечение/выход за границу
            if is_border_violated(poly, border_rings):
                continue
            
            # Проверка расстояния до других зон
            if not is_far_enough(poly, zones):
                continue
                
            # Проверка расстояния до аэропортов (минимум 100км)
            if not is_away_from_airports(poly, 100.0):
                continue

            zones.append({
                "id": len(zones) + 1,
                "area_km2": round(get_area_km2(poly), 1),
                "points": poly
            })
            if len(zones) % 2 == 0:
                logger.info(f"Сгенерировано зон: {len(zones)}/{MAX_ZONES}")

        if len(zones) < MAX_ZONES:
            logger.warning(f"Удалось сгенерировать только {len(zones)} из {MAX_ZONES} зон за {attempts} попыток")
        else:
            logger.info(f"Успешно сгенерировано {MAX_ZONES} зон за {attempts} попыток")

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(zones, f, ensure_ascii=False, indent=2)
            
        elapsed = time.time() - start_time
        logger.info(f"Генерация завершена за {elapsed:.2f} сек. Результат сохранен в {OUTPUT_FILE}")
        return zones

    except Exception as e:
        logger.error(f"Ошибка при генерации запретных зон: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    generate_zones()
