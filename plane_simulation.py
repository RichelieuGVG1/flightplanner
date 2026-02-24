import json
import math
import random
import heapq
from typing import List, Dict, Tuple, Optional

# ── Аэропорты ────────────────────────────────────────────────────────────────
AIRPORTS = [
    {"name": "Sheremetyevo", "coords": [55.9726, 37.4146]},
    {"name": "Pulkovo",      "coords": [59.8003, 30.2625]},
    {"name": "Tolmachevo",   "coords": [55.0126, 82.6507]},
    {"name": "Sochi",        "coords": [43.4499, 39.9566]},
    {"name": "Kazan",        "coords": [55.6062, 49.2787]},
    {"name": "Knevichi",     "coords": [43.3990, 132.1480]},
    {"name": "Novy",         "coords": [48.5280, 135.1880]},
    {"name": "Yelizovo",     "coords": [53.1679, 158.4539]},
]

WAYPOINTS_FILE = "russia_waypoints.json"
CORRIDOR_KM    = 800   # макс. поперечное отклонение waypoint от ортодромии
pos_num        = 50


# ── Геодезия ──────────────────────────────────────────────────────────────────
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing(lat1, lon1, lat2, lon2) -> float:
    """Азимут от точки 1 к точке 2 (радианы)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
    return math.atan2(x, y)


def cross_track_distance(lat: float, lon: float,
                          lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
    """Поперечное расстояние точки от ортодромии lat1→lat2 (км)."""
    R = 6371.0
    d13 = haversine(lat1, lon1, lat, lon) / R
    t13 = _bearing(lat1, lon1, lat, lon)
    t12 = _bearing(lat1, lon1, lat2, lon2)
    return abs(math.asin(math.sin(d13) * math.sin(t13 - t12)) * R)


def along_track_fraction(lat: float, lon: float,
                          lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
    """Продольная доля [0..1] проекции точки на отрезок lat1→lat2."""
    R = 6371.0
    d13 = haversine(lat1, lon1, lat, lon) / R
    d12 = haversine(lat1, lon1, lat2, lon2) / R
    t13 = _bearing(lat1, lon1, lat, lon)
    t12 = _bearing(lat1, lon1, lat2, lon2)
    sin_xtd = math.sin(d13) * math.sin(t13 - t12)
    cos_xtd = max(math.cos(math.asin(sin_xtd)), 1e-10)
    atd = math.acos(min(math.cos(d13) / cos_xtd, 1.0)) * R
    return atd / (d12 * R) if d12 > 1e-6 else 0.0


# ── Загрузка waypoints ────────────────────────────────────────────────────────
def load_waypoints(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ("waypoints", "data", "points"):
        if key in data:
            return data[key]
    raise ValueError("Не удалось распознать структуру файла waypoints")


# ── Фильтрация коридором ──────────────────────────────────────────────────────
def filter_corridor(waypoints: List[Dict],
                    dep_lat: float, dep_lon: float,
                    arr_lat: float, arr_lon: float,
                    corridor_km: float = CORRIDOR_KM) -> List[Dict]:
    """
    Оставляет waypoints, которые:
      - не дальше corridor_km от ортодромии (поперечное расстояние)
      - проецируются между стартом и финишем (продольная доля -0.05..1.05)
    """
    result = []
    for wp in waypoints:
        xtd = cross_track_distance(wp["lat"], wp["lon"],
                                   dep_lat, dep_lon, arr_lat, arr_lon)
        if xtd > corridor_km:
            continue
        frac = along_track_fraction(wp["lat"], wp["lon"],
                                    dep_lat, dep_lon, arr_lat, arr_lon)
        if -0.05 <= frac <= 1.05:
            result.append(wp)
    return result


# ── Ближайший waypoint ────────────────────────────────────────────────────────
def nearest_waypoint_index(waypoints: List[Dict], lat: float, lon: float) -> int:
    best_i, best_d = 0, float("inf")
    for i, wp in enumerate(waypoints):
        d = haversine(lat, lon, wp["lat"], wp["lon"])
        if d < best_d:
            best_d, best_i = d, i
    return best_i


# ── Граф K ближайших соседей ─────────────────────────────────────────────────
def build_graph(waypoints: List[Dict], k: int = 8) -> Dict[int, List[Tuple[float, int]]]:
    n = len(waypoints)
    graph: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(n)}
    for i in range(n):
        dists = sorted(
            (haversine(waypoints[i]["lat"], waypoints[i]["lon"],
                       waypoints[j]["lat"], waypoints[j]["lon"]), j)
            for j in range(n) if j != i
        )
        graph[i] = dists[:k]
    return graph


# ── A* ────────────────────────────────────────────────────────────────────────
def astar(graph: Dict[int, List[Tuple[float, int]]],
          waypoints: List[Dict],
          start: int, goal: int) -> Optional[List[int]]:
    heap: List[Tuple[float, int, List[int]]] = [(0.0, start, [start])]
    visited: Dict[int, float] = {}
    glat, glon = waypoints[goal]["lat"], waypoints[goal]["lon"]

    while heap:
        cost, node, path = heapq.heappop(heap)
        if node == goal:
            return path
        if visited.get(node, float("inf")) <= cost:
            continue
        visited[node] = cost
        for edge_d, nb in graph[node]:
            h = haversine(waypoints[nb]["lat"], waypoints[nb]["lon"], glat, glon)
            heapq.heappush(heap, (cost + edge_d + h, nb, path + [nb]))
    return None


# ── Аппроксимация маршрута к N точкам ────────────────────────────────────────
def approximate_route(route_coords: List[Tuple[float, float]],
                      n: int = pos_num) -> List[Tuple[float, float]]:
    if len(route_coords) <= n:
        return list(route_coords)
    cum = [0.0]
    for i in range(1, len(route_coords)):
        cum.append(cum[-1] + haversine(*route_coords[i-1], *route_coords[i]))
    total = cum[-1]
    result = []
    for k in range(n):
        target = total * k / (n - 1)
        for i in range(1, len(cum)):
            if cum[i] >= target or i == len(cum) - 1:
                seg = cum[i] - cum[i-1]
                if seg == 0:
                    lat, lon = route_coords[i]
                else:
                    t = (target - cum[i-1]) / seg
                    lat = route_coords[i-1][0] + t*(route_coords[i][0] - route_coords[i-1][0])
                    lon = route_coords[i-1][1] + t*(route_coords[i][1] - route_coords[i-1][1])
                result.append((round(lat, 6), round(lon, 6)))
                break
    return result


# ── Основная логика ───────────────────────────────────────────────────────────
def simulate(waypoints_path: str = WAYPOINTS_FILE,
             k_neighbors: int = 8,
             corridor_km: float = CORRIDOR_KM) -> Tuple[Dict, List[Dict]]:

    print("Загрузка waypoints...")
    all_waypoints = load_waypoints(waypoints_path)
    print(f"  Загружено {len(all_waypoints)} точек")

    dep_ap, arr_ap = random.sample(AIRPORTS, 2)
    dep_lat, dep_lon = dep_ap["coords"]
    arr_lat, arr_lon = arr_ap["coords"]
    gc_dist = haversine(dep_lat, dep_lon, arr_lat, arr_lon)

    print(f"\nМаршрут: {dep_ap['name']} → {arr_ap['name']}")
    print(f"  Ортодромия: {gc_dist:.0f} км")

    # ── Фильтрация коридором ───
    corridor_wps = filter_corridor(all_waypoints,
                                   dep_lat, dep_lon,
                                   arr_lat, arr_lon,
                                   corridor_km)
    print(f"  Waypoints в коридоре ±{corridor_km} км: "
          f"{len(corridor_wps)} из {len(all_waypoints)}")

    if len(corridor_wps) < 2:
        raise RuntimeError(
            "Слишком мало waypoints в коридоре — "
            "увеличьте corridor_km или проверьте данные"
        )

    start_idx = nearest_waypoint_index(corridor_wps, dep_lat, dep_lon)
    goal_idx  = nearest_waypoint_index(corridor_wps, arr_lat, arr_lon)

    print(f"  WP старта : [{start_idx}] {corridor_wps[start_idx]['name']} "
          f"({corridor_wps[start_idx]['lat']:.4f}, {corridor_wps[start_idx]['lon']:.4f})")
    print(f"  WP финиша : [{goal_idx}]  {corridor_wps[goal_idx]['name']} "
          f"({corridor_wps[goal_idx]['lat']:.4f}, {corridor_wps[goal_idx]['lon']:.4f})")

    print(f"\nПостроение графа (k={k_neighbors}, {len(corridor_wps)} узлов)...")
    graph = build_graph(corridor_wps, k=k_neighbors)

    print("Поиск кратчайшего пути (A*)...")
    path_indices = astar(graph, corridor_wps, start_idx, goal_idx)

    if path_indices is None:
        raise RuntimeError(
            "Путь не найден — попробуйте увеличить k_neighbors или corridor_km"
        )
    print(f"  Найден путь из {len(path_indices)} waypoints")

    route_waypoints = [
        {"name": corridor_wps[i]["name"],
         "lat":  corridor_wps[i]["lat"],
         "lon":  corridor_wps[i]["lon"]}
        for i in path_indices
    ]

    route_coords: List[Tuple[float, float]] = (
        [(dep_lat, dep_lon)]
        + [(wp["lat"], wp["lon"]) for wp in route_waypoints]
        + [(arr_lat, arr_lon)]
    )

    # Контрольная метрика
    max_dev = max(
        (cross_track_distance(wp["lat"], wp["lon"],
                              dep_lat, dep_lon, arr_lat, arr_lon)
         for wp in route_waypoints),
        default=0.0
    )
    print(f"  Макс. отклонение от ортодромии: {max_dev:.0f} км")

    approx = approximate_route(route_coords, n=pos_num)
    approx_list = [{"lat": lat, "lon": lon} for lat, lon in approx]

    result = {
        "departure":        {"name": dep_ap["name"], "lat": dep_lat, "lon": dep_lon},
        "arrival":          {"name": arr_ap["name"], "lat": arr_lat, "lon": arr_lon},
        "gc_distance_km":   round(gc_dist, 1),
        "max_deviation_km": round(max_dev, 1),
        "corridor_km":      corridor_km,
        "route_waypoints":  route_waypoints,
        "approximated_20":  approx_list,
    }

    with open("simulation_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\nРезультат сохранён: simulation_result.json")

    print(f"\n=== АППРОКСИМИРОВАННЫЕ {pos_num} ПОЗИЦИЙ САМОЛЁТА ===")
    for i, pt in enumerate(approx_list):
        print(f"  [{i+1:2d}]  lat={pt['lat']:.5f}  lon={pt['lon']:.5f}")

    return result, all_waypoints


# ── Визуализация Plotly ───────────────────────────────────────────────────────
def draw_map(result: Dict, all_waypoints: List[Dict]) -> None:
    import plotly.graph_objects as go

    dep       = result["departure"]
    arr       = result["arrival"]
    route_wps = result["route_waypoints"]
    approx_20 = result["approximated_20"]

    fig = go.Figure()

    # Все waypoints — серый фон
    fig.add_trace(go.Scattergeo(
        lat=[wp["lat"] for wp in all_waypoints],
        lon=[wp["lon"] for wp in all_waypoints],
        mode="markers",
        marker=dict(size=3, color="#94a3b8", opacity=0.35),
        name="Все waypoints",
        hovertemplate="%{text}<br>(%{lat:.4f}, %{lon:.4f})<extra></extra>",
        text=[wp["name"] for wp in all_waypoints],
    ))

    # Ортодромия — тонкая пунктирная серая линия
    fig.add_trace(go.Scattergeo(
        lat=[dep["lat"], arr["lat"]],
        lon=[dep["lon"], arr["lon"]],
        mode="lines",
        line=dict(width=1.2, color="#cbd5e1", dash="dash"),
        name="Ортодромия",
        hoverinfo="skip",
    ))

    # Маршрут по waypoints
    route_lats = [dep["lat"]] + [wp["lat"] for wp in route_wps] + [arr["lat"]]
    route_lons = [dep["lon"]] + [wp["lon"] for wp in route_wps] + [arr["lon"]]
    fig.add_trace(go.Scattergeo(
        lat=route_lats, lon=route_lons,
        mode="lines",
        line=dict(width=1.5, color="#64748b", dash="dot"),
        name="Маршрут по WP",
        hoverinfo="skip",
    ))

    # Waypoints маршрута — синие
    fig.add_trace(go.Scattergeo(
        lat=[wp["lat"] for wp in route_wps],
        lon=[wp["lon"] for wp in route_wps],
        mode="markers",
        marker=dict(size=6, color="#3b82f6", symbol="circle",
                    line=dict(width=1, color="#1d4ed8")),
        name="Waypoints маршрута",
        hovertemplate="%{text}<br>(%{lat:.4f}, %{lon:.4f})<extra></extra>",
        text=[wp["name"] for wp in route_wps],
    ))

    # Оранжевая линия аппроксимации
    fig.add_trace(go.Scattergeo(
        lat=[p["lat"] for p in approx_20],
        lon=[p["lon"] for p in approx_20],
        mode="lines",
        line=dict(width=3, color="#f97316"),
        name=f"Траектория ({pos_num} поз.)",
        hoverinfo="skip",
    ))

    # Позиции самолёта — оранжевые ромбы
    fig.add_trace(go.Scattergeo(
        lat=[p["lat"] for p in approx_20],
        lon=[p["lon"] for p in approx_20],
        mode="markers+text",
        marker=dict(size=8, color="#f97316", symbol="diamond",
                    line=dict(width=1, color="#c2410c")),
        text=[str(i+1) for i in range(len(approx_20))],
        textposition="top right",
        textfont=dict(size=8, color="#c2410c"),
        name=f"Позиции ({pos_num})",
        hovertemplate="Поз. %{text}<br>(%{lat:.5f}, %{lon:.5f})<extra></extra>",
    ))

    # Аэропорт вылета
    fig.add_trace(go.Scattergeo(
        lat=[dep["lat"]], lon=[dep["lon"]],
        mode="markers+text",
        marker=dict(size=16, color="#16a34a", symbol="star",
                    line=dict(width=1.5, color="#14532d")),
        text=[f"  {dep['name']}"],
        textposition="middle right",
        textfont=dict(size=12, color="#15803d", family="Inter, sans-serif"),
        name=f"Вылет: {dep['name']}",
        hovertemplate=f"<b>Вылет</b>: {dep['name']}<br>({dep['lat']:.4f}, {dep['lon']:.4f})<extra></extra>",
    ))

    # Аэропорт прилёта
    fig.add_trace(go.Scattergeo(
        lat=[arr["lat"]], lon=[arr["lon"]],
        mode="markers+text",
        marker=dict(size=16, color="#dc2626", symbol="star",
                    line=dict(width=1.5, color="#7f1d1d")),
        text=[f"  {arr['name']}"],
        textposition="middle right",
        textfont=dict(size=12, color="#b91c1c", family="Inter, sans-serif"),
        name=f"Прилёт: {arr['name']}",
        hovertemplate=f"<b>Прилёт</b>: {arr['name']}<br>({arr['lat']:.4f}, {arr['lon']:.4f})<extra></extra>",
    ))

    all_lats = [dep["lat"], arr["lat"]] + [wp["lat"] for wp in route_wps]
    all_lons = [dep["lon"], arr["lon"]] + [wp["lon"] for wp in route_wps]
    pad_lat  = max((max(all_lats) - min(all_lats)) * 0.3, 5)
    pad_lon  = max((max(all_lons) - min(all_lons)) * 0.3, 5)

    fig.update_layout(
        title=dict(
            text=(f"✈  {dep['name']}  →  {arr['name']}   "
                  f"<span style='font-size:13px;color:#64748b'>"
                  f"ортодромия {result['gc_distance_km']} км · "
                  f"макс. откл. {result['max_deviation_km']} км</span>"),
            font=dict(size=17, family="Inter, sans-serif", color="#0f172a"),
            x=0.5,
        ),
        geo=dict(
            projection_type="natural earth",
            showland=True,       landcolor="#f8fafc",
            showocean=True,      oceancolor="#dbeafe",
            showlakes=True,      lakecolor="#bfdbfe",
            showcountries=True,  countrycolor="#cbd5e1", countrywidth=0.7,
            showcoastlines=True, coastlinecolor="#94a3b8", coastlinewidth=0.8,
            showrivers=False,    bgcolor="#f1f5f9",
            lataxis=dict(range=[min(all_lats)-pad_lat, max(all_lats)+pad_lat]),
            lonaxis=dict(range=[min(all_lons)-pad_lon, max(all_lons)+pad_lon]),
            center=dict(lat=(max(all_lats)+min(all_lats))/2,
                        lon=(max(all_lons)+min(all_lons))/2),
        ),
        paper_bgcolor="#ffffff",
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)", bordercolor="#e2e8f0", borderwidth=1,
            font=dict(size=11, family="Inter, sans-serif"),
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=750,
    )

    fig.write_html("route_map.html")
    print("\nКарта сохранена: route_map.html")
    fig.show()


# ── Точка входа ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result, all_waypoints = simulate()
    draw_map(result, all_waypoints)