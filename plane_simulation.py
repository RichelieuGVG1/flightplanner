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


# ── Геодезия ─────────────────────────────────────────────────────────────────
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между двумя точками на сфере (км)."""
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Загрузка waypoints ────────────────────────────────────────────────────────
def load_waypoints(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Поддерживаем как список, так и обёртку {"waypoints": [...]}
    if isinstance(data, list):
        return data
    for key in ("waypoints", "data", "points"):
        if key in data:
            return data[key]
    raise ValueError("Не удалось распознать структуру файла waypoints")


# ── Поиск ближайшего waypoint к аэропорту ────────────────────────────────────
def nearest_waypoint_index(waypoints: List[Dict], lat: float, lon: float) -> int:
    best_i, best_d = 0, float("inf")
    for i, wp in enumerate(waypoints):
        d = haversine(lat, lon, wp["lat"], wp["lon"])
        if d < best_d:
            best_d, best_i = d, i
    return best_i


# ── Построение графа соседей (K ближайших) ────────────────────────────────────
def build_graph(waypoints: List[Dict], k: int = 8) -> Dict[int, List[Tuple[float, int]]]:
    """Для каждого узла — k ближайших соседей (dist, idx)."""
    n = len(waypoints)
    graph: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(n)}
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            d = haversine(waypoints[i]["lat"], waypoints[i]["lon"],
                          waypoints[j]["lat"], waypoints[j]["lon"])
            dists.append((d, j))
        dists.sort()
        graph[i] = dists[:k]
    return graph


# ── A* по графу ──────────────────────────────────────────────────────────────
def astar(graph: Dict[int, List[Tuple[float, int]]],
          waypoints: List[Dict],
          start: int, goal: int) -> Optional[List[int]]:
    open_heap: List[Tuple[float, int, List[int]]] = []
    heapq.heappush(open_heap, (0.0, start, [start]))
    visited = {}

    goal_lat = waypoints[goal]["lat"]
    goal_lon = waypoints[goal]["lon"]

    while open_heap:
        cost, node, path = heapq.heappop(open_heap)

        if node == goal:
            return path

        if node in visited and visited[node] <= cost:
            continue
        visited[node] = cost

        for edge_dist, neighbor in graph[node]:
            new_cost = cost + edge_dist
            h = haversine(waypoints[neighbor]["lat"], waypoints[neighbor]["lon"],
                          goal_lat, goal_lon)
            heapq.heappush(open_heap, (new_cost + h, neighbor, path + [neighbor]))

    return None


# ── Аппроксимация маршрута к N точкам ────────────────────────────────────────
def approximate_route(route_coords: List[Tuple[float, float]], n: int = 20) -> List[Tuple[float, float]]:
    """
    Равномерно выбирает n точек вдоль маршрута по накопленному расстоянию.
    """
    if len(route_coords) <= n:
        return list(route_coords)

    # Накопленные расстояния
    cum = [0.0]
    for i in range(1, len(route_coords)):
        d = haversine(*route_coords[i - 1], *route_coords[i])
        cum.append(cum[-1] + d)
    total = cum[-1]

    result = []
    for k in range(n):
        target = total * k / (n - 1)
        # Найти сегмент
        for i in range(1, len(cum)):
            if cum[i] >= target or i == len(cum) - 1:
                seg_len = cum[i] - cum[i - 1]
                if seg_len == 0:
                    lat, lon = route_coords[i]
                else:
                    t = (target - cum[i - 1]) / seg_len
                    lat = route_coords[i - 1][0] + t * (route_coords[i][0] - route_coords[i - 1][0])
                    lon = route_coords[i - 1][1] + t * (route_coords[i][1] - route_coords[i - 1][1])
                result.append((round(lat, 6), round(lon, 6)))
                break
    return result


# ── Основная логика ───────────────────────────────────────────────────────────
def simulate(waypoints_path: str = WAYPOINTS_FILE, k_neighbors: int = 8) -> Dict:
    print("Загрузка waypoints...")
    waypoints = load_waypoints(waypoints_path)
    print(f"  Загружено {len(waypoints)} точек")

    # Случайный выбор аэропортов (разные)
    dep_ap, arr_ap = random.sample(AIRPORTS, 2)
    print(f"\nМаршрут: {dep_ap['name']} -> {arr_ap['name']}")

    # Ближайшие waypoints к аэропортам
    dep_lat, dep_lon = dep_ap["coords"]
    arr_lat, arr_lon = arr_ap["coords"]
    start_idx = nearest_waypoint_index(waypoints, dep_lat, dep_lon)
    goal_idx  = nearest_waypoint_index(waypoints, arr_lat, arr_lon)

    print(f"  Ближайший WP к вылету:  [{start_idx}] {waypoints[start_idx]['name']} "
          f"({waypoints[start_idx]['lat']:.4f}, {waypoints[start_idx]['lon']:.4f})")
    print(f"  Ближайший WP к прилёту: [{goal_idx}] {waypoints[goal_idx]['name']} "
          f"({waypoints[goal_idx]['lat']:.4f}, {waypoints[goal_idx]['lon']:.4f})")

    # Построение графа
    print(f"\nПостроение графа (k={k_neighbors})...")
    graph = build_graph(waypoints, k=k_neighbors)

    # A* поиск
    print("Поиск кратчайшего пути (A*)...")
    path_indices = astar(graph, waypoints, start_idx, goal_idx)

    if path_indices is None:
        raise RuntimeError("Путь не найден — попробуйте увеличить k_neighbors")

    print(f"  Найден путь из {len(path_indices)} waypoints")

    # Координаты маршрута (включая сами аэропорты на концах)
    route_coords: List[Tuple[float, float]] = (
        [(dep_lat, dep_lon)]
        + [(waypoints[i]["lat"], waypoints[i]["lon"]) for i in path_indices]
        + [(arr_lat, arr_lon)]
    )

    # Waypoints маршрута (без аэропортов — только найденные WP)
    route_waypoints = [
        {"name": waypoints[i]["name"], "lat": waypoints[i]["lat"], "lon": waypoints[i]["lon"]}
        for i in path_indices
    ]

    # Аппроксимация к 20 позициям
    approx_20 = approximate_route(route_coords, n=20)
    approx_list = [{"lat": lat, "lon": lon} for lat, lon in approx_20]

    result = {
        "departure": {"name": dep_ap["name"], "lat": dep_lat, "lon": dep_lon},
        "arrival":   {"name": arr_ap["name"], "lat": arr_lat, "lon": arr_lon},
        "route_waypoints": route_waypoints,
        "approximated_20": approx_list,
    }

    # Сохранение
    out_path = "simulation_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nРезультат сохранён: {out_path}")

    # Краткий вывод
    print("\n=== АППРОКСИМИРОВАННЫЕ 20 ПОЗИЦИЙ САМОЛЁТА ===")
    for i, pt in enumerate(approx_list):
        print(f"  [{i+1:2d}]  lat={pt['lat']:.5f}  lon={pt['lon']:.5f}")

    return result


# ── Визуализация Plotly ───────────────────────────────────────────────────────
def draw_map(result: Dict, all_waypoints: List[Dict]) -> None:
    import plotly.graph_objects as go

    dep      = result["departure"]
    arr      = result["arrival"]
    route_wps = result["route_waypoints"]
    approx_20 = result["approximated_20"]

    fig = go.Figure()

    # 1. Все waypoints из файла — серый фон
    fig.add_trace(go.Scattergeo(
        lat=[wp["lat"] for wp in all_waypoints],
        lon=[wp["lon"] for wp in all_waypoints],
        mode="markers",
        marker=dict(size=3, color="#94a3b8", opacity=0.45),
        name="Все waypoints",
        hovertemplate="%{text}<br>(%{lat:.4f}, %{lon:.4f})<extra></extra>",
        text=[wp["name"] for wp in all_waypoints],
    ))

    # 2. Пунктирная линия маршрута по waypoints
    route_lats = [dep["lat"]] + [wp["lat"] for wp in route_wps] + [arr["lat"]]
    route_lons = [dep["lon"]] + [wp["lon"] for wp in route_wps] + [arr["lon"]]
    fig.add_trace(go.Scattergeo(
        lat=route_lats,
        lon=route_lons,
        mode="lines",
        line=dict(width=1.2, color="#64748b", dash="dot"),
        name="Маршрут по WP",
        hoverinfo="skip",
    ))

    # 3. Waypoints маршрута — синие кружки
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

    # 4. Сплошная линия 20 аппроксимированных позиций
    fig.add_trace(go.Scattergeo(
        lat=[p["lat"] for p in approx_20],
        lon=[p["lon"] for p in approx_20],
        mode="lines",
        line=dict(width=3, color="#f97316"),
        name="Траектория (20 позиций)",
        hoverinfo="skip",
    ))

    # 5. Аппроксимированные 20 позиций самолёта — оранжевые ромбы с номерами
    fig.add_trace(go.Scattergeo(
        lat=[p["lat"] for p in approx_20],
        lon=[p["lon"] for p in approx_20],
        mode="markers+text",
        marker=dict(size=8, color="#f97316", symbol="diamond",
                    line=dict(width=1, color="#c2410c")),
        text=[str(i + 1) for i in range(len(approx_20))],
        textposition="top right",
        textfont=dict(size=8, color="#c2410c"),
        name="Позиции самолёта (20)",
        hovertemplate="Позиция %{text}<br>(%{lat:.5f}, %{lon:.5f})<extra></extra>",
    ))

    # 6. Аэропорт вылета — зелёная звезда
    fig.add_trace(go.Scattergeo(
        lat=[dep["lat"]],
        lon=[dep["lon"]],
        mode="markers+text",
        marker=dict(size=16, color="#16a34a", symbol="star",
                    line=dict(width=1.5, color="#14532d")),
        text=[f"  {dep['name']}"],
        textposition="middle right",
        textfont=dict(size=12, color="#15803d", family="Inter, sans-serif"),
        name=f"Вылет: {dep['name']}",
        hovertemplate=f"<b>Вылет</b>: {dep['name']}<br>({dep['lat']:.4f}, {dep['lon']:.4f})<extra></extra>",
    ))

    # 7. Аэропорт прилёта — красная звезда
    fig.add_trace(go.Scattergeo(
        lat=[arr["lat"]],
        lon=[arr["lon"]],
        mode="markers+text",
        marker=dict(size=16, color="#dc2626", symbol="star",
                    line=dict(width=1.5, color="#7f1d1d")),
        text=[f"  {arr['name']}"],
        textposition="middle right",
        textfont=dict(size=12, color="#b91c1c", family="Inter, sans-serif"),
        name=f"Прилёт: {arr['name']}",
        hovertemplate=f"<b>Прилёт</b>: {arr['name']}<br>({arr['lat']:.4f}, {arr['lon']:.4f})<extra></extra>",
    ))

    # Вычисляем bbox маршрута для автозума
    all_lats = [dep["lat"], arr["lat"]] + [wp["lat"] for wp in route_wps]
    all_lons = [dep["lon"], arr["lon"]] + [wp["lon"] for wp in route_wps]
    pad_lat = max((max(all_lats) - min(all_lats)) * 0.25, 5)
    pad_lon = max((max(all_lons) - min(all_lons)) * 0.25, 5)
    center_lat = (max(all_lats) + min(all_lats)) / 2
    center_lon = (max(all_lons) + min(all_lons)) / 2

    fig.update_layout(
        title=dict(
            text=f"✈  {dep['name']}  →  {arr['name']}",
            font=dict(size=18, family="Inter, sans-serif", color="#0f172a"),
            x=0.5,
        ),
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="#f8fafc",
            showocean=True,
            oceancolor="#dbeafe",
            showlakes=True,
            lakecolor="#bfdbfe",
            showcountries=True,
            countrycolor="#cbd5e1",
            countrywidth=0.7,
            showcoastlines=True,
            coastlinecolor="#94a3b8",
            coastlinewidth=0.8,
            showrivers=False,
            bgcolor="#f1f5f9",
            lataxis=dict(range=[min(all_lats) - pad_lat, max(all_lats) + pad_lat]),
            lonaxis=dict(range=[min(all_lons) - pad_lon, max(all_lons) + pad_lon]),
            center=dict(lat=center_lat, lon=center_lon),
        ),
        paper_bgcolor="#ffffff",
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=11, family="Inter, sans-serif"),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
        margin=dict(l=0, r=0, t=55, b=0),
        height=750,
    )

    out_html = "route_map.html"
    fig.write_html(out_html)
    print(f"\nКарта сохранена: {out_html}")
    fig.show()


# ── Точка входа ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    waypoints = load_waypoints(WAYPOINTS_FILE)
    result = simulate(waypoints_path=WAYPOINTS_FILE)
    draw_map(result, waypoints)