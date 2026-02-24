"""
Генерирует погодные данные для навигационных точек и строит интерактивную карту.
"""

import json, math, random
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

WAYPOINTS_FILE = "russia_waypoints.json"
OUTPUT_FILE    = "weather_data.json"

N_TIMES  = 100
N_LEVELS = 5
N_FRONTS = 4

LAT_MIN, LAT_MAX = 41.0, 77.0
LON_MIN, LON_MAX = 26.0, 170.0

SEED = 42
rng  = np.random.default_rng(SEED)
random.seed(SEED)


# ── Шум Перлина ───────────────────────────────────────────────────────────────
def _fade(t): return t*t*t*(t*(t*6-15)+10)
def _lerp(a,b,t): return a+t*(b-a)

class PerlinNoise2D:
    def __init__(self, size=16):
        self.size = size
        angles = rng.uniform(0, 2*math.pi, (size+1, size+1))
        self.gx = np.cos(angles)
        self.gy = np.sin(angles)

    def sample(self, x, y):
        ix, iy = int(x) % self.size, int(y) % self.size
        fx, fy = x-int(x), y-int(y)
        u, v = _fade(fx), _fade(fy)
        def dot(ddx, ddy):
            cx = (ix+ddx) % (self.size+1)
            cy = (iy+ddy) % (self.size+1)
            return self.gx[cy,cx]*(fx-ddx) + self.gy[cy,cx]*(fy-ddy)
        return _lerp(_lerp(dot(0,0),dot(1,0),u), _lerp(dot(0,1),dot(1,1),u), v)


# ── Поле ветра ────────────────────────────────────────────────────────────────
class WindField:
    """
    Реалистичное поле ветра:
      - Ослабленный западный перенос (фон), не доминирует
      - 12 региональных центров давления (циклоны/антициклоны) с разными
        знаками и позициями по всей территории — создают разнонаправленные потоки
      - Мезомасштабные центры (малый радиус) для локальных завихрений
      - Два независимых шума Перлина (u и v) с разными дрейфами
      - Все центры дрейфуют и пульсируют с разными частотами
    """
    def __init__(self):
        # ── Крупные центры давления (синоптический масштаб) ──
        self.n_large = 12
        # Равномерно покрываем всю территорию России
        self.clat0 = np.array([
            47, 52, 58, 65, 72,   # широтный пояс 1
            49, 55, 62, 68,       # пояс 2
            50, 60, 70            # пояс 3
        ], dtype=float)
        self.clon0 = np.array([
            40, 80, 130, 60, 100,
            50, 110, 90, 140,
            155, 35, 70
        ], dtype=float)
        self.cstr  = rng.uniform(1.2, 3.5, self.n_large)
        # Знаки чередуются случайно, но с достаточным количеством обоих
        self.csign = np.array([1,-1,1,-1,-1,1,-1,1,1,-1,1,-1], dtype=float)
        self.dlat  = rng.uniform(-0.15, 0.15, self.n_large)
        self.dlon  = rng.uniform(-0.10, 0.20, self.n_large)
        self.phase = rng.uniform(0, 2*math.pi, self.n_large)
        self.pfreq = rng.uniform(0.03, 0.09, self.n_large)

        # ── Мезомасштабные центры (локальные завихрения) ──
        self.n_meso = 8
        self.mlat0 = rng.uniform(45, 73, self.n_meso)
        self.mlon0 = rng.uniform(30, 165, self.n_meso)
        self.mstr  = rng.uniform(0.8, 2.0, self.n_meso)
        self.msign = rng.choice([-1,1], self.n_meso).astype(float)
        self.mdlat = rng.uniform(-0.20, 0.20, self.n_meso)
        self.mdlon = rng.uniform(-0.15, 0.25, self.n_meso)
        self.mphase= rng.uniform(0, 2*math.pi, self.n_meso)
        self.mfreq = rng.uniform(0.06, 0.14, self.n_meso)

        # ── Два независимых шума Перлина ──
        self.pu = PerlinNoise2D(14)
        self.pv = PerlinNoise2D(14)
        # u и v дрейфуют в РАЗНЫХ направлениях → поле вращается
        self.pu_dx = rng.uniform( 0.02, 0.06)
        self.pu_dy = rng.uniform(-0.03, 0.03)
        self.pv_dx = rng.uniform(-0.04, 0.01)
        self.pv_dy = rng.uniform( 0.01, 0.05)

    def _px(self, lat, lon): return (lon-LON_MIN)/(LON_MAX-LON_MIN)*14
    def _py(self, lat, lon): return (lat-LAT_MIN)/(LAT_MAX-LAT_MIN)*14

    def uv(self, lat, lon, level, t):
        # ── Нет доминирующего западного переноса.
        # Слабый фон пульсирует около нуля — иногда западный, иногда нет.
        bg_u =  3.0*math.sin(t*0.06 + lat*0.04)
        bg_v =  3.0*math.cos(t*0.08 + lon*0.02)
        u, v = bg_u, bg_v

        # ── Крупные центры давления ──
        # Каждый создаёт вращение вокруг себя: циклон — против часовой,
        # антициклон — по часовой. Вместе 12 центров покрывают всю территорию
        # и дают принципиально разные направления в разных регионах.
        for i in range(self.n_large):
            clat = self.clat0[i] + self.dlat[i]*t
            clon = self.clon0[i] + self.dlon[i]*t
            # Сила пульсирует, иногда меняет знак → центр «переключается»
            strength = self.cstr[i] * math.sin(self.phase[i] + self.pfreq[i]*t)
            dlat_ = lat - clat
            dlon_ = lon - clon
            dist2 = dlat_**2 + dlon_**2 + 0.5
            # Вращательное поле: перпендикуляр к радиусу
            # u ∝ -dlat (север-юг), v ∝ dlon (восток-запад)
            f = strength * 14.0 / dist2
            u += (-dlat_) * f
            v += ( dlon_) * f

        # ── Мезомасштабные центры ──
        for i in range(self.n_meso):
            clat = self.mlat0[i] + self.mdlat[i]*t
            clon = self.mlon0[i] + self.mdlon[i]*t
            strength = self.mstr[i] * math.sin(self.mphase[i] + self.mfreq[i]*t)
            dlat_ = lat - clat
            dlon_ = lon - clon
            dist2 = dlat_**2 + dlon_**2 + 0.2
            f = strength * 7.0 / dist2
            u += (-dlat_) * f
            v += ( dlon_) * f

        # ── Два независимых шума Перлина, оба полноценные ──
        # u и v дрейфуют в разные стороны → поле медленно «вращается»
        px = self._px(lat, lon)
        py = self._py(lat, lon)
        pxu = (px + self.pu_dx*t) % 14
        pyu = (py + self.pu_dy*t) % 14
        pxv = (px + self.pv_dx*t) % 14
        pyv = (py + self.pv_dy*t) % 14
        # Амплитуды u и v равны и пульсируют независимо
        amp_u = 10.0 + 5.0*math.sin(t*0.05 + 0.3)
        amp_v = 10.0 + 5.0*math.cos(t*0.07 + 2.0)
        u += self.pu.sample(pxu, pyu) * amp_u
        v += self.pv.sample(pxv, pyv) * amp_v

        # ── Масштаб по эшелону ──
        scale = 1.0 + (level-1)*0.28
        return u*scale, v*scale


# ── Грозовые фронты ───────────────────────────────────────────────────────────
class StormFront:
    """
    Фронт движется вместе с ветром (скорость и направление берутся из WindField
    в центре фронта). Форма (полуоси) медленно деформируется.
    """
    def __init__(self, fid, wind_field: WindField):
        self.wf          = wind_field
        self.lat0        = rng.uniform(50, 67)
        self.lon0        = rng.uniform(45, 125)
        self.semi_lat0   = rng.uniform(3.5, 8.0)
        self.semi_lon0   = rng.uniform(7.0, 16.0)
        self.max_power   = rng.uniform(2.5, 5.0)
        self.rotation0   = rng.uniform(-30, 30)
        self.base_level  = rng.integers(2, 5)   # пик на этом эшелоне
        # Деформация формы: медленная синусоидальная
        self.deform_freq  = rng.uniform(0.03, 0.07)
        self.deform_phase = rng.uniform(0, 2*math.pi)
        self.deform_amp   = rng.uniform(0.5, 2.0)

        # Предрасчитываем траекторию центра через интегрирование ветра
        self._traj = self._precompute_trajectory()

    def _precompute_trajectory(self):
        """
        Интегрируем траекторию: фронт несёт ветром, но с автономией.
        Автономия = случайное блуждание (Браун) + собственная инерция направления.
        Это даёт отклонение от ветрового дрейфа на 20–40%.
        """
        lats, lons = [self.lat0], [self.lon0]
        lat, lon = self.lat0, self.lon0
        R = 6371.0
        # DT_H = 2 часа на шаг — фронт заметно смещается за 100 шагов
        # При скорости ветра ~15 м/с → ~108 км/шаг → ~10800 км за 100 шагов
        # Это реалистично для атмосферных фронтов за ~8 суток
        DT_H = 2.0

        # Собственная инерция фронта (отклонение от ветра)
        own_u = float(rng.uniform(-4.0, 4.0))
        own_v = float(rng.uniform(-4.0, 4.0))

        for t in range(1, N_TIMES+1):
            u_wind, v_wind = self.wf.uv(lat, lon, self.base_level, t)

            # Инерция: случайное блуждание, слабо меняется
            own_u += float(rng.uniform(-0.6, 0.6))
            own_v += float(rng.uniform(-0.6, 0.6))
            own_u = float(np.clip(own_u, -6.0, 6.0))
            own_v = float(np.clip(own_v, -6.0, 6.0))

            # Результат: 75% ветер + 25% собственная инерция
            # Скорость фронта чуть медленнее ветра (коэффициент 0.75)
            u_eff = (0.75*u_wind + 0.25*own_u) * 0.75
            v_eff = (0.75*v_wind + 0.25*own_v) * 0.75

            dlat = (v_eff * DT_H * 3600) / (R * 1000 / 57.296)
            dlon = (u_eff * DT_H * 3600) / (R * 1000 / 57.296 * max(math.cos(math.radians(lat)), 0.05))
            lat = max(LAT_MIN - 8, min(LAT_MAX + 8, lat + dlat))
            lon = lon + dlon
            if lon > 200: lon -= 360
            if lon < -20: lon += 360
            lats.append(lat)
            lons.append(lon)
        return list(zip(lats, lons))

    def center(self, t):
        return self._traj[min(t, len(self._traj)-1)]

    def semi_axes(self, t):
        """Деформация полуосей во времени."""
        d = self.deform_amp * math.sin(self.deform_phase + self.deform_freq*t)
        return self.semi_lat0 + d*0.4, self.semi_lon0 + d

    def rotation(self, t):
        return self.rotation0 + 8*math.sin(t*0.04)

    def travel_dir_speed(self, level, t):
        """Направление и скорость движения фронта = ветер в его центре на эшелоне."""
        clat, clon = self.center(t)
        u, v = self.wf.uv(clat, clon, level, t)
        spd = math.sqrt(u**2 + v**2) * 3.6   # м/с → км/ч
        bearing = (math.degrees(math.atan2(u, v)) + 360) % 360
        return round(bearing, 1), round(spd, 1)

    def power_at(self, lat, lon, level, t):
        clat, clon = self.center(t)
        sl, so = self.semi_axes(t)
        rot = math.radians(self.rotation(t))

        dlat = lat - clat
        dlon = (lon - clon) * math.cos(math.radians(clat))
        dr =  dlat*math.cos(rot) + dlon*math.sin(rot)
        dc = -dlat*math.sin(rot) + dlon*math.cos(rot)

        nd = math.sqrt((dr/sl)**2 + (dc/so)**2)
        if nd >= 1.0:
            return 0.0

        spatial = math.exp(-2.2*nd**2)
        lf = max(1.0 - 0.3*abs(level - self.base_level), 0.15)
        return self.max_power * spatial * lf


# ── Вспомогательные ───────────────────────────────────────────────────────────
def calc_turbulence(wind_spd, storm_power, level, rng_):
    base = 1.0 + min(wind_spd/18.0, 1.8) + storm_power*0.38 + (level-1)*0.08
    base += rng_.uniform(-0.25, 0.25)
    return int(min(max(round(base), 1), 4))

def calc_icing(level, storm_power, rng_):
    if level < 4:
        return 0
    prob = 0.03 + (level-4)*0.05 + storm_power*0.02
    return 1 if rng_.random() < min(prob, 0.15) else 0


# ── Генератор ─────────────────────────────────────────────────────────────────
def generate(waypoints_path=WAYPOINTS_FILE, output_path=OUTPUT_FILE):
    print("Загрузка waypoints...")
    raw = json.loads(Path(waypoints_path).read_text(encoding="utf-8"))
    wps = raw if isinstance(raw, list) else raw.get("waypoints", raw)
    print(f"  {len(wps)} точек")

    wf     = WindField()
    fronts = [StormFront(i, wf) for i in range(N_FRONTS)]
    print(f"  {N_FRONTS} фронтов · {N_LEVELS} эшелонов · {N_TIMES} шагов")

    records = []
    total = len(wps)*N_LEVELS*N_TIMES
    done  = 0

    for wp in wps:
        lat, lon = wp["lat"], wp["lon"]
        for z in range(1, N_LEVELS+1):
            for t in range(1, N_TIMES+1):
                u, v  = wf.uv(lat, lon, z, t)
                wsp   = math.sqrt(u**2+v**2)
                wdir  = (math.degrees(math.atan2(u, v))+360) % 360

                sp_total = 0.0
                dom_fr, dom_pw = None, 0.0
                for fr in fronts:
                    pw = fr.power_at(lat, lon, z, t)
                    sp_total += pw
                    if pw > dom_pw:
                        dom_pw, dom_fr = pw, fr

                sp = min(round(sp_total, 2), 5.0)
                if dom_fr and dom_pw > 0.1:
                    sdir, sspd = dom_fr.travel_dir_speed(z, t)
                else:
                    sdir, sspd = 0.0, 0.0

                records.append({
                    "name": wp["name"],
                    "lat":  round(lat, 5),
                    "lon":  round(lon, 5),
                    "z": z, "t": t,
                    "wind_speed":         round(wsp, 2),
                    "wind_dir":           round(wdir, 1),
                    "storm_power":        sp,
                    "storm_dir":          sdir,
                    "storm_travel_speed": sspd,
                    "turbulence":         calc_turbulence(wsp, sp, z, rng),
                    "ice":                calc_icing(z, sp, rng),
                })
                done += 1
                if done % 50000 == 0:
                    print(f"  {done/total*100:.1f}%")

    Path(output_path).write_text(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")
    print(f"Сохранено {len(records)} записей → {output_path}")
    return records


# ── Визуализация (чистый HTML+JS, без Plotly frames) ─────────────────────────
def draw_weather(records: list, output_html="weather_map.html"):
    """
    Строит интерактивную карту через Plotly JS напрямую.
    Все данные предрасчитаны в Python и встроены в HTML как JSON.
    Переключение эшелона/времени/параметра — через JS без перерисовки фигуры целиком.
    """
    import plotly.graph_objects as go

    LEVEL_NAMES = {1:"FL100", 2:"FL200", 3:"FL300", 4:"FL350", 5:"FL400"}
    N_T = max(r["t"] for r in records)
    N_Z = max(r["z"] for r in records)

    # Индекс: data[z][t] = list of records
    idx: dict = {}
    for r in records:
        idx.setdefault(r["z"], {}).setdefault(r["t"], []).append(r)

    # Предрасчёт компактного payload для JS
    # Структура: payload[z][t] = {lats, lons, names, wsp, wdir, sp, sdir, turb, ice}
    payload = {}
    for z in range(1, N_Z+1):
        payload[z] = {}
        for t in range(1, N_T+1):
            rows = idx.get(z, {}).get(t, [])
            payload[z][t] = {
                "la": [r["lat"]         for r in rows],
                "lo": [r["lon"]         for r in rows],
                "nm": [r["name"]        for r in rows],
                "ws": [r["wind_speed"]  for r in rows],
                "wd": [r["wind_dir"]    for r in rows],
                "sp": [r["storm_power"] for r in rows],
                "sd": [r["storm_dir"]   for r in rows],
                "tb": [r["turbulence"]  for r in rows],
                "ic": [r["ice"]         for r in rows],
            }

    payload_json = json.dumps(payload, separators=(",",":"))

    # Базовая пустая фигура — карта
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lat=[], lon=[], mode="markers", name="data",
                                marker=dict(size=8)))
    fig.update_layout(
        geo=dict(
            scope="asia", projection_type="mercator",
            showland=True,       landcolor="#f1f5f9",
            showocean=True,      oceancolor="#dbeafe",
            showlakes=True,      lakecolor="#bae6fd",
            showcountries=True,  countrycolor="#cbd5e1", countrywidth=0.6,
            showcoastlines=True, coastlinecolor="#94a3b8", coastlinewidth=0.7,
            lonaxis=dict(range=[26,170]), lataxis=dict(range=[40,78]),
            bgcolor="#f8fafc",
        ),
        paper_bgcolor="#fff",
        margin=dict(l=0,r=0,t=0,b=0),
        height=600,
        showlegend=False,
    )

    plotly_div = fig.to_html(full_html=False, include_plotlyjs=False,
                              div_id="map-div", config={"responsive": True})

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Погода — интерактивная карта</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:'Inter',sans-serif}}
body{{background:#f8fafc;color:#0f172a;display:flex;flex-direction:column;height:100vh;overflow:hidden}}
#toolbar{{background:#fff;border-bottom:1px solid #e2e8f0;padding:10px 16px;display:flex;flex-direction:column;gap:8px;flex-shrink:0}}
.tb-row{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.tb-label{{font-size:11px;font-weight:700;text-transform:uppercase;color:#64748b;letter-spacing:.04em;min-width:64px}}
.btn-group{{display:flex;gap:4px;flex-wrap:wrap}}
button{{border:1px solid #e2e8f0;background:#f8fafc;color:#475569;padding:4px 12px;border-radius:6px;font-size:12px;font-weight:500;cursor:pointer;transition:all .15s}}
button:hover{{background:#e0f2fe;border-color:#7dd3fc}}
button.active{{background:#0284c7;color:#fff;border-color:#0284c7}}
.param-btn{{padding:4px 14px}}
.layer-cb{{display:flex;align-items:center;gap:5px;padding:4px 10px;border:1px solid #e2e8f0;border-radius:6px;background:#f8fafc;font-size:12px;font-weight:500;cursor:pointer;color:#475569;user-select:none}}
.layer-cb input{{accent-color:#0284c7;cursor:pointer;width:14px;height:14px}}
.layer-cb:has(input:checked){{background:#dbeafe;border-color:#7dd3fc;color:#0369a1}}
#slider-row{{display:flex;align-items:center;gap:10px}}
#t-slider{{flex:1;accent-color:#0284c7;height:4px}}
#t-label{{font-size:12px;font-weight:600;color:#0284c7;min-width:36px}}
#play-btn{{padding:4px 14px;background:#f0fdf4;border-color:#86efac;color:#15803d}}
#play-btn.playing{{background:#fef2f2;border-color:#fca5a5;color:#dc2626}}
#map-container{{flex:1;position:relative;overflow:hidden}}
#map-div{{width:100%;height:100%}}
#legend{{position:absolute;bottom:12px;right:12px;background:rgba(255,255,255,.93);border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;font-size:11px;min-width:140px;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
#legend h4{{font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:6px;letter-spacing:.04em}}
.legend-item{{display:flex;align-items:center;gap:7px;margin-bottom:4px}}
.legend-dot{{width:12px;height:12px;border-radius:50%;flex-shrink:0}}
.legend-sq{{width:12px;height:12px;border-radius:2px;flex-shrink:0}}
</style>
</head>
<body>

<div id="toolbar">
  <div class="tb-row">
    <span class="tb-label">Эшелон</span>
    <div class="btn-group" id="z-btns">
      <button onclick="setZ(1)" id="z1">FL100</button>
      <button onclick="setZ(2)" id="z2">FL200</button>
      <button onclick="setZ(3)" id="z3" class="active">FL300</button>
      <button onclick="setZ(4)" id="z4">FL350</button>
      <button onclick="setZ(5)" id="z5">FL400</button>
    </div>
  </div>
  <div class="tb-row">
    <span class="tb-label">Параметр</span>
    <div class="btn-group" id="layer-checks">
      <label class="layer-cb"><input type="checkbox" value="wind_speed" checked> Скорость ветра</label>
      <label class="layer-cb"><input type="checkbox" value="wind_dir"> Направление ветра</label>
      <label class="layer-cb"><input type="checkbox" value="storm"> Грозовой фронт</label>
      <label class="layer-cb"><input type="checkbox" value="turbulence"> Турбулентность</label>
      <label class="layer-cb"><input type="checkbox" value="ice"> Обледенение ❄</label>
    </div>
  </div>
  <div class="tb-row" id="slider-row">
    <span class="tb-label">Время</span>
    <input type="range" id="t-slider" min="1" max="{N_T}" value="1" oninput="setT(+this.value)">
    <span id="t-label">t = 1</span>
    <button id="play-btn" onclick="togglePlay()">▶ Play</button>
  </div>
</div>

<div id="map-container">
  {plotly_div}
  <div id="legend"></div>
</div>

<script>
const DATA = {payload_json};
const N_T  = {N_T};
const N_Z  = {N_Z};

let curZ = 3, curT = 1;
function getActiveLayers(){{
  return [...document.querySelectorAll('#layer-checks input:checked')].map(el=>el.value);
}}
let playTimer = null;

// ── Цветовые шкалы ────────────────────────────────────────────────────────────
function windColor(v) {{
  const stops=[[0,'#7dd3fc'],[10,'#4ade80'],[18,'#facc15'],[26,'#f97316'],[35,'#dc2626']];
  v=Math.max(0,Math.min(35,v));
  for(let i=1;i<stops.length;i++){{
    if(v<=stops[i][0]){{
      const t=(v-stops[i-1][0])/(stops[i][0]-stops[i-1][0]);
      return lerpColor(stops[i-1][1],stops[i][1],t);
    }}
  }}
  return stops[stops.length-1][1];
}}

function stormColor(v) {{
  // 0..5  →  бледно-голубой → тёмно-синий
  const stops=[[0,'#e0f2fe'],[1,'#93c5fd'],[2.5,'#6b7baa'],[4,'#3b5bbd'],[5,'#1e1b4b']];
  v=Math.max(0,Math.min(5,v));
  for(let i=1;i<stops.length;i++){{
    if(v<=stops[i][0]){{
      const t=(v-stops[i-1][0])/(stops[i][0]-stops[i-1][0]);
      return lerpColor(stops[i-1][1],stops[i][1],t);
    }}
  }}
  return stops[stops.length-1][1];
}}

function turbColor(lvl) {{
  // 1=белый, 2=светло-серый, 3=тёмно-серый, 4=чёрный
  return ['#f8fafc','#cbd5e1','#64748b','#0f172a'][lvl-1];
}}

function lerpColor(c1,c2,t){{
  const h=s=>parseInt(s.slice(1),16);
  const r1=h(c1)>>16,g1=(h(c1)>>8)&255,b1=h(c1)&255;
  const r2=h(c2)>>16,g2=(h(c2)>>8)&255,b2=h(c2)&255;
  const r=Math.round(r1+t*(r2-r1)),g=Math.round(g1+t*(g2-g1)),b=Math.round(b1+t*(b2-b1));
  return '#'+[r,g,b].map(x=>x.toString(16).padStart(2,'0')).join('');
}}

// ── Построение traces ─────────────────────────────────────────────────────────
function buildTraces(z, t, params){{
  const d = DATA[z][t];
  const lats=d.la, lons=d.lo, names=d.nm;
  const ws=d.ws, wd=d.wd, sp=d.sp, sd=d.sd, tb=d.tb, ic=d.ic;
  const n = lats.length;
  const traces = [];

  if(params.includes('wind_speed')){{
    traces.push({{
      type:'scattergeo', lat:lats, lon:lons, mode:'markers',
      marker:{{size:9, symbol:'square',
               color:ws.map(windColor), opacity:0.78}},
      text:names.map((nm,i)=>`${{nm}}<br>Ветер: ${{ws[i].toFixed(1)}} м/с`),
      hovertemplate:'<b>%{{text}}</b><extra></extra>',
      name:'Скорость ветра'
    }});
    const AL=0.85, alat=[], alon=[];
    for(let i=0;i<n;i++){{
      const dr=wd[i]*Math.PI/180;
      const dl=AL*Math.cos(dr);
      const dlo=AL*Math.sin(dr)/Math.max(Math.cos(lats[i]*Math.PI/180),.05);
      alat.push(lats[i],lats[i]+dl,null);
      alon.push(lons[i],lons[i]+dlo,null);
    }}
    traces.push({{
      type:'scattergeo', lat:alat, lon:alon, mode:'lines',
      line:{{width:0.6,color:'rgba(71,85,105,0.4)'}},
      name:'Направление ветра', hoverinfo:'skip'
    }});
  }}

  if(params.includes('wind_dir')){{
    traces.push({{
      type:'scattergeo', lat:lats, lon:lons, mode:'markers',
      marker:{{
        size:11, symbol:'arrow',
        color:wd,
        colorscale:'HSV', cmin:0, cmax:360,
        colorbar:{{title:'Направл.°',thickness:12,len:.55,x:1.01}},
        angle:wd, angleref:'up', opacity:0.82
      }},
      text:names.map((nm,i)=>`${{nm}}<br>Направление: ${{Math.round(wd[i])}}°`),
      hovertemplate:'<b>%{{text}}</b><extra></extra>',
      name:'Ветер'
    }});
  }}

  if(params.includes('storm')){{
    const mask=sp.map(v=>v>0.1);
    const sl=lats.filter((_,i)=>mask[i]);
    const so=lons.filter((_,i)=>mask[i]);
    const sv=sp.filter((_,i)=>mask[i]);
    const sdv=sd.filter((_,i)=>mask[i]);
    const sn=names.filter((_,i)=>mask[i]);
    if(sl.length){{
      traces.push({{
        type:'scattergeo', lat:sl, lon:so, mode:'markers',
        marker:{{
          size:sv.map(v=>5+v*3.5),
          color:sv.map(stormColor),
          opacity:0.76,
          line:{{width:0.4,color:'rgba(30,27,75,0.3)'}}
        }},
        text:sn.map((nm,i)=>`${{nm}}<br>Мощность: ${{sv[i].toFixed(1)}}<br>Направление: ${{Math.round(sdv[i])}}°`),
        hovertemplate:'<b>%{{text}}</b><extra></extra>',
        name:'Грозовой фронт'
      }});
      // Стрелки движения фронта
      const FA=1.3, fl=[], fo=[];
      for(let i=0;i<sl.length;i++){{
        const dr=sdv[i]*Math.PI/180;
        const dl=FA*Math.cos(dr);
        const dlo=FA*Math.sin(dr)/Math.max(Math.cos(sl[i]*Math.PI/180),.05);
        fl.push(sl[i],sl[i]+dl,null);
        fo.push(so[i],so[i]+dlo,null);
      }}
      traces.push({{
        type:'scattergeo', lat:fl, lon:fo, mode:'lines',
        line:{{width:0.9,color:'rgba(30,27,75,0.45)'}},
        name:'Движение фронта', hoverinfo:'skip'
      }});
    }}
  }}

  if(params.includes('turbulence')){{
    for(let lvl=1;lvl<=4;lvl++){{
      const ml=lats.filter((_,i)=>tb[i]===lvl);
      const mo=lons.filter((_,i)=>tb[i]===lvl);
      const mn=names.filter((_,i)=>tb[i]===lvl);
      const labels=['Норма','Умеренная','Сильная','Экстремальная'];
      if(!ml.length) continue;
      traces.push({{
        type:'scattergeo', lat:ml, lon:mo, mode:'markers',
        marker:{{size:9, symbol:'triangle-up',
                 color:turbColor(lvl), opacity:0.82,
                 line:{{width:0.5,color:'rgba(51,65,85,.4)'}}}},
        text:mn.map(nm=>`${{nm}}<br>Турбулентность: ${{labels[lvl-1]}}`),
        hovertemplate:'<b>%{{text}}</b><extra></extra>',
        name:`Турб.${{lvl}}: ${{labels[lvl-1]}}`
      }});
    }}
  }}

  if(params.includes('ice')){{
    // Нет обледенения
    const nl=lats.filter((_,i)=>!ic[i]);
    const no=lons.filter((_,i)=>!ic[i]);
    const nn=names.filter((_,i)=>!ic[i]);
    if(nl.length) traces.push({{
      type:'scattergeo', lat:nl, lon:no, mode:'markers',
      marker:{{size:7,symbol:'square',color:'#e2e8f0',opacity:0.45}},
      text:nn, hovertemplate:'<b>%{{text}}</b><br>Нет обледенения<extra></extra>',
      name:'Нет обледенения'
    }});
    // Обледенение — снежинки
    const il=lats.filter((_,i)=>ic[i]);
    const io=lons.filter((_,i)=>ic[i]);
    const inames=names.filter((_,i)=>ic[i]);
    if(il.length) traces.push({{
      type:'scattergeo', lat:il, lon:io, mode:'markers+text',
      marker:{{size:13,symbol:'star-diamond',
               color:'#bae6fd',opacity:.9,
               line:{{width:1,color:'#0284c7'}}}},
      text:il.map(()=>'❄'), textposition:'middle center',
      textfont:{{size:9,color:'#0369a1'}},
      customdata:inames,
      hovertemplate:'<b>%{{customdata}}</b><br>Обледенение ❄<extra></extra>',
      name:'Обледенение ❄'
    }});
  }}

  return traces;
}}

// ── Обновление карты ─────────────────────────────────────────────────────────
function updateMap(){{
  const traces = buildTraces(curZ, curT, getActiveLayers());
  Plotly.react('map-div', traces, document.getElementById('map-div').layout || {{}});
  updateLegend();
}}

function updateLegend(){{
  const layers = getActiveLayers();
  const leg = document.getElementById('legend');
  let html = '';
  if(layers.includes('wind_speed')){{
    html+='<h4>Ветер (м/с)</h4>';
    [['#7dd3fc','0–10'],['#4ade80','10–18'],['#facc15','18–26'],['#f97316','26–35'],['#dc2626','35+']].forEach(([c,l])=>{{
      html+=`<div class="legend-item"><div class="legend-sq" style="background:${{c}}"></div><span>${{l}}</span></div>`;
    }});
  }}
  if(layers.includes('storm')){{
    html+='<h4>Мощность (0–5)</h4>';
    [['#e0f2fe','0–1'],['#93c5fd','1–2.5'],['#6b7baa','2.5–4'],['#1e1b4b','4–5']].forEach(([c,l])=>{{
      html+=`<div class="legend-item"><div class="legend-dot" style="background:${{c}}"></div><span>${{l}}</span></div>`;
    }});
  }}
  if(layers.includes('turbulence')){{
    html+='<h4>Турбулентность</h4>';
    [['#f8fafc','1 — Норма'],['#cbd5e1','2 — Умеренная'],['#64748b','3 — Сильная'],['#0f172a','4 — Экстрем.']].forEach(([c,l])=>{{
      html+=`<div class="legend-item"><div class="legend-dot" style="background:${{c}};border:1px solid #94a3b8"></div><span>${{l}}</span></div>`;
    }});
  }}
  if(layers.includes('ice')){{
    html+='<h4>Обледенение</h4>';
    html+='<div class="legend-item"><div class="legend-sq" style="background:#e2e8f0"></div><span>Нет</span></div>';
    html+='<div class="legend-item"><div class="legend-sq" style="background:#bae6fd;border:1px solid #0284c7"></div><span>Есть ❄</span></div>';
  }}
  if(layers.includes('wind_dir')){{
    html+='<h4>Направление ветра</h4><p style="font-size:10px;color:#64748b">Цвет = азимут 0–360°<br>Стрелка = куда дует</p>';
  }}
  leg.innerHTML = html;
}}

// ── Управление ────────────────────────────────────────────────────────────────
function setZ(z){{
  curZ=z;
  document.querySelectorAll('#z-btns button').forEach(b=>b.classList.remove('active'));
  document.getElementById('z'+z).classList.add('active');
  updateMap();
}}

function setT(t){{
  curT=t;
  document.getElementById('t-label').textContent='t = '+t;
  document.getElementById('t-slider').value=t;
  updateMap();
}}

function togglePlay(){{
  const btn=document.getElementById('play-btn');
  if(playTimer){{
    clearInterval(playTimer); playTimer=null;
    btn.textContent='▶ Play'; btn.classList.remove('playing');
  }} else {{
    btn.textContent='⏸ Pause'; btn.classList.add('playing');
    playTimer=setInterval(()=>{{
      curT = curT>=N_T ? 1 : curT+1;
      setT(curT);
    }}, 150);
  }}
}}

// ── Старт ─────────────────────────────────────────────────────────────────────
window.onload = () => {{
  // Инициализируем карту с правильным geo layout
  Plotly.react('map-div', [], {{
    geo:{{
      scope:'asia', projection_type:'mercator',
      showland:true, landcolor:'#f1f5f9',
      showocean:true, oceancolor:'#dbeafe',
      showlakes:true, lakecolor:'#bae6fd',
      showcountries:true, countrycolor:'#cbd5e1', countrywidth:0.6,
      showcoastlines:true, coastlinecolor:'#94a3b8', coastlinewidth:0.7,
      lonaxis:{{range:[26,170]}}, lataxis:{{range:[40,78]}},
      bgcolor:'#f8fafc'
    }},
    paper_bgcolor:'#fff',
    margin:{{l:0,r:0,t:0,b:0}},
    showlegend:false
  }}).then(() => {{ document.getElementById('layer-checks').addEventListener('change', updateMap); updateMap(); }});
}};
</script>
</body>
</html>"""

    Path(output_html).write_text(html, encoding="utf-8")
    print(f"Карта сохранена: {output_html}")
    import webbrowser, os
    webbrowser.open("file://"+os.path.abspath(output_html))


# ── Точка входа ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    skip_gen = "--skip-gen" in sys.argv
    if skip_gen and Path(OUTPUT_FILE).exists():
        print(f"Загрузка {OUTPUT_FILE}...")
        records = json.loads(Path(OUTPUT_FILE).read_text(encoding="utf-8"))
        print(f"  {len(records)} записей")
    else:
        records = generate()
    draw_weather(records)