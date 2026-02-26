const appData = {
    "header": {
        "title": "Интеллектуальная система расчета маршрутов воздушных судов",
        "subtitle": "Моделирование и динамическая оптимизация траектории полета с учетом погодных, навигационных и эксплуатационных факторов",
        "copyright": "© 2026 Василий Гурьянов"
    },
    "airports": [
        { "name": "Sheremetyevo", "iata": "SVO", "city": "Москва", "coords": [55.9726, 37.4146] },
        { "name": "Pulkovo", "iata": "LED", "city": "Санкт-Петербург", "coords": [59.8003, 30.2625] },
        { "name": "Tolmachevo", "iata": "OVB", "city": "Новосибирск", "coords": [55.0126, 82.6507] },
        { "name": "Sochi", "iata": "AER", "city": "Сочи", "coords": [43.4499, 39.9566] },
        { "name": "Kazan", "iata": "KZN", "city": "Казань", "coords": [55.6062, 49.2787] },
        { "name": "Knevichi", "iata": "VVO", "city": "Владивосток", "coords": [43.3990, 132.1480] },
        { "name": "Novy", "iata": "KHV", "city": "Хабаровск", "coords": [48.5280, 135.1880] },
        { "name": "Yelizovo", "iata": "PKC", "city": "Петропавловск-Камчатский", "coords": [53.1679, 158.4539] }
    ],
    "aircraft_types": [
        { "type": "Airbus A320-200", "max_passengers": 180 },
        { "type": "Boeing 737-800", "max_passengers": 189 },
        { "type": "Sukhoi Superjet 100-95B", "max_passengers": 98 },
        { "type": "Airbus A321-200", "max_passengers": 220 },
        { "type": "Boeing 777-300ER", "max_passengers": 396 }
    ]
};

function initApp() {
    document.getElementById('header-title').textContent = appData.header.title;
    document.getElementById('header-subtitle').textContent = appData.header.subtitle;
    document.getElementById('copyright').textContent = appData.header.copyright;

    const dep = document.getElementById('dep-airport');
    const arr = document.getElementById('arr-airport');
    appData.airports.forEach(a => {
        dep.add(new Option(`${a.city} (${a.iata})`, a.iata));
        arr.add(new Option(`${a.city} (${a.iata})`, a.iata));
    });
    arr.selectedIndex = 1;

    const ac = document.getElementById('aircraft-type');
    appData.aircraft_types.forEach(a => ac.add(new Option(a.type, a.type)));

    const now = new Date();
    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
    document.getElementById('departure-time').value = now.toISOString().slice(0, 16);

    updateAircraftConstraints();

    // Сохраняем начальные состояния для обоих рейсов
    saveCurrentFlightData();
    flightsData[2] = JSON.parse(JSON.stringify(flightsData[1]));
}

const flightsData = {
    1: null,
    2: null
};
let currentFlight = 1;

function saveCurrentFlightData() {
    flightsData[currentFlight] = {
        dep: document.getElementById('dep-airport').value,
        arr: document.getElementById('arr-airport').value,
        aircraft: document.getElementById('aircraft-type').value,
        pax: document.getElementById('pax-slider').value,
        baggage: document.getElementById('baggage-slider').value,
        fullLoad: document.getElementById('full-load').checked,
        fuel: document.getElementById('fuel-reserve').value,
        depTime: document.getElementById('departure-time').value
    };
}

function loadFlightData(flightNo) {
    const data = flightsData[flightNo];
    if (!data) return;

    document.getElementById('dep-airport').value = data.dep;
    document.getElementById('arr-airport').value = data.arr;
    document.getElementById('aircraft-type').value = data.aircraft;

    const aircraft = appData.aircraft_types.find(a => a.type === data.aircraft);
    const paxSlider = document.getElementById('pax-slider');
    paxSlider.max = aircraft.max_passengers;
    paxSlider.value = data.pax;

    const bagSlider = document.getElementById('baggage-slider');
    bagSlider.value = data.baggage;

    const fullLoadCb = document.getElementById('full-load');
    fullLoadCb.checked = data.fullLoad;
    bagSlider.disabled = data.fullLoad;

    document.getElementById('fuel-reserve').value = data.fuel;
    document.getElementById('departure-time').value = data.depTime;

    updatePaxValue();
    updateBaggageValue();
}

function switchFlight(flightNo) {
    if (flightNo === currentFlight) return;
    saveCurrentFlightData();
    currentFlight = flightNo;
    loadFlightData(flightNo);

    document.querySelectorAll('.flight-tab').forEach(tab => tab.classList.remove('active'));
    document.querySelector(`.flight-tab[onclick="switchFlight(${flightNo})"]`).classList.add('active');
}

function updateAircraftConstraints() {
    const selected = document.getElementById('aircraft-type').value;
    const aircraft = appData.aircraft_types.find(a => a.type === selected);
    const slider = document.getElementById('pax-slider');
    slider.max = aircraft.max_passengers;
    if (parseInt(slider.value) > aircraft.max_passengers) slider.value = aircraft.max_passengers;
    updatePaxValue();
}

function updatePaxValue() {
    document.getElementById('pax-value').textContent = `${document.getElementById('pax-slider').value} чел.`;
}

function updateBaggageValue() {
    document.getElementById('baggage-value').textContent = `${document.getElementById('baggage-slider').value} кг`;
}

function toggleFullLoad() {
    const cb = document.getElementById('full-load');
    const slider = document.getElementById('baggage-slider');
    slider.disabled = cb.checked;
    if (cb.checked) slider.value = 1000;
    updateBaggageValue();
}

// --- MAP (OpenLayers) ---
const map = new ol.Map({
    target: 'map',
    controls: [
        new ol.control.Attribution({
            collapsible: false
        })
    ],
    layers: [
        new ol.layer.Tile({
            source: new ol.source.OSM()
        })
    ],
    view: new ol.View({
        center: ol.proj.fromLonLat([80, 58]),
        zoom: 3,
        minZoom: 2,
        maxZoom: 11
    })
});

// --- DYNAMIC COORDINATE GRID ---
const gridSource = new ol.source.Vector();
const gridLayer = new ol.layer.Vector({
    source: gridSource,
    style: new ol.style.Style({
        stroke: new ol.style.Stroke({
            color: '#cbd5e1',
            width: 0.5,
            lineDash: [3, 7]
        })
    })
});
map.addLayer(gridLayer);

let gridOverlays = [];

function getGridStep(zoom) {
    if (zoom <= 2) return { line: 30, label: 60 };
    if (zoom <= 3) return { line: 20, label: 40 };
    if (zoom <= 4) return { line: 15, label: 30 };
    if (zoom <= 5) return { line: 10, label: 20 };
    if (zoom <= 6) return { line: 5, label: 10 };
    if (zoom <= 8) return { line: 2, label: 4 };
    return { line: 1, label: 2 };
}

function formatCoord(val, isLat) {
    if (isLat) return val === 0 ? '0°' : (val > 0 ? `${val}°N` : `${Math.abs(val)}°S`);
    return val === 0 ? '0°' : (val > 0 ? `${val}°E` : `${Math.abs(val)}°W`);
}

function updateGrid() {
    gridSource.clear();
    gridOverlays.forEach(o => map.removeOverlay(o));
    gridOverlays = [];

    const zoom = Math.floor(map.getView().getZoom());
    const extent = map.getView().calculateExtent(map.getSize());
    const coords = ol.proj.transformExtent(extent, 'EPSG:3857', 'EPSG:4326');
    const west = coords[0], south = coords[1], east = coords[2], north = coords[3];

    const { line: step } = getGridStep(zoom);

    const latMin = Math.floor(Math.max(south, -85) / step) * step;
    const latMax = Math.ceil(Math.min(north, 85) / step) * step;
    const lngMin = Math.floor(Math.max(west, -180) / step) * step;
    const lngMax = Math.ceil(Math.min(east, 180) / step) * step;

    // Latitudes
    for (let lat = latMin; lat <= latMax; lat += step) {
        const line = new ol.Feature(new ol.geom.LineString([
            ol.proj.fromLonLat([-180, lat]),
            ol.proj.fromLonLat([180, lat])
        ]));
        gridSource.addFeature(line);

        const labelLng = Math.max(west + (east - west) * 0.01, -179);
        const el = document.createElement('div');
        el.className = 'grid-label';
        el.textContent = formatCoord(lat, true);

        const overlay = new ol.Overlay({
            position: ol.proj.fromLonLat([labelLng, lat]),
            element: el,
            offset: [0, -9],
            positioning: 'center-left'
        });
        map.addOverlay(overlay);
        gridOverlays.push(overlay);
    }

    // Longitudes
    for (let lng = lngMin; lng <= lngMax; lng += step) {
        const line = new ol.Feature(new ol.geom.LineString([
            ol.proj.fromLonLat([lng, -85]),
            ol.proj.fromLonLat([lng, 85])
        ]));
        gridSource.addFeature(line);

        const labelLat = Math.max(south + (north - south) * 0.01, -84);
        const el = document.createElement('div');
        el.className = 'grid-label';
        el.textContent = formatCoord(lng, false);

        const overlay = new ol.Overlay({
            position: ol.proj.fromLonLat([lng, labelLat]),
            element: el,
            offset: [0, 9],
            positioning: 'top-center'
        });
        map.addOverlay(overlay);
        gridOverlays.push(overlay);
    }
}

map.getView().on('change:resolution', updateGrid);
map.getView().on('change:center', updateGrid);
updateGrid();

// --- AIRPORTS ---
const mainAirportSource = new ol.source.Vector();
const mainAirportLayer = new ol.layer.Vector({
    source: mainAirportSource,
    zIndex: 2000 // Topmost layer
});
map.addLayer(mainAirportLayer);

appData.airports.forEach(a => {
    const pos = ol.proj.fromLonLat([a.coords[1], a.coords[0]]);

    // Marker circle featuer
    const marker = new ol.Feature(new ol.geom.Point(pos));
    marker.setStyle(new ol.style.Style({
        image: new ol.style.Circle({
            radius: 5,
            fill: new ol.style.Fill({ color: '#0284c7' }),
            stroke: new ol.style.Stroke({ color: '#fff', width: 2 })
        })
    }));
    mainAirportSource.addFeature(marker);

    // Tooltip Overlay
    const tip = document.createElement('div');
    tip.className = 'custom-tooltip';
    tip.textContent = a.city;
    const overlay = new ol.Overlay({
        position: pos,
        element: tip,
        offset: [0, -12],
        positioning: 'bottom-center'
    });
    map.addOverlay(overlay);
});

// --- ROUTE VISUALIZATION ---
const routeSource = new ol.source.Vector();
const routeLayer = new ol.layer.Vector({
    source: routeSource,
    zIndex: 998,
    style: function (feature) {
        const type = feature.get('type');
        if (type === 'line') {
            return new ol.style.Style({
                stroke: new ol.style.Stroke({
                    color: '#f97316',
                    width: 3
                })
            });
        } else if (type === 'point') {
            return new ol.style.Style({
                image: new ol.style.Circle({
                    radius: 4,
                    fill: new ol.style.Fill({ color: '#f97316' }),
                    stroke: new ol.style.Stroke({ color: '#fff', width: 1.5 })
                }),
                text: new ol.style.Text({
                    text: feature.get('index'),
                    font: 'bold 10px Inter',
                    fill: new ol.style.Fill({ color: '#c2410c' }),
                    offsetY: -10
                })
            });
        }
    }
});
map.addLayer(routeLayer);

// --- PLANE SIMULATION ---
const planeSource = new ol.source.Vector();
const planeLayer = new ol.layer.Vector({
    source: planeSource,
    zIndex: 3000
});
map.addLayer(planeLayer);

// Plane Icon (Simple SVG Data URI)
const planeSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#38bdf8" width="48px" height="48px"><path stroke="#000" stroke-width="0.3" d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/></svg>`;
const planeIconUrl = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(planeSvg.replace('#38bdf8', '#85a2afff'));
//цвет встречного самолета
const planeStyle = new ol.style.Style({
    image: new ol.style.Icon({
        src: planeIconUrl,
        scale: 1.5,
        anchor: [0.5, 0.5],
        rotation: 0,
        rotateWithView: true
    })
});

let currentRoutePoints = [];
let planeFeature = null;
let animationInterval = null;
let isPlaying = false;

// ── ЗАПРЕТНЫЕ ЗОНЫ ────────────────────────────────────────────────────────
let PROHIBITED_ZONES = [];
let zonesLayer = null;

function getHatchPattern() {
    const canvas = document.createElement('canvas');
    canvas.width = 8;
    canvas.height = 8;
    const ctx = canvas.getContext('2d');

    ctx.strokeStyle = 'rgba(239, 68, 68, 0.2)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, 8);
    ctx.lineTo(8, 0);
    ctx.stroke();

    return ctx.createPattern(canvas, 'repeat');
}

// Слой запретных зон (под погодой и траекториями)
zonesLayer = new ol.layer.Vector({
    source: new ol.source.Vector(),
    zIndex: 997, // Below routeLayer (998)
    style: function (feature) {
        return new ol.style.Style({
            stroke: new ol.style.Stroke({
                color: 'rgba(239, 68, 68, 0.4)',
                width: 1.5
            }),
            fill: new ol.style.Fill({
                color: getHatchPattern()
            })
        });
    }
});
map.addLayer(zonesLayer);

function renderProhibitedZones() {
    if (!zonesLayer) return;
    const source = zonesLayer.getSource();
    source.clear();

    if (!document.getElementById('toggle-zones').checked) return;

    PROHIBITED_ZONES.forEach(zone => {
        const coords = zone.points.map(p => ol.proj.fromLonLat([p.lon, p.lat]));
        coords.push(coords[0]); // замыкаем полигон

        const feature = new ol.Feature({
            geometry: new ol.geom.Polygon([coords])
        });
        source.addFeature(feature);
    });
}

function toggleZonesVisibility() {
    renderProhibitedZones();
}

// ── WAYPOINTS ─────────────────────────────────────────────────────────────
const waypointSource = new ol.source.Vector();
const waypointLayer = new ol.layer.Vector({
    source: waypointSource,
    zIndex: 5, // Below everything else (prohibited zones 997, route 998, plane 999)
    visible: false,
    style: new ol.style.Style({
        image: new ol.style.Circle({
            radius: 3.5,
            fill: new ol.style.Fill({ color: '#4b5563' }),
            stroke: new ol.style.Stroke({ color: '#fff', width: 1 })
        })
    })
});
map.addLayer(waypointLayer);

async function loadWaypoints() {
    try {
        const resp = await fetch('/api/waypoints');
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();

        const features = data.map(wp => {
            const f = new ol.Feature({
                geometry: new ol.geom.Point(ol.proj.fromLonLat([wp.lon, wp.lat])),
                name: wp.name,
                lat: wp.lat,
                lon: wp.lon
            });
            return f;
        });
        waypointSource.addFeatures(features);
    } catch (e) {
        console.error('Waypoints load error:', e);
    }
}
loadWaypoints();

function toggleWaypointsVisibility() {
    waypointLayer.setVisible(document.getElementById('toggle-waypoints').checked);
}

// --- TOOLTIP INTERACTION ---
const mapTooltipEl = document.createElement('div');
mapTooltipEl.className = 'custom-tooltip';
mapTooltipEl.style.display = 'none';
mapTooltipEl.style.position = 'absolute';
mapTooltipEl.style.zIndex = '1001';
document.body.appendChild(mapTooltipEl);

const waypointTooltipOverlay = new ol.Overlay({
    element: mapTooltipEl,
    offset: [10, 0],
    positioning: 'center-left'
});
map.addOverlay(waypointTooltipOverlay);

map.on('pointermove', function (evt) {
    if (evt.dragging) {
        mapTooltipEl.style.display = 'none';
        return;
    }

    const pixel = map.getEventPixel(evt.originalEvent);

    // Check airports first (higher zIndex/importance)
    const airportFeature = map.forEachFeatureAtPixel(pixel, f => f, {
        layerFilter: l => l === airportLayer
    });

    if (airportFeature) {
        const name = airportFeature.get('name');
        const runway = airportFeature.get('runway');
        mapTooltipEl.innerHTML = `<strong>${name}</strong><br>Полоса: ${runway} м`;
        mapTooltipEl.style.display = 'block';
        waypointTooltipOverlay.setPosition(evt.coordinate);
        map.getTargetElement().style.cursor = 'pointer';
        return;
    }

    const waypointFeature = map.forEachFeatureAtPixel(pixel, f => f, {
        layerFilter: l => l === waypointLayer
    });

    if (waypointFeature) {
        const name = waypointFeature.get('name');
        const lat = waypointFeature.get('lat').toFixed(2);
        const lon = waypointFeature.get('lon').toFixed(2);
        mapTooltipEl.innerHTML = `${name}<br><span style="font-size: 9px; color: #64748b;">${lat}, ${lon}</span>`;
        mapTooltipEl.style.display = 'block';
        waypointTooltipOverlay.setPosition(evt.coordinate);
        map.getTargetElement().style.cursor = 'pointer';
    } else {
        mapTooltipEl.style.display = 'none';
        map.getTargetElement().style.cursor = '';
    }
});

// ── АЭРОПОРТЫ ─────────────────────────────────────────────────────────────
const airportSource = new ol.source.Vector();
const airportLayer = new ol.layer.Vector({
    source: airportSource,
    zIndex: 6, // Above waypoints (5)
    visible: false,
    style: function (feature) {
        const isLong = feature.get('class') === 'Long';
        return new ol.style.Style({
            image: new ol.style.Circle({
                fill: new ol.style.Fill({ color: isLong ? '#fbbf24' : '#ef4444' }), // Yellow / Red
                radius: isLong ? 6 : 4
            })
        });
    }
});
map.addLayer(airportLayer);

async function loadAirports() {
    try {
        const resp = await fetch('/airports/russian_civil_airports.json');
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();

        const features = data.map(ap => {
            return new ol.Feature({
                geometry: new ol.geom.Point(ol.proj.fromLonLat([ap.lon, ap.lat])),
                name: ap.name,
                runway: ap.max_runway_m,
                class: ap.class
            });
        });
        airportSource.addFeatures(features);
    } catch (e) {
        console.error('Airports load error:', e);
    }
}
loadAirports();

function toggleAirportsVisibility() {
    airportLayer.setVisible(document.getElementById('toggle-airports').checked);
}

// ── WEATHER DATA ─────────────────────────────────────────────────────────
let WEATHER_DATA = null;  // indexed as WEATHER_DATA[z][t]
let WEATHER_CACHE = {}; // Cache for weather data to avoid re-fetching
let weatherLayers = [];   // ol.layer.Vector instances for weather
let curFL = 3;            // current flight level (1–5)
let curWT = 1;            // current weather time step
let activeWeatherParams = new Set(['wind_speed']); // active params
let weatherTKeys = [];    // sorted time-step keys available in WEATHER_DATA, limited to trajectory length

const FL_LABELS = { 1: 'FL100', 2: 'FL200', 3: 'FL300', 4: 'FL350', 5: 'FL400' };

// Color helpers (same as weather_map.html)
function lerpColor(c1, c2, t) {
    const h = s => parseInt(s.slice(1), 16);
    const r1 = h(c1) >> 16, g1 = (h(c1) >> 8) & 255, b1 = h(c1) & 255;
    const r2 = h(c2) >> 16, g2 = (h(c2) >> 8) & 255, b2 = h(c2) & 255;
    const r = Math.round(r1 + t * (r2 - r1)), g = Math.round(g1 + t * (g2 - g1)), b = Math.round(b1 + t * (b2 - b1));
    return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
}
function windColor(v) {
    const stops = [[0, '#7dd3fc'], [10, '#4ade80'], [18, '#facc15'], [26, '#f97316'], [35, '#dc2626']];
    v = Math.max(0, Math.min(35, v));
    for (let i = 1; i < stops.length; i++) {
        if (v <= stops[i][0]) {
            const tt = (v - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]);
            return lerpColor(stops[i - 1][1], stops[i][1], tt);
        }
    }
    return stops[stops.length - 1][1];
}
function stormColor(v) {
    const stops = [[0, '#e0f2fe'], [1, '#93c5fd'], [2.5, '#6b7baa'], [4, '#3b5bbd'], [5, '#1e1b4b']];
    v = Math.max(0, Math.min(5, v));
    for (let i = 1; i < stops.length; i++) {
        if (v <= stops[i][0]) {
            const tt = (v - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]);
            return lerpColor(stops[i - 1][1], stops[i][1], tt);
        }
    }
    return stops[stops.length - 1][1];
}
function turbColor(lvl) {
    return ['#f8fafc', '#cbd5e1', '#64748b', '#0f172a'][lvl - 1];
}
function hexToRgba(hex, alpha) {
    const h = parseInt(hex.slice(1), 16);
    const r = (h >> 16) & 255, g = (h >> 8) & 255, b = h & 255;
    return `rgba(${r},${g},${b},${alpha})`;
}

function setFL(z) {
    curFL = z;
    document.querySelectorAll('.fl-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('fl' + z).classList.add('active');
    // Re-sync tKeys from new FL, keep current slider position
    if (WEATHER_DATA && WEATHER_DATA[z]) {
        const allKeys = Object.keys(WEATHER_DATA[z]).map(Number).sort((a, b) => a - b);
        weatherTKeys = allKeys; // already filtered to N during load
        // keep curWT if it exists in new FL, else pick first
        if (!weatherTKeys.includes(curWT)) curWT = weatherTKeys[0] || 1;
    }
    renderWeatherLayer();
}

function toggleWeatherParam(param) {
    if (activeWeatherParams.has(param)) {
        activeWeatherParams.delete(param);
        document.getElementById('wp-' + param).classList.remove('active');
    } else {
        activeWeatherParams.add(param);
        document.getElementById('wp-' + param).classList.add('active');
    }
    renderWeatherLayer();
    updateWeatherColorLegend();
}

function clearWeatherLayers() {
    weatherLayers.forEach(l => map.removeLayer(l));
    weatherLayers = [];
}

function renderWeatherLayer() {
    if (!WEATHER_DATA) return;
    clearWeatherLayers();

    const zKey = curFL;
    const tKey = curWT;
    if (!WEATHER_DATA[zKey] || !WEATHER_DATA[zKey][tKey]) return;

    const d = WEATHER_DATA[zKey][tKey];
    const n = d.la.length;

    // Build features for each active param
    const allFeatures = [];

    if (activeWeatherParams.has('wind_speed')) {
        const AL = 0.85;
        for (let i = 0; i < n; i++) {
            const f = new ol.Feature({
                geometry: new ol.geom.Point(ol.proj.fromLonLat([d.lo[i], d.la[i]]))
            });
            const color = windColor(d.ws[i]);
            f.setStyle(new ol.style.Style({
                image: new ol.style.RegularShape({
                    fill: new ol.style.Fill({ color: hexToRgba(color, 0.78) }),
                    stroke: new ol.style.Stroke({ color: hexToRgba(color, 0.4), width: 0.5 }),
                    points: 4,
                    radius: 9,
                    angle: Math.PI / 4
                })
            }));
            allFeatures.push(f);

            // Direction line
            const dr = d.wd[i] * Math.PI / 180;
            const dl = AL * Math.cos(dr);
            const dlo = AL * Math.sin(dr) / Math.max(Math.cos(d.la[i] * Math.PI / 180), 0.05);
            const lineF = new ol.Feature({
                geometry: new ol.geom.LineString([
                    ol.proj.fromLonLat([d.lo[i], d.la[i]]),
                    ol.proj.fromLonLat([d.lo[i] + dlo, d.la[i] + dl])
                ])
            });
            lineF.setStyle(new ol.style.Style({
                stroke: new ol.style.Stroke({
                    color: 'rgba(71,85,105,0.4)',
                    width: 1.5
                })
            }));
            allFeatures.push(lineF);
        }
    }

    if (activeWeatherParams.has('storm')) {
        const FA = 1.3;
        for (let i = 0; i < n; i++) {
            if (d.sp[i] <= 0.1) continue;
            const f = new ol.Feature({
                geometry: new ol.geom.Point(ol.proj.fromLonLat([d.lo[i], d.la[i]]))
            });
            const color = stormColor(d.sp[i]);
            const r = 5 + d.sp[i] * 3.5;
            f.setStyle(new ol.style.Style({
                image: new ol.style.Circle({
                    radius: r,
                    fill: new ol.style.Fill({ color: hexToRgba(color, 0.76) }),
                    stroke: new ol.style.Stroke({ color: 'rgba(30,27,75,0.3)', width: 0.6 })
                })
            }));
            allFeatures.push(f);

            // Direction line
            const dr = d.sd[i] * Math.PI / 180;
            const dl = FA * Math.cos(dr);
            const dlo = FA * Math.sin(dr) / Math.max(Math.cos(d.la[i] * Math.PI / 180), 0.05);
            const lineF = new ol.Feature({
                geometry: new ol.geom.LineString([
                    ol.proj.fromLonLat([d.lo[i], d.la[i]]),
                    ol.proj.fromLonLat([d.lo[i] + dlo, d.la[i] + dl])
                ])
            });
            lineF.setStyle(new ol.style.Style({
                stroke: new ol.style.Stroke({
                    color: 'rgba(30,27,75,0.45)',
                    width: 1.5
                })
            }));
            allFeatures.push(lineF);
        }
    }

    if (activeWeatherParams.has('turbulence')) {
        const labels = ['Норма', 'Умеренная', 'Сильная', 'Экстремальная'];
        for (let i = 0; i < n; i++) {
            const lvl = d.tb[i];
            if (!lvl) continue;
            const f = new ol.Feature({
                geometry: new ol.geom.Point(ol.proj.fromLonLat([d.lo[i], d.la[i]]))
            });
            const color = turbColor(lvl);
            f.setStyle(new ol.style.Style({
                image: new ol.style.RegularShape({
                    fill: new ol.style.Fill({ color: hexToRgba(color, 0.82) }),
                    stroke: new ol.style.Stroke({ color: 'rgba(51,65,85,0.4)', width: 0.5 }),
                    points: 3,
                    radius: 9,
                    angle: 0
                })
            }));
            allFeatures.push(f);
        }
    }

    if (activeWeatherParams.has('ice')) {
        for (let i = 0; i < n; i++) {
            const f = new ol.Feature({
                geometry: new ol.geom.Point(ol.proj.fromLonLat([d.lo[i], d.la[i]]))
            });
            if (d.ic[i]) {
                // Icing: star shape, light blue
                f.setStyle(new ol.style.Style({
                    image: new ol.style.RegularShape({
                        fill: new ol.style.Fill({ color: 'rgba(186,230,253,0.9)' }),
                        stroke: new ol.style.Stroke({ color: '#0284c7', width: 1 }),
                        points: 6,
                        radius: 8,
                        radius2: 4,
                        angle: 0
                    })
                }));
            } else {
                f.setStyle(new ol.style.Style({
                    image: new ol.style.RegularShape({
                        fill: new ol.style.Fill({ color: 'rgba(226,232,240,0.45)' }),
                        stroke: new ol.style.Stroke({ color: 'rgba(148,163,184,0.3)', width: 0.5 }),
                        points: 4,
                        radius: 6,
                        angle: Math.PI / 4
                    })
                }));
            }
            allFeatures.push(f);
        }
    }

    if (allFeatures.length > 0) {
        const wSource = new ol.source.Vector({ features: allFeatures });
        const wLayer = new ol.layer.Vector({
            source: wSource,
            zIndex: 10  // below trajectory (zIndex 999)
        });
        map.addLayer(wLayer);
        weatherLayers.push(wLayer);
    }

    updateWeatherColorLegend();
}

function updateWeatherColorLegend() {
    const el = document.getElementById('weather-color-legend');
    if (!el) return;
    let html = '';
    if (activeWeatherParams.has('wind_speed')) {
        html += '<div class="legend-section-title">Ветер (м/с)</div>';
        [['#7dd3fc', '0–10'], ['#4ade80', '10–18'], ['#facc15', '18–26'], ['#f97316', '26–35'], ['#dc2626', '35+']].forEach(([c, l]) => {
            html += `<div class="legend-swatch-row"><div class="legend-swatch" style="background:${c}"></div><span>${l}</span></div>`;
        });
    }
    if (activeWeatherParams.has('storm')) {
        html += '<div class="legend-section-title">Гроза (0–5)</div>';
        [['#e0f2fe', '0–1'], ['#93c5fd', '1–2.5'], ['#6b7baa', '2.5–4'], ['#1e1b4b', '4–5']].forEach(([c, l]) => {
            html += `<div class="legend-swatch-row"><div class="legend-swatch-dot" style="background:${c};border:1px solid #94a3b8"></div><span>${l}</span></div>`;
        });
    }
    if (activeWeatherParams.has('turbulence')) {
        html += '<div class="legend-section-title">Турбулентность</div>';
        [['#f8fafc', '1-Норма'], ['#cbd5e1', '2-Умеренная'], ['#64748b', '3-Сильная'], ['#0f172a', '4-Экстрем.']].forEach(([c, l]) => {
            html += `<div class="legend-swatch-row"><div class="legend-swatch-dot" style="background:${c};border:1px solid #94a3b8"></div><span>${l}</span></div>`;
        });
    }
    if (activeWeatherParams.has('ice')) {
        html += '<div class="legend-section-title">Обледенение</div>';
        html += '<div class="legend-swatch-row"><div class="legend-swatch" style="background:#e2e8f0"></div><span>Нет</span></div>';
        html += '<div class="legend-swatch-row"><div class="legend-swatch-dot" style="background:#bae6fd;border:1px solid #0284c7"></div><span>Да</span></div>';
    }
    el.innerHTML = html;
}

// Load weather_data.json and index only t = 1..N records (N = trajectory length)
async function loadWeatherData(N) {
    if (WEATHER_DATA) {
        // Already loaded — just sync to new trajectory length and re-render
        syncWeatherToTrajectory(N);
        return;
    }
    try {
        const resp = await fetch('/weather/weather_data.json?t=' + Date.now());
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const records = await resp.json();
        WEATHER_DATA = {};
        for (const r of records) {
            const z = r.z, t = r.t;
            // Pre-filter: only keep time steps up to N
            if (t < 1 || t > N) continue;
            if (!WEATHER_DATA[z]) WEATHER_DATA[z] = {};
            if (!WEATHER_DATA[z][t]) WEATHER_DATA[z][t] = { la: [], lo: [], nm: [], ws: [], wd: [], sp: [], sd: [], tb: [], ic: [] };
            const d = WEATHER_DATA[z][t];
            d.la.push(r.lat); d.lo.push(r.lon); d.nm.push(r.name);
            d.ws.push(r.wind_speed); d.wd.push(r.wind_dir);
            d.sp.push(r.storm_power); d.sd.push(r.storm_dir);
            d.tb.push(r.turbulence); d.ic.push(r.ice);
        }
        syncWeatherToTrajectory(N);
    } catch (e) {
        console.error('Weather load error:', e);
        alert('Не удалось загрузить weather_data.json: ' + e.message);
    }
}

// Set up weatherTKeys as sorted array of t values for curFL, length <= N
function syncWeatherToTrajectory(N) {
    if (!WEATHER_DATA || !WEATHER_DATA[curFL]) return;
    const allKeys = Object.keys(WEATHER_DATA[curFL]).map(Number).sort((a, b) => a - b);
    // Take first N keys (each key corresponds to one trajectory point)
    weatherTKeys = allKeys.slice(0, N);
    // Start at first time step
    curWT = weatherTKeys[0] || 1;
    renderWeatherLayer();
}

// Called when slider moves: map slider index → weather time step
function updateWeatherForSlider(idx) {
    if (!WEATHER_DATA || weatherTKeys.length === 0) return;
    idx = parseInt(idx);
    // Clamp to available weather keys
    const clampedIdx = Math.min(idx, weatherTKeys.length - 1);
    const newWT = weatherTKeys[clampedIdx];
    if (newWT !== undefined && newWT !== curWT) {
        curWT = newWT;
        renderWeatherLayer();
    }
}

async function optimizeRoute() {
    const btn = document.getElementById('optimize-btn');
    const originalText = btn.textContent;
    const refreshWeather = document.getElementById('refresh-weather').checked;
    const refreshRestrictedAreas = document.getElementById('refresh-restricted-areas').checked;
    const refreshTrajectories = document.getElementById('refresh-trajectories').checked;

    try {
        btn.textContent = 'Вычисление...';
        btn.disabled = true;

        const response = await fetch('/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                refresh_weather: refreshWeather,
                refresh_restricted_areas: refreshRestrictedAreas,
                refresh_trajectories: refreshTrajectories
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Ошибка сервера');
        }

        if (data.prohibited_zones) {
            PROHIBITED_ZONES = data.prohibited_zones;
            renderProhibitedZones();
        }

        drawRouteOnMap(data);

        if (refreshWeather) {
            WEATHER_DATA = null;
        }

        // Load weather and sync to trajectory after route is available
        if (data.approximated_20) {
            const N = data.approximated_20.length;
            await loadWeatherData(N);
        }

        document.getElementById('playback-panel').classList.remove('disabled');
        document.getElementById('map-legend').classList.remove('disabled');
        btn.textContent = 'Маршрут рассчитан';
        btn.style.background = '#059669';

    } catch (error) {
        console.error('Optimization failed:', error);
        alert('Ошибка расчета: ' + error.message);
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

function drawRouteOnMap(data) {
    routeSource.clear();
    planeSource.clear();
    currentRoutePoints = data.approximated_20;

    if (!currentRoutePoints || currentRoutePoints.length === 0) return;

    const coords = currentRoutePoints.map(p => ol.proj.fromLonLat([p.lon, p.lat]));

    // Draw line
    const routeLine = new ol.Feature({
        geometry: new ol.geom.LineString(coords),
        type: 'line'
    });
    routeSource.addFeature(routeLine);

    // Draw points
    currentRoutePoints.forEach((p, i) => {
        const feature = new ol.Feature({
            geometry: new ol.geom.Point(ol.proj.fromLonLat([p.lon, p.lat])),
            type: 'point',
            index: (i + 1).toString()
        });
        routeSource.addFeature(feature);
    });

    // Initialize Plane at Start
    const startCoords = ol.proj.fromLonLat([currentRoutePoints[0].lon, currentRoutePoints[0].lat]);
    planeFeature = new ol.Feature({
        geometry: new ol.geom.Point(startCoords)
    });
    planeFeature.setStyle(planeStyle);
    planeSource.addFeature(planeFeature);

    // Setup Slider
    const timeline = document.getElementById('timeline');
    timeline.min = 0;
    timeline.max = currentRoutePoints.length - 1;
    timeline.value = 0;

    // Rotation for initial position
    if (currentRoutePoints.length > 1) {
        updatePlaneRotation(0);
    }

    // Zoom to route
    map.getView().fit(routeSource.getExtent(), {
        padding: [50, 50, 50, 50],
        duration: 1000
    });
}

function updatePlanePosition(index) {
    index = parseInt(index);
    if (!planeFeature || !currentRoutePoints[index]) return;

    const point = currentRoutePoints[index];
    const coords = ol.proj.fromLonLat([point.lon, point.lat]);

    planeFeature.getGeometry().setCoordinates(coords);
    updatePlaneRotation(index);
}

function updatePlaneRotation(index) {
    if (index < currentRoutePoints.length - 1) {
        const current = currentRoutePoints[index];
        const next = currentRoutePoints[index + 1];

        const c1 = ol.proj.fromLonLat([current.lon, current.lat]);
        const c2 = ol.proj.fromLonLat([next.lon, next.lat]);
        const dxProj = c2[0] - c1[0];
        const dyProj = c2[1] - c1[1];

        // standard atan2(y, x) gives angle from East (0) counter-clockwise (usually).
        // OpenLayers rotation is radians clockwise.
        // We want 0 to be North (Up).
        // If we use atan2(dx, dy), we get angle from North, clockwise.
        // Example: East (dx=1, dy=0) -> atan2(1, 0) = PI/2 (90 deg). Correct.
        // South (dx=0, dy=-1) -> atan2(0, -1) = PI (180 deg). Correct.
        const rotation = Math.atan2(dxProj, dyProj);

        const newStyle = new ol.style.Style({
            image: new ol.style.Icon({
                src: planeIconUrl,
                scale: 1.5,
                anchor: [0.5, 0.5],
                rotation: rotation,
                rotateWithView: true
            })
        });
        planeFeature.setStyle(newStyle);
    }
}

function toggleTrajectoryVisibility() {
    const isVisible = document.getElementById('toggle-trajectory').checked;
    routeLayer.setVisible(isVisible);
}

// --- PLAYBACK CONTROLS ---
const playBtn = document.getElementById('play-pause');
const timeline = document.getElementById('timeline');

playBtn.addEventListener('click', togglePlay);

timeline.addEventListener('input', (e) => {
    if (isPlaying) togglePlay(); // Pause on manual interaction
    updatePlanePosition(e.target.value);
    updateWeatherForSlider(e.target.value);
});

function togglePlay() {
    if (!currentRoutePoints || currentRoutePoints.length === 0) return;

    isPlaying = !isPlaying;
    playBtn.classList.toggle('playing', isPlaying);

    if (isPlaying) {
        if (parseInt(timeline.value) >= parseInt(timeline.max)) {
            timeline.value = 0; // Restart from beginning
            updatePlanePosition(0);
        }

        animationInterval = setInterval(() => {
            let val = parseInt(timeline.value);
            if (val < parseInt(timeline.max)) {
                val++;
                timeline.value = val;
                updatePlanePosition(val);
                updateWeatherForSlider(val);
            } else {
                // Stop at end
                togglePlay();
            }
        }, 500);
    } else {
        clearInterval(animationInterval);
        animationInterval = null;
    }
}

initApp();
