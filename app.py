from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime
from plane_simulation.plane_simulation import simulate
from prohibited_zones.prohibited_zones import generate_zones

app = Flask(__name__)
CORS(app)

# Файл для хранения точек
DATA_FILE = 'map_points.json'

def load_points():
    """Загрузка точек из файла"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_points(points):
    """Сохранение точек в файл"""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(points, f, ensure_ascii=False, indent=2)

@app.route('/')
def index():
    """Главная страница приложения"""
    return send_from_directory('.', 'index.html')

@app.route('/api/points', methods=['GET'])
def get_points():
    """Получить все точки"""
    points = load_points()
    return jsonify(points)

@app.route('/api/points', methods=['POST'])
def add_point():
    """Добавить новую точку"""
    data = request.json
    
    # Валидация данных
    if not data or 'lat' not in data or 'lng' not in data:
        return jsonify({'error': 'Требуются координаты lat и lng'}), 400
    
    try:
        lat = float(data['lat'])
        lng = float(data['lng'])
    except (ValueError, TypeError):
        return jsonify({'error': 'Некорректные координаты'}), 400
    
    # Проверка диапазона координат
    if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
        return jsonify({'error': 'Координаты вне допустимого диапазона'}), 400
    
    # Создание новой точки
    point = {
        'id': datetime.now().timestamp(),
        'lat': lat,
        'lng': lng,
        'label': data.get('label', 'Точка'),
        'description': data.get('description', ''),
        'created_at': datetime.now().isoformat()
    }
    
    # Сохранение точки
    points = load_points()
    points.append(point)
    save_points(points)
    
    return jsonify(point), 201

@app.route('/api/points/<point_id>', methods=['DELETE'])
def delete_point(point_id):
    """Удалить точку по ID"""
    points = load_points()
    point_id = float(point_id)
    
    points = [p for p in points if p['id'] != point_id]
    save_points(points)
    
    return jsonify({'message': 'Точка удалена'}), 200

@app.route('/api/points/batch', methods=['POST'])
def add_points_batch():
    """Добавить несколько точек за раз"""
    data = request.json
    
    if not data or 'points' not in data:
        return jsonify({'error': 'Требуется массив points'}), 400
    
    points = load_points()
    added_points = []
    
    for item in data['points']:
        try:
            lat = float(item['lat'])
            lng = float(item['lng'])
            
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                continue
            
            point = {
                'id': datetime.now().timestamp() + len(added_points) * 0.001,
                'lat': lat,
                'lng': lng,
                'label': item.get('label', 'Точка'),
                'description': item.get('description', ''),
                'created_at': datetime.now().isoformat()
            }
            
            points.append(point)
            added_points.append(point)
        except (ValueError, TypeError, KeyError):
            continue
    
    save_points(points)
    
    return jsonify({
        'message': f'Добавлено {len(added_points)} точек',
        'points': added_points
    }), 201

@app.route('/api/points/clear', methods=['DELETE'])
def clear_points():
    """Удалить все точки"""
    save_points([])
    return jsonify({'message': 'Все точки удалены'}), 200

@app.route('/api/points/filter', methods=['GET'])
def filter_points():
    """Фильтрация точек по границам карты"""
    min_lat = request.args.get('min_lat', type=float)
    max_lat = request.args.get('max_lat', type=float)
    min_lng = request.args.get('min_lng', type=float)
    max_lng = request.args.get('max_lng', type=float)
    
    points = load_points()
    
    if all(v is not None for v in [min_lat, max_lat, min_lng, max_lng]):
        filtered = [
            p for p in points
            if min_lat <= p['lat'] <= max_lat and min_lng <= p['lng'] <= max_lng
        ]
        return jsonify(filtered)
    
    return jsonify(points)

@app.route('/api/optimize', methods=['POST'])
def optimize():
    """Оптимизировать маршрут и вернуть 20 точек"""
    try:
        data = request.json or {}
        refresh_weather = data.get('refresh_weather')
        refresh_restricted_areas = data.get('refresh_restricted_areas')
        refresh_trajectories = data.get('refresh_trajectories')
        
        if refresh_weather:
            from weather.weather_generator import generate
            
            base_dir = os.path.dirname(__file__)
            wp_path = os.path.join(base_dir, 'weather', 'russia_waypoints.json')
            
            # Если файл russia_waypoints.json лежит в корне, а не в weather/ (на всякий случай)
            if not os.path.exists(wp_path):
                wp_path = os.path.join(base_dir, 'russia_waypoints.json')
                
            out_path = os.path.join(base_dir, 'weather', 'weather_data.json')
            
            generate(waypoints_path=wp_path, output_path=out_path)
            
            global _weather_cache
            _weather_cache = None

        zones_path = os.path.join(os.path.dirname(__file__), 'prohibited_zones', 'prohibited_zones.json')
        
        if refresh_restricted_areas or not os.path.exists(zones_path):
            generate_zones()

        result_path = os.path.join(os.path.dirname(__file__), 'plane_simulation', 'simulation_result.json')
        
        if refresh_trajectories or not os.path.exists(result_path):
            import random
            plane_no = random.randint(100, 999)
            result, _all_waypoints = simulate(plane_number=plane_no)
        else:
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
        
        # Добавляем данные о запретных зонах в результат
        if os.path.exists(zones_path):
            with open(zones_path, 'r', encoding='utf-8') as f:
                result['prohibited_zones'] = json.load(f)
                
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Получить статистику по точкам"""
    points = load_points()
    
    if not points:
        return jsonify({
            'total': 0,
            'bounds': None
        })
    
    lats = [p['lat'] for p in points]
    lngs = [p['lng'] for p in points]
    
    return jsonify({
        'total': len(points),
        'bounds': {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lng': min(lngs),
            'max_lng': max(lngs)
        },
        'center': {
            'lat': sum(lats) / len(lats),
            'lng': sum(lngs) / len(lngs)
        }
    })

@app.route('/weather/<path:filename>')
def serve_weather(filename):
    """Отдать файлы из папки weather (weather_data.json и т.д.)"""
    return send_from_directory('weather', filename)

# Cache for weather data indexed by z level
_weather_cache = None  # dict: z -> list of records

def _load_weather_cache():
    global _weather_cache
    if _weather_cache is not None:
        return _weather_cache
    path = os.path.join(os.path.dirname(__file__), 'weather', 'weather_data.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    cache = {}
    for r in records:
        z = r['z']
        if z not in cache:
            cache[z] = []
        cache[z].append(r)
    _weather_cache = cache
    return cache

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """Вернуть погодные данные для заданного эшелона z и шага времени t.
       z: 1-5 (эшелон), t: 1-100 (шаг времени).
       Если t не задан — возвращаем все t для данного z."""
    z = request.args.get('z', type=int)
    t = request.args.get('t', type=int)
    cache = _load_weather_cache()
    if cache is None:
        return jsonify({'error': 'weather_data.json не найден'}), 404
    if z is None:
        return jsonify({'error': 'Параметр z обязателен'}), 400
    records = cache.get(z, [])
    if t is not None:
        records = [r for r in records if r['t'] == t]
    return jsonify(records)

@app.route('/health', methods=['GET'])
def health():
    """Проверка состояния API"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)