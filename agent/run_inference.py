import os
import sys
import json
import builtins
from pathlib import Path

# Добавляем текущую директорию в path, чтобы импортировать the_two_planes
sys.path.append(os.path.dirname(__file__))

import the_two_planes

def run_agent_inference(flights_data=None):
    print("Starting agent AI inference...")
    
    # Stub print to avoid encoding crashes in Windows console from the_two_planes.py
    original_print = builtins.print
    def safe_print(*args, **kwargs):
        pass # Skip printing to avoid encoding errors
    builtins.print = safe_print
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agent_dir = os.path.join(base_dir, 'agent')
    
    # Пути к файлам данных
    data_files = {
        "russian_civil_airports.json": os.path.join(base_dir, 'airports', 'russian_civil_airports.json'),
        "prohibited_zones.json": os.path.join(base_dir, 'prohibited_zones', 'prohibited_zones.json'),
        "allowed_to_use_waypoints.json": os.path.join(base_dir, 'prohibited_zones', 'allowed_to_use_waypoints.json'),
        "weather_data.json": os.path.join(base_dir, 'weather', 'weather_data.json'),
        "simulation_result.json": os.path.join(base_dir, 'plane_simulation', 'simulation_result.json')
    }
    
    # Проверяем наличие всех файлов
    for name, path in data_files.items():
        if not os.path.exists(path):
            original_print(f"Error: Missing data file {path}")
            builtins.print = original_print
            return []

    # Загружаем аэропорты для маппинга
    with open(data_files["russian_civil_airports.json"], "r", encoding="utf-8") as f:
        airports_list = json.load(f)
    
    # Создаем lookup (названия могут быть разными, ищем по ICAO/IATA или имени)
    airport_lookup = {}
    for ap in airports_list:
        if ap.get("iata"): airport_lookup[ap["iata"]] = ap
        if ap.get("icao"): airport_lookup[ap["icao"]] = ap
        airport_lookup[ap["name"]] = ap

    # Маппим flights_data в формат агентов
    agent_routes = None
    agent_aircraft = None
    
    if flights_data:
        try:
            agent_routes = []
            agent_aircraft = []
            for f_id, f_val in flights_data.items():
                # Ищем аэропорты
                dep_ap = airport_lookup.get(f_val.get("dep"))
                arr_ap = airport_lookup.get(f_val.get("arr"))
                
                if not dep_ap or not arr_ap:
                    original_print(f"Warning: Could not find coordinates for {f_val.get('dep')} or {f_val.get('arr')}")
                    continue
                
                # Маппим тип самолета (используем алиасы из the_two_planes если нужно)
                raw_type = f_val.get("aircraft", "B773ER")
                mapped_type = the_two_planes.AIRCRAFT_TYPE_ALIAS.get(raw_type, raw_type)
                
                # Составляем маршрут
                route = {
                    "plane_number": int(f_id),
                    "departure": {"name": dep_ap["name"], "lat": dep_ap["lat"], "lon": dep_ap["lon"]},
                    "arrival": {"name": arr_ap["name"], "lat": arr_ap["lat"], "lon": arr_ap["lon"]},
                    "gc_distance_km": the_two_planes.haversine(dep_ap["lat"], dep_ap["lon"], arr_ap["lat"], arr_ap["lon"]),
                    "corridor_km": 1000,
                    "start_t": 1,
                    "start_altitude": 3
                }
                agent_routes.append(route)
                
                # Составляем конфиг самолета
                config = {
                    "aircraft_type": mapped_type,
                    "passengers": int(f_val.get("pax", 150)),
                    "baggage_kg": float(f_val.get("baggage", 2000)),
                    "min_reserve_fuel_kg": float(f_val.get("fuel", 5000))
                }
                agent_aircraft.append(config)
        except Exception as me:
            original_print(f"Error mapping flights_data: {me}")
            agent_routes = None
            agent_aircraft = None

    # Патчим load_environment_data в the_two_planes
    def patched_load():
        with open(data_files["prohibited_zones.json"], "r", encoding="utf-8") as f:
            prohibited = json.load(f)
        with open(data_files["allowed_to_use_waypoints.json"], "r", encoding="utf-8") as f:
            wp_list = json.load(f)
        with open(data_files["weather_data.json"], "r", encoding="utf-8") as f:
            weather = json.load(f)
        with open(data_files["simulation_result.json"], "r", encoding="utf-8") as f:
            oncoming = json.load(f)

        allowed_wp = {w["name"]: w for w in wp_list}
        weather_db = the_two_planes.WeatherDB(weather)
        return airports_list, prohibited, allowed_wp, weather_db, oncoming

    the_two_planes.load_environment_data = patched_load
    
    # Запускаем инференс
    try:
        # Загружаем данные окружения (аэропорты, зоны, погоду и т.д.)
        airports, prohibited, allowed_wp, weather_db, oncoming = the_two_planes.load_environment_data()
        
        routes = agent_routes if agent_routes is not None else the_two_planes.AGENT_ROUTES
        aircrafts = agent_aircraft if agent_aircraft is not None else the_two_planes.AGENT_AIRCRAFT
        
        agents = []
        for i in range(len(routes)):
            # Загружаем конкретные файлы весов q_table_agent*.pkl
            model_base = os.path.join(agent_dir, f"q_table_agent{i+1}.pkl")
            
            if os.path.exists(model_base):
                pkl_path = model_base
                original_print(f"Loading weights for agent {i+1}: {pkl_path}")
            else:
                raise FileNotFoundError(f"No weights found for agent {i+1} in {agent_dir}")
                
            agent = the_two_planes.QLearningAgent.from_file(pkl_path)
            agents.append(agent)
            
        results = the_two_planes._run_greedy(
            agents=agents,
            airports=airports,
            prohibited=prohibited,
            allowed_wp=allowed_wp,
            weather_db=weather_db,
            oncoming=oncoming,
            save=Path(agent_dir),
            agent_routes=routes,
            agent_aircraft=aircrafts
        )
        
        # Сохраняем результат в two_planes_route.json в корне проекта
        out_path = os.path.join(base_dir, 'two_planes_route.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        return results
    except Exception as e:
        original_print(f"Error during agent inference: {e}")
        import traceback
        original_print(traceback.format_exc())
        return []
    finally:
        builtins.print = original_print

if __name__ == "__main__":
    run_agent_inference()
