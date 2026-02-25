from flask import Flask, send_from_directory, jsonify

app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/cities")
def cities():
    # Простейший набор городов (можно заменить на БД)
    data = [
        {"name": "Москва", "lat": 55.7558, "lon": 37.6173},
        {"name": "Санкт‑Петербург", "lat": 59.9375, "lon": 30.3086},
        {"name": "Новосибирск", "lat": 55.0084, "lon": 82.9357},
        {"name": "Екатеринбург", "lat": 56.8389, "lon": 60.6057},
        {"name": "Владивосток", "lat": 43.1155, "lon": 131.8855},
    ]
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)