import os
import json
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# инициализирую путь к бинарному артефакту модели 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_rf.pkl')

# загрузка сериализованного графа вычислений (Pipeline)
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    print(f"Системное предупреждение: конфигурационный артефакт {MODEL_PATH} не удалось загрузить. Детали: {e}")

@app.route('/', methods=['GET'])
def index():
    # инициализирую базовый вектор признаков для нагрузочного тестирования
    default_payload = json.dumps({
        "visit_number_clipped": 2,
        "visit_month": 5,
        "visit_dayofweek": 4,
        "visit_hour": 19,
        "is_paid": 1,
        "is_social": 1,
        "device_category": "mobile",
        "source_top": "ZpYIoDJMcFzVoPFsHGJL",
        "city_top": "Moscow"
    }, indent=4)
    return render_template('index.html', default_payload=default_payload, model_status=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    # готовность вычислительного ядра
    if not model_loaded:
        return jsonify({"error": "Внутренняя ошибка сервера: Артефакт модели недоступен для чтения."}), 500

    try:
        if request.is_json:
            data = request.get_json()
        else:
            raw_text = request.form.get('json_data', '')
            if not raw_text:
                try:
                    data = json.loads(request.data)
                except:
                    return jsonify({"error": "Пустое тело запроса или неверный заголовок Content-Type."}), 400
            else:
                data = json.loads(raw_text)

        df = pd.DataFrame([data])
        
        prob = model.predict_proba(df)[0][1]
        pred_class = model.predict(df)[0]
        
        return jsonify({
            "status": "success",
            "probability": round(prob, 4),
            "predicted_class": int(pred_class)
        })

    except json.JSONDecodeError:
        return jsonify({"error": "Синтаксическая ошибка: переданный вектор не является валидным JSON-объектом."}), 400
    except Exception as e:
        return jsonify({"error": f"Глобальное исключение в модуле предсказания: {str(e)}"}), 500

if __name__ == '__main__':
    # активация слушателя на локальном интерфейсе
    app.run(host='0.0.0.0', port=5000, debug=True)