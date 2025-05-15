# api/api_app.py

from flask import Flask, request, jsonify
import joblib

# بارگذاری مدل XGBoost از پوشه models/
model = joblib.load("models/xgboost_model.pkl")

# ساخت API
app = Flask(__name__)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST':
            data = request.get_json(force=True)
        else:  # GET
            data = request.args

        # بررسی وجود features
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" parameter'}), 400

        # دریافت لیست ویژگی‌ها
        if request.method == 'GET':
            features_str = data['features']
            features = list(map(float, features_str.split(',')))
        else:
            features = data['features']

        # بررسی تعداد ویژگی‌ها
        if len(features) != 30:
            return jsonify({'error': f'Expected 30 features, got {len(features)}'}), 400

        # پیش‌بینی
        prediction = model.predict([features])
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'API is working properly!'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)