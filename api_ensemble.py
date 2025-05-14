# Import necessary libraries
from flask import Flask, request, jsonify
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load("xgboost_model.pkl")  # Replace with your model file name

# Define the prediction endpoint
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # بررسی روش درخواست
        if request.method == 'POST':
            data = request.json  # برای POST، داده‌ها از JSON خوانده می‌شوند
        elif request.method == 'GET':
            data = request.args  # برای GET، داده‌ها از پارامترهای URL خوانده می‌شوند

        # بررسی وجود پارامتر 'features'
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" parameter'}), 400

        # تبدیل داده‌ها به لیست
        features = list(map(float, data['features'].split(','))) if request.method == 'GET' else data['features']

        # بررسی طول لیست 'features'
        if len(features) != 30:
            return jsonify({'error': 'Expected 30 features, got {}'.format(len(features))}), 400

        # پیش‌بینی با مدل
        prediction = model.predict([features])
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Add a simple test endpoint
@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'API is working!'})

# Run the app on port 5001
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change the port to 5001