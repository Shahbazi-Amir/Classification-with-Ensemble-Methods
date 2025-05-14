import joblib

# Load the model
model = joblib.load("xgboost_model.pkl")

# Test prediction with 30 features
features = [5.1, 3.5, 1.4, 0.2] + [0.0] * 26  # 4 ویژگی اصلی + 26 صفر برای تکمیل به 30 ویژگی
prediction = model.predict([features])
print("Prediction:", prediction)