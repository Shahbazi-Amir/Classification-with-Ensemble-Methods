# 🚀 Breast Cancer Prediction API Documentation

This section describes how to run and test the Flask-based API for breast cancer prediction using a trained XGBoost model.

---

## 📁 Project Structure (API Part)

```
CLASSIFICATION_ENSEMBLE_PROJECT/
├── models/
│   └── xgboost_model.pkl
├── api/
│   ├── api_app.py     ← This file
│   └── testapi.py     ← Test script
```

---

## 🔧 Requirements

Install dependencies:

```bash
pip install flask joblib scikit-learn xgboost
```

---

## 🚀 Running the API

1. Navigate to the `api` folder:

```bash
cd api
```

2. Run the API:

```bash
python api_app.py
```

The API will start at:

```
http://127.0.0.1:5001/
```

---

## 🧪 Testing the API

### Option 1: GET Request (via Browser or Postman)

Paste this URL in your browser or Postman:

```
http://127.0.0.1:5001/predict?features=13.5,14.36,87.46,566.3,0.09779,0.1267,0.1976,0.1065,0.1812,0.06163,0.822,1.323,5.052,39.73,0.0078,0.02934,0.03266,0.01672,0.01958,0.003636,15.68,19.69,99.74,711.2,0.1324,0.2465,0.3374,0.1806,0.2809,0.0786
```

Expected Response:

```json
{
  "prediction": 0
}
```

---

### Option 2: POST Request (with JSON)

Use `curl` or Postman:

```bash
curl -X POST http://127.0.0.1:5001/predict \
     -H "Content-Type: application/json" \
     -d '{
         "features": [13.5,14.36,87.46,566.3,0.09779,0.1267,0.1976,0.1065,0.1812,0.06163,0.822,1.323,5.052,39.73,0.0078,0.02934,0.03266,0.01672,0.01958,0.003636,15.68,19.69,99.74,711.2,0.1324,0.2465,0.3374,0.1806,0.2809,0.0786]
       }'
```

---

### Option 3: Python Script (`testapi.py`)

Run the following code in a Python script or notebook cell:

```python
import requests

url = "http://127.0.0.1:5001/predict"

input_data = {
    "features": [
        13.5, 14.36, 87.46, 566.3, 0.09779, 0.1267, 0.1976, 0.1065, 0.1812, 0.06163,
        0.822, 1.323, 5.052, 39.73, 0.0078, 0.02934, 0.03266, 0.01672, 0.01958, 0.003636,
        15.68, 19.69, 99.74, 711.2, 0.1324, 0.2465, 0.3374, 0.1806, 0.2809, 0.0786
    ]
}

response = requests.post(url, json=input_data)
print("Status Code:", response.status_code)
print("Response Body:", response.json())
```

Expected Output:

```
Status Code: 200
Response Body: {'prediction': 0}
```

---

## 📌 Notes

- Make sure the model file `models/xgboost_model.pkl` exists in the root directory.
- The input must contain exactly **30 float features**.
- The output is either:
  - `0`: Benign (non-cancerous)
  - `1`: Malignant (cancerous)


