# api/testapi.py

import requests

url = "http://127.0.0.1:5001/predict"

input_data = {
    "features": [
        13.5, 14.36, 87.46, 566.3, 0.09779, 0.1267, 0.1976, 0.1065, 0.1812, 0.06163,
        0.822, 1.323, 5.052, 39.73, 0.0078, 0.02934, 0.03266, 0.01672, 0.01958, 0.003636,
        15.68, 19.69, 99.74, 711.2, 0.1324, 0.2465, 0.3374, 0.1806, 0.2809, 0.0786
    ]
}

try:
    response = requests.post(url, json=input_data)
    print("Status Code:", response.status_code)
    print("Response Body:", response.json())
except Exception as e:
    print("Error during request:", str(e))