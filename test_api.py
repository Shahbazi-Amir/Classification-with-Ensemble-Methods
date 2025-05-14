import requests

# آدرس API
url = "http://127.0.0.1:5001/predict"

# داده‌های ورودی (Features)
input_data = {
    "features": [5.1, 3.5, 1.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 30 ویژگی
}

try:
    # ارسال درخواست POST به API
    response = requests.post(url, json=input_data)

    # چاپ پاسخ
    print("Response Status Code:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    print("Error:", str(e))