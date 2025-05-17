# تصویر پایه - Python 3.9
FROM python:3.11-slim

# تنظیمات محیطی
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# دایرکتوری کاری داخل دوکر
WORKDIR /app

# کپی کردن فایل‌های ضروری
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 

# کپی کردن API و مدل
COPY api/ /app/api/
COPY models/ /app/models/

# اجرای برنامه
CMD ["python", "api/api_app.py"]