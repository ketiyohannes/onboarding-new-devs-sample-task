FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

COPY . .

CMD ["python", "repository_after/scalar_tensor_ad.py"]
