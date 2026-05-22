FROM python:3.11-slim

WORKDIR /app

# Railway usa la raíz del repo como contexto — paths con prefijo backend/
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
