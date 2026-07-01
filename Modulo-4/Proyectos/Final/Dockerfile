FROM python:3.11-slim

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --timeout 120 -r requirements-deploy.txt

COPY app/ app/
COPY models/ models/
COPY .env.example .env.example

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
