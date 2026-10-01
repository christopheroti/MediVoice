FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + trained model only
COPY app ./app
COPY models/medivoice_triage_model.joblib ./models/medivoice_triage_model.joblib

# Hosts like Render set $PORT; default to 8000 locally
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
