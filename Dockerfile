# BIEX Backend - Python 3.11+
FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema (mínimas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libpq5 \
  && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código (.env se inyecta en runtime vía docker-compose)
COPY app/ ./app/

# Puerto
EXPOSE 8000

# Usuario no root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
