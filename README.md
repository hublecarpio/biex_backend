# BIEX Backend – Agente Tutor Cognitivo

API FastAPI + LangGraph para el tutor cognitivo BIEX.

## Requisitos

- Python 3.11+ o Docker
- Variables de entorno (ver `.env.example`)

## Clonar en otro servidor

```bash
git clone https://github.com/TU_USUARIO/biex_backend.git
cd biex_backend
cp .env.example .env
# Editar .env con tus claves (GEMINI_API_KEY, SUPABASE_*, API_KEY_SECRET)
```

## Ejecución

**Con Docker (recomendado):**
```bash
docker compose up --build
```
API en `http://localhost:8000`.

**Local:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health` – Health check
- `POST /api/v1/chat` – Chat (header `X-API-Key` obligatorio). Body: `user_id`, `message`, `session_id`, `stream` (bool).
