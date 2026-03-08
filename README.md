# BIEX Backend – Agente Tutor Cognitivo

API FastAPI + LangGraph para el tutor cognitivo BIEX.

## Documentación

- **[Flujo del agente y herramientas](docs/flujo-agente.md)** – Diagrama de la lógica del agente y uso de Supabase, Gemini, Postgres y parser.

## Requisitos

- Python 3.11+ o Docker
- Variables de entorno (ver `.env.example`)

## Clonar en otro servidor

```bash
git clone https://github.com/TU_USUARIO/biex_backend.git
cd biex_backend
cp .env.example .env
# Editar .env con tus claves (GEMINI_API_KEY, SUPABASE_*, API_KEY_SECRET, opcional DATABASE_URL para memoria Postgres)
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
- `POST /api/v1/chat` – Chat (header `X-API-Key` obligatorio).

**Body (JSON):**

- `mensaje` (str) – Texto del usuario
- `id_conversation` (str) – ID de conversación
- `id_user` (str) – ID del usuario
- `stream` (bool, opcional) – `false` por defecto

**Respuesta (objeto estructurado):**

- `mensajes` – Lista de segmentos de texto (sin markdown, sin URLs de imagen)
- `images` – URLs de imágenes disponibles de inmediato
- `images_count` – Número total de imágenes (incluyendo pendientes)
- `images_pending` – Imágenes generándose en background (frontend muestra placeholders)
- `images_job_id` – ID del job para polling via `GET /api/v1/images/{job_id}` (null si no hay imágenes)
- `current_phase` – Fase actual del tutor (`generativa`, `vicaria`, `socratica`, `metacognicion`)

Si se define `DATABASE_URL` (Postgres), la memoria por conversación se persiste en la BD (tablas de LangGraph).
