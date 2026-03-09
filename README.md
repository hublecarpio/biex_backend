# BIEX Backend – Agente Tutor Cognitivo

API FastAPI + LangGraph para el tutor cognitivo BIEX.

## Documentación

- **[Flujo del agente y herramientas](docs/flujo-agente.md)** – Diagrama detallado de la lógica del agente y uso de Supabase, Gemini, Postgres y parser.

## Requisitos

- Python 3.11+ o Docker
- Variables de entorno (ver `.env.example`)

## Clonar en otro servidor

```bash
git clone https://github.com/hublecarpio/biex_backend.git
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
- `GET /api/v1/images/{job_id}` – Polling de imágenes generadas en background.

**Body (JSON) de `/api/v1/chat`:**

| Campo             | Tipo   | Descripción                   |
| ----------------- | ------ | ----------------------------- |
| `mensaje`         | `str`  | Texto del usuario             |
| `id_conversation` | `str`  | ID de conversación            |
| `id_user`         | `str`  | ID del usuario                |
| `stream`          | `bool` | Opcional, `false` por defecto |

**Respuesta (objeto estructurado):**

| Campo            | Tipo        | Descripción                                                                   |
| ---------------- | ----------- | ----------------------------------------------------------------------------- |
| `mensajes`       | `list[str]` | Segmentos de texto (sin markdown, sin URLs de imagen)                         |
| `images`         | `list[str]` | URLs de imágenes disponibles de inmediato                                     |
| `images_count`   | `int`       | Total de imágenes (disponibles + pendientes)                                  |
| `images_pending` | `int`       | Imágenes aún generándose en background                                        |
| `images_job_id`  | `str\|null` | ID del job para polling vía `GET /api/v1/images/{job_id}`                     |
| `current_phase`  | `str`       | Fase pedagógica activa: `generativa`, `vicario`, `socratico`, `metacognicion` |

## Fases pedagógicas

| Fase            | Descripción                                            |
| --------------- | ------------------------------------------------------ |
| `generativa`    | Explicación educativa con RAG e imágenes en background |
| `vicario`       | Empatía y pensamiento en voz alta (frustración alta)   |
| `socratico`     | Preguntas de pensamiento crítico                       |
| `metacognicion` | Cierre y reflexión metacognitiva                       |

El **supervisor** determina la fase en cada turno usando 6 reglas determinísticas basadas en comprensión, frustración y contadores de sesión — sin LLM.

## Sesión y persistencia

Cada conversación mantiene un `session_state` en Supabase con los siguientes campos clave:

| Campo                         | Descripción                                        |
| ----------------------------- | -------------------------------------------------- |
| `session_phase`               | Fase pedagógica actual                             |
| `interaction_count`           | Número de interacciones en la sesión               |
| `comprehension_history`       | Últimos 20 scores de comprensión (0-100)           |
| `frustration_history`         | Últimos 20 niveles de frustración (0-10)           |
| `engagement_history`          | Últimos 20 scores de engagement (0-100)            |
| `zdp_level`                   | Zona de desarrollo próximo estimada (0-100)        |
| `current_topic`               | Tema educativo actualmente activo                  |
| `topics_covered`              | Lista de los últimos 20 temas cubiertos            |
| `misconceptions`              | Últimos 10 malentendidos detectados                |
| `vicario_triggers`            | Cantidad de veces que se activó la fase vicario    |
| `socratic_questions_answered` | Preguntas socráticas respondidas en la fase actual |
| `socratic_correct_answers`    | Respuestas correctas en socrático (score ≥ 60)     |
| `phase_transitions`           | Historial de transiciones entre fases              |

> Los contadores `socratic_questions_answered` y `socratic_correct_answers` se resetean a 0 al salir de la fase socrática.

## Variables de entorno

| Variable                    | Descripción                                     |
| --------------------------- | ----------------------------------------------- |
| `GEMINI_API_KEY`            | API key de Google Gemini                        |
| `SUPABASE_URL`              | URL del proyecto Supabase                       |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key de Supabase                    |
| `API_KEY_SECRET`            | Secreto para `X-API-Key`                        |
| `DATABASE_URL`              | (Opcional) Postgres para checkpointer LangGraph |
