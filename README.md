# Audio Transcript Service

Production-oriented Flask API for audio transcription with:

- Groq remote transcription with multiple API keys
- Mistral remote transcription with multiple API keys
- cross-provider round robin across Groq and Mistral
- local `whisper.cpp` server fallback at `http://127.0.0.1:8334`
- Postgres-backed durable job metadata
- Parquet-backed transcript artifact storage
- Redis-backed async job processing and transient runtime state
- worker/web separation
- normalized transcript output with file metadata and provider attempt history

## Architecture

The codebase is now organized as a package under `src/audio_transcript/`:

- `api/` Flask app factory, routes, auth, and JSON error handling
- `domain/` shared models and exceptions
- `services/` orchestration, routing, audio metadata, chunking, and transcript merge logic
- `infra/` providers, Postgres repository, Redis queue/runtime state, and Parquet artifact storage
- `worker/` background job runner

## Requirements

- Python 3.10+
- PostgreSQL
- Redis
- `ffmpeg` and `ffprobe`

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Runtime-only install:

```bash
pip install .
```

## Configuration

Create a `.env` file:

```env
SERVICE_API_KEY=change-me
DATABASE_URL=postgresql://postgres:password@127.0.0.1:5432/postgres
GROQ_API_KEYS=gsk_key_1,gsk_key_2

REDIS_URL=redis://127.0.0.1:6379/0
QUEUE_NAME=audio-transcript:jobs
STORAGE_ROOT=./data
TRANSCRIPT_DATASET_ROOT=./transcript_dataset

MISTRAL_API_KEYS=ms_key_1,ms_key_2

GROQ_MODEL=whisper-large-v3
MISTRAL_MODEL=voxtral-mini-latest

WHISPER_CPP_BASE_URL=http://127.0.0.1:8334
WHISPER_CPP_MODEL_PATH=
WHISPER_CPP_TEMPERATURE=0.0
WHISPER_CPP_TEMPERATURE_INC=0.2

REQUEST_TIMEOUT_SEC=300
PROVIDER_MAX_RETRIES=3
CHUNK_DURATION_SEC=600
CHUNK_OVERLAP_SEC=5
MAX_FILE_SIZE_MB=25
MAX_PARALLEL_CHUNKS=3
JOB_RETENTION_DAYS=7
DB_POOL_MIN_SIZE=2
DB_POOL_MAX_SIZE=10
DB_POOL_TIMEOUT_SEC=30
LOG_LEVEL=INFO
```

Required variables:

| Variable | Description |
|----------|-------------|
| `SERVICE_API_KEY` | API key required for authenticated API routes |
| `DATABASE_URL` | PostgreSQL connection URL using `postgres` or `postgresql` |
| `GROQ_API_KEYS` or `MISTRAL_API_KEYS` | At least one remote provider key must be configured |

Optional variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis connection for queue and runtime state |
| `QUEUE_NAME` | `audio-transcript:jobs` | Redis queue key prefix |
| `STORAGE_ROOT` | `./data` | Uploaded file storage directory |
| `TRANSCRIPT_DATASET_ROOT` | `./transcript_dataset` | Transcript parquet dataset directory |
| `CHUNK_DURATION_SEC` | `600` | Chunk duration for large audio |
| `CHUNK_OVERLAP_SEC` | `5` | Overlap between chunks |
| `MAX_FILE_SIZE_MB` | `25` | Single-request size threshold before chunking |
| `MAX_PARALLEL_CHUNKS` | `3` | Parallel chunk transcriptions |
| `REQUEST_TIMEOUT_SEC` | `300` | Provider request timeout |
| `PROVIDER_MAX_RETRIES` | `3` | Worker retry limit |
| `DB_POOL_MIN_SIZE` | `2` | Postgres pool minimum size |
| `DB_POOL_MAX_SIZE` | `10` | Postgres pool maximum size |
| `DB_POOL_TIMEOUT_SEC` | `30` | Postgres pool acquisition timeout |
| `WHISPER_CPP_BASE_URL` | `http://127.0.0.1:8334` | whisper.cpp server base URL |
| `LOG_LEVEL` | `INFO` | App logging level |

## Run

Start the API:

```bash
.venv/bin/python main.py
```

Start the worker in a separate terminal:

```bash
.venv/bin/python worker.py
```

## API

### Health

```bash
curl http://127.0.0.1:8000/v1/health
```

### Provider Status

```bash
curl http://127.0.0.1:8000/v1/providers/status \
  -H "X-API-Key: change-me"
```

### Create Job

```bash
curl http://127.0.0.1:8000/v1/jobs \
  -H "X-API-Key: change-me" \
  -F file=@sample.wav
```

Optional form fields:

- `model`
- `chunk_duration_sec`
- `chunk_overlap_sec`

### Job Status

```bash
curl http://127.0.0.1:8000/v1/jobs/<job_id> \
  -H "X-API-Key: change-me"
```

### Job List / Search

```bash
curl "http://127.0.0.1:8000/v1/jobs?status=succeeded&provider=groq&search=trading" \
  -H "X-API-Key: change-me"
```

### Job Result

```bash
curl http://127.0.0.1:8000/v1/jobs/<job_id>/result \
  -H "X-API-Key: change-me"
```

## whisper.cpp Fallback

The final fallback provider uses the local server contract:

```bash
curl 127.0.0.1:8334/inference \
  -H "Content-Type: multipart/form-data" \
  -F file="@<file-path>" \
  -F temperature="0.0" \
  -F temperature_inc="0.2" \
  -F response_format="json"
```

Optional startup model load:

```bash
curl 127.0.0.1:8334/load \
  -H "Content-Type: multipart/form-data" \
  -F model="<path-to-model-file>"
```

If `WHISPER_CPP_MODEL_PATH` is set, the service loads that model at startup.

## Testing

```bash
.venv/bin/pytest -q
```

## Notes

- Groq and Mistral are used as first-class remote providers.
- `whisper.cpp` is only used after remote providers fail or are exhausted.
- Job metadata is stored in Postgres.
- Completed transcript artifacts are written as partitioned Parquet datasets under `TRANSCRIPT_DATASET_ROOT`.
- Redis is only used for queueing, locks, cooldowns, and other short-lived runtime state.
- Do not commit real API keys. Move exposed secrets into environment variables and rotate them if they were shared publicly.
