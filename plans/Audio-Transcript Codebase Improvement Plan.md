Audio-Transcript Codebase Improvement Plan

This plan addresses code duplication, architectural gaps, reliability issues, and technical debt discovered during the codebase review. The architecture is fundamentally sound (clean layered design with Flask API → Services → Infrastructure → Domain), but has accumulated debt in provider implementations, error handling, and operational concerns.

Phase 1: Critical Fixes (Reliability & Security)

1. Add timing-safe API key comparison in `src/audio_transcript/api/auth.py` - replace string `==` with `secrets.compare_digest()` to prevent timing attacks

2. Add file size validation at upload time in `src/audio_transcript/api/routes.py` - check `request.content_length` before writing to disk, not just at transcription time

3. Implement connection pooling in `src/audio_transcript/infra/repository.py` - replace per-call `_connect()` with `psycopg2.pool.ThreadedConnectionPool`

4. Add dead letter queue for failed jobs in `src/audio_transcript/worker/runner.py` - implement retry counter (3 attempts) with exponential backoff, then move to DLQ instead of dropping

5. Suppress internal error details in `src/audio_transcript/api/errors.py` - log full exception but return generic message to client

Phase 2: Code Consolidation

6. Extract `RemoteAPIProvider` base class - create new `src/audio_transcript/infra/providers/base.py` abstract class containing shared HTTP request logic from `src/audio_transcript/infra/providers/groq.py` and `src/audio_transcript/infra/providers/mistral.py`. Subclasses only define URL, provider name, and response normalization

7. Extract segment parsing helper - deduplicate segment extraction logic at `src/audio_transcript/infra/providers/groq.py`, `src/audio_transcript/infra/providers/mistral.py`, and `src/audio_transcript/infra/providers/whisper_cpp.py` into shared `parse_segments(json_data) -> list[TranscriptSegment]` function

8. Consolidate entry points - remove `main.py`, keep only `app.py` as the single WSGI entry point

9. Normalize `provider_data` handling - fix inconsistent `"tokens"` exclusion between providers. Define explicit schema for stored provider metadata

10. Create dedicated `AuthenticationError` - replace use of `ValidationError` in auth and remove fragile string checks in `errors.py`

Phase 3: Performance & Scalability

11. Optimize job attempts persistence - replace DELETE+INSERT pattern in `src/audio_transcript/infra/repository.py` with `ON CONFLICT DO UPDATE` upsert

12. Add parallel chunk transcription - modify `src/audio_transcript/services/transcription.py` to use `concurrent.futures.ThreadPoolExecutor` for processing multiple chunks simultaneously (configurable concurrency)

13. Add streaming Parquet reads - modify `src/audio_transcript/infra/storage.py` to use `pyarrow.parquet.ParquetFile` with row-group iteration for large transcripts

Phase 4: Error Handling & Robustness

14. Wrap ffmpeg/ffprobe errors - add try/except in `src/audio_transcript/services/audio.py` to convert subprocess errors to domain `AudioProcessingError`

15. Add storage transaction support - wrap Parquet writes in `src/audio_transcript/infra/storage.py` with temp file + atomic rename for safety

16. Move import to module level - fix delayed import in `src/audio_transcript/infra/repository.py` (move `from ..domain.models import JobPayload` to top)

Phase 5: Configuration & Dependencies

17. Consolidate dependency files - remove `requirements.txt`, use only `pyproject.toml`. Add `pip-tools` workflow if pinning needed

18. Add URL validation in `src/audio_transcript/config.py` - validate `redis_url`, `database_url`, `whisper_cpp_base_url` formats at startup

19. Add default `QUEUE_NAME` in `src/audio_transcript/config.py` - default to `"transcription_jobs"`

Phase 6: Testing Improvements

20. Add repository integration tests - test `PostgresJobRepository` against real Postgres (use `pytest-docker` or `testcontainers`)

21. Add Redis integration tests - test `RedisJobQueue` and `RedisRuntimeState` with real Redis

22. Add ffmpeg-dependent tests - mark with `@pytest.mark.ffmpeg`, test `AudioChunker.chunk_audio()` with sample audio files

23. Add concurrent access tests - test `ProviderKeyPool` and job locks under concurrent load

24. Extract shared test fixtures - consolidate duplicated `FakeInspector` and similar helpers into `tests/conftest.py`

Phase 7: Documentation & Observability

25. Add docstrings - document all public classes/methods, especially `TranscriptionService.process_job`, `ProviderKeyPool`, provider classes

26. Enhance README - add API error response documentation, rate limiting behavior, deployment requirements

27. Add health check depth - extend `/health` endpoint to verify ffmpeg/ffprobe availability

28. Add structured logging - include job_id, provider, duration in log entries for traceability

Verification Steps

- Run existing tests: `pytest tests/`
- Run new integration tests: `pytest tests/ -m integration` (requires Docker)
- Manual API test: submit audio, verify transcription completes
- Load test: submit 10 concurrent jobs, verify no race conditions
- Security: verify error responses don't leak paths/stack traces

Decisions & Scope

- Prioritize Phase 1 (security/reliability) before refactoring
- Keep synchronous Flask worker (async migration is a larger effort)
- Use `psycopg2` pooling rather than introducing SQLAlchemy for minimal change
- Not adding Pydantic validation in this initial pass to limit scope

Notes

- Consider splitting larger refactors (provider base class, async worker) into separate PRs per service boundary
- Keep changes small and verifiable; add tests for each behavior change

If you want, I can now:
- open a PR scaffold with the first changes (timing-safe auth, file size validation, and error suppression), or
- refine this plan into a prioritized sprint backlog (3-5 tasks per sprint).

Which would you like next?
