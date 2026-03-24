"""Flask app factory."""

from __future__ import annotations

import uuid

import redis
from flask import Flask, g, request

from ..config import Settings
from ..domain.errors import ConfigurationError
from ..logging_utils import clear_request_context, configure_logging, set_request_context
from ..infra.runtime_state import RedisRuntimeState
from ..services.audio import AudioChunker, AudioInspector
from ..services.router import ProviderKeyPool, ProviderRouter
from ..services.transcription import DirectoryScanService, RuntimeDependencies, TranscriptionService
from ..infra.providers.groq import GroqProvider
from ..infra.providers.mistral import MistralProvider
from ..infra.providers.whisper_cpp import WhisperCppProvider
from ..infra.queue import QueueBackend, RedisQueueBackend
from ..infra.repository import JobRepository, PostgresJobRepository
from ..infra.storage import TranscriptArtifactStore
from .errors import register_error_handlers
from .routes import bp


def build_runtime(settings: Settings):
    """Construct the runtime dependency graph."""
    configure_logging(settings.log_level, settings.log_format)
    redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    if not settings.database_url:
        raise ConfigurationError("DATABASE_URL is required")
    repository: JobRepository = PostgresJobRepository(
        settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        timeout_sec=settings.db_pool_timeout_sec,
    )
    queue: QueueBackend = RedisQueueBackend(redis_client, settings.queue_name)
    runtime_state = RedisRuntimeState(redis_client)
    artifact_store = TranscriptArtifactStore(settings.storage_root, settings.transcript_dataset_root)
    inspector = AudioInspector()
    chunker = AudioChunker(inspector)

    providers = {}
    remote_names = []
    if settings.groq_api_keys:
        providers["groq"] = GroqProvider(
            ProviderKeyPool("groq", settings.groq_api_keys, runtime_state=runtime_state),
            settings.groq_model,
            settings.request_timeout_sec,
        )
        remote_names.append("groq")
    if settings.mistral_api_keys:
        providers["mistral"] = MistralProvider(
            ProviderKeyPool("mistral", settings.mistral_api_keys, runtime_state=runtime_state),
            settings.mistral_model,
            settings.request_timeout_sec,
        )
        remote_names.append("mistral")

    fallback = WhisperCppProvider(
        settings.whisper_cpp_base_url,
        settings.request_timeout_sec,
        settings.whisper_cpp_temperature,
        settings.whisper_cpp_temperature_inc,
    )
    if settings.whisper_cpp_model_path:
        fallback.load_model(settings.whisper_cpp_model_path)

    service = TranscriptionService(
        RuntimeDependencies(
            settings=settings,
            repository=repository,
            artifact_store=artifact_store,
            runtime_state=runtime_state,
            router=ProviderRouter(remote_names),
            remote_providers=providers,
            fallback_provider=fallback,
            inspector=inspector,
            chunker=chunker,
        )
    )
    directory_scan_service = DirectoryScanService(repository, queue)
    return {
        "settings": settings,
        "repository": repository,
        "queue": queue,
        "artifact_store": artifact_store,
        "runtime_state": runtime_state,
        "service": service,
        "directory_scan_service": directory_scan_service,
        "providers": providers,
        "fallback_provider": fallback,
    }


def create_app(
    settings: Settings | None = None,
    *,
    repository=None,
    queue=None,
    artifact_store=None,
    runtime_state=None,
    service=None,
    directory_scan_service=None,
    providers=None,
    fallback_provider=None,
) -> Flask:
    """Create a Flask app."""
    app = Flask(__name__)
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level, settings.log_format)
    runtime = None
    if all(item is None for item in (repository, queue, artifact_store, runtime_state, service, directory_scan_service, providers, fallback_provider)):
        runtime = build_runtime(settings)
    else:
        runtime = {
            "settings": settings,
            "repository": repository,
            "queue": queue,
            "artifact_store": artifact_store,
            "runtime_state": runtime_state,
            "service": service,
            "directory_scan_service": directory_scan_service,
            "providers": providers or {},
            "fallback_provider": fallback_provider,
        }

    if runtime["repository"] is None or runtime["queue"] is None or runtime["artifact_store"] is None or runtime["runtime_state"] is None or runtime["service"] is None:
        raise ValueError("repository, queue, artifact_store, runtime_state, and service are required when overriding app runtime")
    if runtime.get("directory_scan_service") is None:
        runtime["directory_scan_service"] = DirectoryScanService(runtime["repository"], runtime["queue"])

    app.config.update(runtime)

    @app.before_request
    def bind_request_id() -> None:
        request_id = request.headers.get("X-Request-ID") or request.headers.get("X-Request-Id")
        request_id = request_id or str(uuid.uuid4())
        g.request_id = request_id
        g.request_id_token = set_request_context(request_id)

    @app.after_request
    def attach_request_id(response):
        request_id = getattr(g, "request_id", None)
        if request_id:
            response.headers["X-Request-ID"] = request_id
        return response

    @app.teardown_request
    def cleanup_request_context(exception=None) -> None:
        clear_request_context(getattr(g, "request_id_token", None))

    app.register_blueprint(bp)
    register_error_handlers(app)
    return app
