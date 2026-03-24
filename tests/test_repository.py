from contextlib import contextmanager

from audio_transcript.infra.repository import PostgresJobRepository


class FakeCursor:
    def execute(self, query, params=None):
        self.query = query
        self.params = params

    def fetchone(self):
        return {"ok": 1}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def cursor(self):
        return FakeCursor()


class FakePool:
    def __init__(self):
        self.stats = {"pool_size": 4, "pool_available": 3}

    def get_stats(self):
        return self.stats


def test_postgres_repository_healthcheck_includes_pool_stats():
    repository = PostgresJobRepository.__new__(PostgresJobRepository)
    repository._pool = FakePool()

    @contextmanager
    def fake_connection():
        yield FakeConnection()

    repository._connection = fake_connection

    status = repository.healthcheck()

    assert status == {
        "postgres": "ok",
        "postgres_pool_size": "4",
        "postgres_pool_available": "3",
    }
