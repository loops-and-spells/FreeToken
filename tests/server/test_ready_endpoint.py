"""/ready: strict readiness probe -- 200 only when the engine is serving.

/health stays 200 through the whole lifecycle (loading/ok/error), which is
right for liveness but wrong for a router/load-balancer probe; /ready is the
one that flips with maintenance_state.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from freetoken.server.control_api import register_control_routes


def _client(state) -> TestClient:
    app = FastAPI(version="test")
    register_control_routes(app, lambda: state)
    return TestClient(app)


def _state(mstate: str, fatal: str | None = None):
    return SimpleNamespace(
        maintenance_state=mstate,
        fatal_error=fatal,
        instance_id="i",
        config=SimpleNamespace(served_model_name="m"),
        load_progress=None,
        ready_monotonic=None,
    )


def test_ready_503_while_loading():
    r = _client(_state("loading")).get("/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "loading"


def test_ready_200_when_serving():
    r = _client(_state("serving")).get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_503_on_fatal_error():
    r = _client(_state("serving", fatal="boom")).get("/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "error"


def test_health_stays_200_through_lifecycle():
    for st in ("loading", "serving"):
        assert _client(_state(st)).get("/health").status_code == 200
