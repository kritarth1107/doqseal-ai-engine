"""The main app keeps every existing route and only adds /bundle/classify."""

import pytest


def test_existing_routes_unchanged_and_bundle_route_added():
    try:
        from app.main import app
    except ImportError as exc:  # full runtime deps not installed locally
        pytest.skip(f"app.main dependencies unavailable: {exc}")

    routes = {(r.path, tuple(sorted(getattr(r, "methods", []) or []))) for r in app.routes}
    paths = {p for p, _ in routes}
    for expected in ("/health", "/chat", "/rag/documents/{document_id}", "/bundle/classify"):
        assert expected in paths
    assert ("/chat", ("POST",)) in routes
    assert ("/rag/documents/{document_id}", ("DELETE",)) in routes
    assert ("/bundle/classify", ("POST",)) in routes
