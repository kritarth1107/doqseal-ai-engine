import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.bundle import classify as classify_mod
from app.bundle import router as router_mod
from app.config import settings

TOKEN = "test-service-token"
ORG_A = "org-a"
ORG_B = "org-b"

SLOTS = [
    {"key": "pan", "label": "PAN card", "hints": ["income tax", "permanent account number"]},
    {"key": "salary_slip", "label": "Salary slip", "hints": ["payslip", "net pay"]},
    {"key": "trf", "label": "Test requisition form", "hints": ["lab", "tests requested"]},
]

# Documents that exist, keyed by (organisationId, documentId).
DOCS = {(ORG_A, "doc-a1"), (ORG_A, "doc-a2"), (ORG_B, "doc-b1")}


@pytest.fixture()
def model_calls(monkeypatch):
    calls = []

    def fake_model(messages):
        calls.append(messages)
        return {
            "slot": "pan",
            "confidence": 0.93,
            "reasons": ["Has PAN layout", "Income Tax Department header"],
            "alternatives": [{"slot": "salary_slip", "confidence": 0.04}, {"slot": "bogus", "confidence": 0.5}],
            "keyFields": {
                "full_name": "Ravi Kumar",
                "date_of_birth": "1990-04-12",
                "pan": "ABCDE1234F",
                "aadhaar_last4": "1234 5678 9012",
                "unexpected": "dropped",
            },
        }

    monkeypatch.setattr(classify_mod, "_call_model", fake_model)
    return calls


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(settings, "ai_engine_service_token", TOKEN)
    monkeypatch.setattr(
        router_mod,
        "document_belongs_to_org",
        lambda org, doc: (org, doc) in DOCS,
    )
    classify_mod.clear_cache()
    app = FastAPI()
    app.include_router(router_mod.router)
    return TestClient(app)


def body(org=ORG_A, doc="doc-a1", request_id="req-1", **extra):
    payload = {
        "requestId": request_id,
        "organisationId": org,
        "bundleId": "bundle-1",
        "documentId": doc,
        "slots": SLOTS,
        "document": {
            "documentType": "PAN card",
            "text": "INCOME TAX DEPARTMENT  Ravi Kumar  ABCDE1234F",
            "fields": {"name": "Ravi Kumar"},
        },
    }
    payload.update(extra)
    return payload


def headers(org=ORG_A, token=TOKEN):
    return {"X-Service-Token": token, "X-Organisation-Id": org}


def test_happy_path_returns_slot_confidence_reasons_and_key_fields(client, model_calls):
    res = client.post("/bundle/classify", json=body(), headers=headers())
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["slot"] == "pan"
    assert data["confidence"] == pytest.approx(0.93)
    assert data["reasons"][0] == "Has PAN layout"
    assert data["organisationId"] == ORG_A and data["documentId"] == "doc-a1"
    # Only requested key fields, Aadhaar reduced to last four digits.
    assert data["keyFields"] == {
        "full_name": "Ravi Kumar",
        "date_of_birth": "1990-04-12",
        "pan": "ABCDE1234F",
        "aadhaar_last4": "9012",
    }
    # Unknown alternative slots are dropped.
    assert data["alternatives"] == [{"slot": "salary_slip", "confidence": 0.04}]
    assert len(model_calls) == 1


def test_disabled_without_configured_token(client, model_calls, monkeypatch):
    monkeypatch.setattr(settings, "ai_engine_service_token", "")
    res = client.post("/bundle/classify", json=body(), headers=headers())
    assert res.status_code == 503
    assert model_calls == []


def test_wrong_or_missing_token_is_refused(client, model_calls):
    assert client.post("/bundle/classify", json=body(), headers=headers(token="nope")).status_code == 401
    assert client.post("/bundle/classify", json=body(), headers={"X-Organisation-Id": ORG_A}).status_code == 401
    assert model_calls == []


def test_org_header_must_match_body(client, model_calls):
    res = client.post("/bundle/classify", json=body(org=ORG_A), headers=headers(org=ORG_B))
    assert res.status_code == 400
    assert model_calls == []


def test_document_of_another_org_is_not_found(client, model_calls):
    # doc-b1 belongs to org B; org A must not be able to classify it.
    res = client.post("/bundle/classify", json=body(org=ORG_A, doc="doc-b1"), headers=headers(ORG_A))
    assert res.status_code == 404
    assert model_calls == []


def test_prompt_contains_only_this_document(client, model_calls):
    client.post("/bundle/classify", json=body(), headers=headers())
    other = body(org=ORG_B, doc="doc-b1", request_id="req-b")
    other["document"]["text"] = "ORG B SECRET PATIENT Sunita"
    other["document"]["fields"] = {"name": "Sunita"}
    client.post("/bundle/classify", json=other, headers=headers(ORG_B))
    assert len(model_calls) == 2
    first = json.dumps(model_calls[0])
    second = json.dumps(model_calls[1])
    assert "ORG B SECRET" not in first
    assert "Ravi Kumar" not in second
    # Content is fenced as untrusted data and the system prompt says so.
    assert model_calls[0][0]["role"] == "system"
    assert "untrusted" in model_calls[0][0]["content"]
    assert "<<<DOCUMENT" in model_calls[0][1]["content"]


def test_same_request_id_is_idempotent_and_scoped_per_org(client, model_calls):
    first = client.post("/bundle/classify", json=body(request_id="same"), headers=headers())
    again = client.post("/bundle/classify", json=body(request_id="same"), headers=headers())
    assert first.status_code == again.status_code == 200
    assert again.json()["cached"] is True
    assert len(model_calls) == 1
    # The same requestId from another org is a different cache entry.
    other = client.post(
        "/bundle/classify", json=body(org=ORG_B, doc="doc-b1", request_id="same"), headers=headers(ORG_B)
    )
    assert other.status_code == 200
    assert other.json()["cached"] is False
    assert len(model_calls) == 2


def test_unknown_slot_from_model_becomes_null(client, monkeypatch):
    monkeypatch.setattr(
        classify_mod,
        "_call_model",
        lambda messages: {"slot": "passport", "confidence": 0.99, "reasons": ["looks like a passport"]},
    )
    data = client.post("/bundle/classify", json=body(), headers=headers()).json()
    assert data["slot"] is None
    assert data["confidence"] == 0.0
    assert "not one of the template slots" in data["reasons"][0]


def test_model_failure_is_reported_as_retryable(client, monkeypatch):
    def boom(messages):
        raise classify_mod.ClassificationError("model returned HTTP 429", retryable=True)

    monkeypatch.setattr(classify_mod, "_call_model", boom)
    res = client.post("/bundle/classify", json=body(), headers=headers())
    assert res.status_code == 502
    assert res.json()["detail"]["retryable"] is True


def test_non_retryable_model_failure(client, monkeypatch):
    def boom(messages):
        raise classify_mod.ClassificationError("model is not configured", retryable=False)

    monkeypatch.setattr(classify_mod, "_call_model", boom)
    res = client.post("/bundle/classify", json=body(), headers=headers())
    assert res.status_code == 422
    assert res.json()["detail"]["retryable"] is False


def test_failed_calls_are_not_cached(client, monkeypatch, model_calls):
    state = {"n": 0}
    real = classify_mod._call_model

    def flaky(messages):
        state["n"] += 1
        if state["n"] == 1:
            raise classify_mod.ClassificationError("timeout", retryable=True)
        return real(messages)

    monkeypatch.setattr(classify_mod, "_call_model", flaky)
    assert client.post("/bundle/classify", json=body(request_id="r"), headers=headers()).status_code == 502
    retry = client.post("/bundle/classify", json=body(request_id="r"), headers=headers())
    assert retry.status_code == 200
    assert retry.json()["slot"] == "pan"


def test_duplicate_slot_keys_refused(client, model_calls):
    payload = body(slots=[SLOTS[0], SLOTS[0]])
    assert client.post("/bundle/classify", json=payload, headers=headers()).status_code == 400
    assert model_calls == []


def test_lookup_failure_returns_503(client, monkeypatch, model_calls):
    def broken(org, doc):
        raise RuntimeError("db down")

    monkeypatch.setattr(router_mod, "document_belongs_to_org", broken)
    assert client.post("/bundle/classify", json=body(), headers=headers()).status_code == 503
    assert model_calls == []


def test_confidence_and_key_field_sanitising():
    assert classify_mod._to_confidence("87") == pytest.approx(0.87)
    assert classify_mod._to_confidence(None) == 0.0
    assert classify_mod._to_confidence(5000) == 1.0
    fields = classify_mod.sanitize_key_fields(
        {"id_number": "1234 5678 9012", "full_name": {"nested": 1}, "phone": "  "},
        ["id_number", "full_name", "phone"],
    )
    assert fields == {"id_number": "XXXXXXXX9012"}


def test_call_model_refuses_when_azure_not_configured(monkeypatch):
    # Imports the real Azure OpenAI client module, which needs the image deps.
    pytest.importorskip("app.pipeline.openai_extract", exc_type=ImportError)
    monkeypatch.setattr(settings, "azure_openai_endpoint", "")
    monkeypatch.setattr(settings, "azure_openai_api_key", "")
    with pytest.raises(classify_mod.ClassificationError) as err:
        classify_mod._call_model([{"role": "user", "content": "x"}])
    assert err.value.retryable is False
