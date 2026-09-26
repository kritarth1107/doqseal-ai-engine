"""In-memory stand-ins for Mongo, Qdrant and the model used by chat tests."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import jwt

from app.security import ALGORITHM, AUDIENCE, ISSUER

SECRET = "test-shared-secret-0123456789abcdef-xyz"


def make_token(
    org: str = "org-a",
    user: str = "user-a1",
    scope: str = "chat",
    *,
    secret: str = SECRET,
    pid: str | None = None,
    lifetime: int = 120,
    iat_offset: int = 0,
    **overrides: Any,
) -> str:
    now = int(time.time()) + iat_offset
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": user,
        "org": org,
        "pid": pid,
        "scope": scope,
        "iat": now,
        "exp": now + lifetime,
        "jti": uuid.uuid4().hex,
    }
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not ...}
    return jwt.encode(claims, secret, algorithm=ALGORITHM)


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# --- Mongo -----------------------------------------------------------------


def _matches(doc: dict[str, Any], query: dict[str, Any]) -> bool:
    for key, cond in query.items():
        if key == "$or":
            if not any(_matches(doc, sub) for sub in cond):
                return False
            continue
        value = doc.get(key)
        if isinstance(cond, dict):
            if "$in" in cond and value not in cond["$in"]:
                return False
            if "$ne" in cond and value == cond["$ne"]:
                return False
        elif cond is None:
            if value is not None:
                return False
        elif value != cond:
            return False
    return True


class _Cursor(list):
    def sort(self, key: str, direction: int):
        self.sort_key = (key, direction)
        items = sorted(self, key=lambda d: d.get(key) or 0, reverse=direction < 0)
        return _Cursor(items)

    def limit(self, n: int):
        return _Cursor(self[:n])


class FakeCollection:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.queries: list[dict[str, Any]] = []

    def find(self, query: dict[str, Any], projection: dict[str, Any] | None = None):
        self.queries.append(query)
        return _Cursor([dict(r) for r in self.rows if _matches(r, query)])


class FakeDB:
    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = FakeCollection(documents)


# --- Qdrant ----------------------------------------------------------------


class FakeQdrant:
    """Returns every point in the collection (ignores filters on purpose) so the
    code-level organisation and deletion checks are what gets tested."""

    def __init__(self, collections: dict[str, list[dict[str, Any]]]):
        self.collections = collections
        self.calls: list[dict[str, Any]] = []

    def __call__(self, collection, vector, query_filter, limit):
        self.calls.append({"collection": collection, "filter": query_filter, "limit": limit})
        return list(self.collections.get(collection, []))[:limit]


def point(org: str, doc: str, text: str, score: float = 0.85, **payload: Any) -> dict[str, Any]:
    return {"score": score, "payload": {"organisationId": org, "documentId": doc, "text": text, **payload}}


# --- Model -----------------------------------------------------------------


class FakeModel:
    """Scripted model. Coverage is decided from the excerpts it is shown."""

    def __init__(self, answer: str = "", coverage_keywords: tuple[str, ...] = (), partial: bool = False):
        self.answer = answer
        self.coverage_keywords = coverage_keywords
        self.partial = partial
        self.json_calls: list[list[dict[str, Any]]] = []
        self.stream_calls: list[list[dict[str, Any]]] = []
        self.stream_closed = False
        self.token_delay = 0.0
        self.fail_json: Exception | None = None
        self.fail_stream: Exception | None = None
        self.rewrite_to: str | None = None
        # The judge only says "covered" when the question is about what the
        # documents contain (stands in for the model's own reading).
        self.covered_question = r"trade|trading|day"

    async def complete_json(self, messages, *, max_tokens=400):
        self.json_calls.append(messages)
        if self.fail_json:
            raise self.fail_json
        system = messages[0]["content"]
        if system.startswith("Rewrite"):
            return {"query": self.rewrite_to or messages[-1]["content"]}, {"prompt_tokens": 5, "completion_tokens": 2}
        import re

        question = messages[-1]["content"].split("Excerpts:", 1)[0]
        if not re.search(self.covered_question, question, re.I):
            return {"coverage": "none", "supporting": [], "missing": ""}, {"prompt_tokens": 50, "completion_tokens": 5}
        excerpts = messages[-1]["content"].split("Excerpts:", 1)[-1]
        blocks = excerpts.split("<document ")[1:]
        supporting = [
            i + 1 for i, b in enumerate(blocks) if any(k.lower() in b.lower() for k in self.coverage_keywords)
        ]
        coverage = "none" if not supporting else ("partial" if self.partial else "full")
        return (
            {"coverage": coverage, "supporting": supporting, "missing": "fees" if self.partial else ""},
            {"prompt_tokens": 50, "completion_tokens": 10},
        )

    async def stream_chat(self, messages, *, max_tokens):
        self.stream_calls.append(messages)
        if self.fail_stream:
            raise self.fail_stream
        try:
            pieces = [self.answer[i : i + 7] for i in range(0, len(self.answer), 7)]
            for piece in pieces:
                if self.token_delay:
                    await asyncio.sleep(self.token_delay)
                yield {"text": piece}
            yield {"usage": {"prompt_tokens": 400, "completion_tokens": 60}}
        finally:
            self.stream_closed = True


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events
