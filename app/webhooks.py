"""Fire project webhook URLs for document lifecycle events."""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("doqseal.webhooks")

WEBHOOK_EVENTS = {
    "document.uploaded",
    "document.processing",
    "document.processed",
    "document.failed",
}


def _is_http_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def coerce_project_webhooks(project: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Support new `{url, events}` shape and legacy `webhookUrls: string[]`."""
    if not project:
        return []

    raw = project.get("webhooks")
    if isinstance(raw, list) and raw:
        hooks: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url or not _is_http_url(url):
                continue
            events = [
                e
                for e in (item.get("events") or [])
                if isinstance(e, str) and e in WEBHOOK_EVENTS
            ]
            if not events:
                events = ["document.processed"]
            hooks.append(
                {
                    "url": url,
                    "events": events,
                    "enabled": item.get("enabled", True) is not False,
                }
            )
        return hooks[:1]

    legacy = project.get("webhookUrls")
    if isinstance(legacy, list):
        for u in legacy:
            if isinstance(u, str) and _is_http_url(str(u).strip()):
                return [
                    {
                        "url": str(u).strip(),
                        "events": ["document.processed"],
                        "enabled": True,
                    }
                ]
    return []


def dispatch_project_webhooks(
    project: dict[str, Any] | None,
    *,
    event: str,
    project_id: str,
    document_id: str,
    job_id: str,
    organisation_id: str,
    document: dict[str, Any] | None = None,
    extraction_payload: dict[str, Any] | None = None,
    error: str | None = None,
    status: str | None = None,
) -> None:
    if event not in WEBHOOK_EVENTS:
        return

    hooks = [
        h
        for h in coerce_project_webhooks(project)
        if h.get("enabled", True) and event in (h.get("events") or [])
    ]
    if not hooks:
        return

    payload = {
        "event": event,
        "projectId": project_id,
        "documentId": document_id,
        "jobId": job_id,
        "organisationId": organisation_id,
        "status": status,
        "originalFilename": (document or {}).get("originalFilename"),
        "displayTitle": (extraction_payload or {}).get("displayTitle")
        or (document or {}).get("displayTitle"),
        "error": error,
        "extraction": (
            {
                "data": (extraction_payload or {}).get("data"),
                "fieldConfidence": (extraction_payload or {}).get("fieldConfidence"),
                "strategy": (extraction_payload or {}).get("strategy"),
                "status": (extraction_payload or {}).get("status"),
            }
            if extraction_payload
            else None
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    body = json.dumps(payload).encode("utf-8")

    for hook in hooks:
        url = hook["url"]
        try:
            request = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "DoqSeal-Webhooks/1.0",
                    "X-DoqSeal-Event": event,
                },
            )
            with urllib.request.urlopen(request, timeout=8) as response:
                if response.status >= 400:
                    logger.warning(
                        "Webhook %s responded %s for %s %s",
                        url,
                        response.status,
                        event,
                        document_id,
                    )
        except Exception as err:
            logger.warning(
                "Webhook failed %s for %s %s: %s",
                url,
                event,
                document_id,
                err,
            )
