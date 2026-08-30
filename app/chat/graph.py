"""LangGraph chat skeleton: retrieve → generate → format."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from app.chat.tools import search_documents
from app.config import settings

logger = logging.getLogger("doqseal.chat.graph")

STUB_ANSWER = (
    "I'm running in stub mode because the local LLM (Ollama) is unavailable. "
    "Start Ollama with the configured Qwen model to get live answers grounded "
    "in your indexed documents."
)


class ChatState(TypedDict):
    message: str
    organisation_id: str
    project_id: str | None
    user_id: str | None
    context: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    mode: str


def _build_prompt(message: str, context: list[dict[str, Any]]) -> str:
    if not context:
        return (
            "You are DoqSeal, a helpful document assistant. "
            "No indexed document context was retrieved. "
            "Answer clearly and note when information may be missing.\n\n"
            f"User: {message}\n\nAssistant:"
        )

    context_block = "\n\n".join(
        f"[{idx + 1}] documentId={chunk.get('documentId', 'unknown')}\n"
        f"{chunk.get('snippet', '')}"
        for idx, chunk in enumerate(context)
    )
    return (
        "You are DoqSeal, a helpful document assistant. "
        "Use only the context below. Cite sources by document id when relevant.\n\n"
        f"Context:\n{context_block}\n\n"
        f"User: {message}\n\nAssistant:"
    )


def _call_ollama(prompt: str) -> str | None:
    url = f"{settings.ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": settings.llm_model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        with httpx.Client(timeout=90.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            body = response.json()
            text = (body.get("response") or "").strip()
            return text or None
    except Exception as exc:
        logger.warning("Ollama call failed (%s): %s", settings.ollama_url, exc)
        return None


def retrieve_node(state: ChatState) -> dict[str, Any]:
    chunks = search_documents(
        state["organisation_id"],
        state["message"],
        project_id=state.get("project_id"),
        user_id=state.get("user_id"),
    )
    return {"context": chunks}


def generate_node(state: ChatState) -> dict[str, Any]:
    prompt = _build_prompt(state["message"], state.get("context") or [])
    answer = _call_ollama(prompt)
    if answer:
        return {"answer": answer, "mode": "live"}
    return {"answer": STUB_ANSWER, "mode": "stub"}


def format_node(state: ChatState) -> dict[str, Any]:
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()

    for chunk in state.get("context") or []:
        document_id = chunk.get("documentId")
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        citations.append(
            {
                "documentId": document_id,
                "projectId": chunk.get("projectId"),
                "snippet": chunk.get("snippet", ""),
            }
        )

    return {"citations": citations}


def _build_graph():
    workflow = StateGraph(ChatState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("format", format_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "format")
    workflow.add_edge("format", END)
    return workflow.compile()


_graph = _build_graph()


def run_chat(
    message: str,
    organisation_id: str,
    *,
    project_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    result = _graph.invoke(
        {
            "message": message,
            "organisation_id": organisation_id,
            "project_id": project_id,
            "user_id": user_id,
            "context": [],
            "answer": "",
            "citations": [],
            "mode": "stub",
        }
    )
    return {
        "answer": result.get("answer", STUB_ANSWER),
        "citations": result.get("citations") or [],
        "mode": result.get("mode", "stub"),
    }
