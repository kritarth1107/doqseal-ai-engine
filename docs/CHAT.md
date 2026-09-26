# Grounded chat

Code: `app/chat/` (`engine.py` pipeline, `retrieval.py`, `guard.py`, `llm.py`, `sse.py`),
auth in `app/security.py`. Wire format: [interfaces.md](./interfaces.md).

Pipeline: understand (small talk, library questions, standalone rewrite from history)
→ hybrid retrieval (vector + lexical, per-org collection, organisation filter,
MongoDB visibility and deletion check) → coverage judge (JSON, fails closed) → answer
from the supporting excerpts only, streamed → citation verification → citations.

Uses the existing Azure OpenAI configuration (`AZURE_OPENAI_*`; `CHAT_DEPLOYMENT`
overrides the text deployment). Tuning (env): `CHAT_RETRIEVE_TOP_K` (40),
`CHAT_CONTEXT_CHUNKS` (12), `CHAT_CONTEXT_CHARS` (36000), `CHAT_MIN_SCORE` (0),
`CHAT_HISTORY_TURNS` (10), `CHAT_ANSWER_MAX_TOKENS` (4000), `CHAT_JUDGE_MAX_TOKENS` (2000), `CHAT_TEMPERATURE` (unset),
`CHAT_TIMEOUT_SECONDS` (90), `CHAT_HEARTBEAT_SECONDS` (15).
