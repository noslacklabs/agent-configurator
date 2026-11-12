"""Stand-in for the LLM Configurator, for tests and for trying the blocks without provider keys.

It copies the behaviour of the real service that the blocks depend on (read from
coworkAI ``llm-configurator/backend`` at commit 2456032):

* ``GET /configs`` returns the list shape, including the hard-coded ``endpoint_base_url``.
* ``POST /llm/{name}/chat`` returns an OpenAI-style reply plus ``model_used``.
* a ``model`` key in the body is a 500, ``"stream": true`` is a 502,
  ``max_tokens`` above the cap is a 400, an unknown or disabled config is a 404.
* a guardrail block is HTTP 200 with ``{"blocked": true, "reason": ...}``.
* the requests-per-minute limit is a sliding 60 second window and answers 429
  with ``detail.error_type == "rate_limit"``.
* ``/catalog/global_models`` exists, so a test can prove nobody calls it.

The "model" is a script, not an LLM:

* If tools are offered and the latest user message names one of them, it calls
  that tool. String arguments are filled with the user's text.
* After a tool result it answers ``Final answer: <tool results>``.
* If the system prompt contains ``Reply with exactly one of: a, b, c`` it
  answers with the first option named in the user's text (router preset).
* Otherwise it answers ``Echo: <user text>``.

Run it on the configurator's port with::

    python -m uvicorn tests.mock_configurator:app --port 8001
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import defaultdict
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

CONFIGS: dict[str, dict[str, Any]] = {
    "demo_main": {"operations": ["chat", "embedding"], "rpm": 600, "tpr": 4096, "status": "active"},
    "demo_chatonly": {"operations": ["chat"], "rpm": 600, "tpr": 1024, "status": "active"},
    "demo_guarded": {"operations": ["chat"], "rpm": 600, "tpr": 4096, "status": "active", "blocked_word": "forbidden"},
    "demo_lowrpm": {"operations": ["chat"], "rpm": 2, "tpr": 4096, "status": "active"},
    "demo_disabled": {"operations": ["chat"], "rpm": 600, "tpr": 4096, "status": "disabled"},
}

EMBEDDING_SIZE = 8


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return "" if content is None else str(content)


def _mentioned(name: str, text: str) -> bool:
    lowered = text.lower()
    return name.lower() in lowered or name.lower().replace("_", " ") in lowered


def _arguments(parameters: dict[str, Any], text: str) -> dict[str, Any]:
    properties = parameters.get("properties") or {}
    wanted = parameters.get("required") or list(properties)
    arguments: dict[str, Any] = {}
    for key in wanted:
        kind = (properties.get(key) or {}).get("type", "string")
        if kind in ("integer", "number"):
            arguments[key] = 1
        elif kind == "boolean":
            arguments[key] = True
        elif kind == "array":
            arguments[key] = [text]
        elif kind == "object":
            arguments[key] = {}
        else:
            arguments[key] = text
    return arguments


def scripted_reply(body: dict[str, Any]) -> dict[str, Any]:
    """The assistant message the scripted model gives for this request body."""
    messages = body.get("messages") or []
    tools = body.get("tools") or []
    system = " ".join(_text(m.get("content")) for m in messages if m.get("role") == "system")
    user_text = next((_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")

    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    tool_results = [_text(m.get("content")) for m in messages[last_user + 1:] if m.get("role") == "tool"]
    if tool_results:
        return {"role": "assistant", "content": "Final answer: " + " | ".join(tool_results)}

    for tool in tools:
        function = tool.get("function") or {}
        name = function.get("name") or ""
        if name and _mentioned(name, user_text):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(_arguments(function.get("parameters") or {}, user_text)),
                    },
                }],
            }

    choice = re.search(r"Reply with exactly one of:\s*([^\n.]+)", system)
    if choice:
        options = [option.strip() for option in choice.group(1).split(",") if option.strip()]
        picked = next((option for option in options if _mentioned(option, user_text)), options[0] if options else "")
        return {"role": "assistant", "content": picked}

    return {"role": "assistant", "content": f"Echo: {user_text}"}


def create_app() -> FastAPI:
    app = FastAPI(title="Mock LLM Configurator")
    app.state.requests = []          # (method, path) of every request, for the allow-list test
    app.state.chat_calls = []        # {"config", "agent_id", "body"} of every chat call
    windows: dict[str, list[float]] = defaultdict(list)

    @app.middleware("http")
    async def record(request: Request, call_next):
        app.state.requests.append((request.method, request.url.path))
        return await call_next(request)

    def preflight(full_name: str, operation: str) -> dict[str, Any]:
        config = CONFIGS.get(full_name)
        if not config or config["status"] != "active":
            raise HTTPException(status_code=404, detail="config not found")
        if operation not in config["operations"]:
            raise HTTPException(status_code=400, detail=f"this config has no {operation}-capable model")
        now = time.time()
        windows[full_name] = [stamp for stamp in windows[full_name] if now - stamp < 60]
        if len(windows[full_name]) >= config["rpm"]:
            raise HTTPException(
                status_code=429,
                detail={"error_type": "rate_limit", "message": f"Rate limit exceeded. Max {config['rpm']} RPM."},
            )
        windows[full_name].append(now)
        return config

    @app.get("/configs")
    async def list_configs():
        return [
            {
                "id": index,
                "full_name": name,
                "status": config["status"],
                "operations_present": config["operations"],
                "restrictions": {"tpm": 100000, "rpm": config["rpm"], "tpr": config["tpr"]},
                "endpoint_base_url": f"http://localhost:8001/llm/{name}",
            }
            for index, (name, config) in enumerate(CONFIGS.items(), 1)
        ]

    @app.post("/llm/{full_name}/chat")
    async def chat(full_name: str, request: Request, x_agent_id: str | None = Header(None)):
        body = await request.json()
        config = preflight(full_name, "chat")
        app.state.chat_calls.append({"config": full_name, "agent_id": x_agent_id, "body": body})
        if "model" in body:
            # The real handler calls acompletion(model=..., **body): duplicate keyword argument.
            raise HTTPException(status_code=500, detail="Internal Server Error")
        max_tokens = body.get("max_tokens")
        if max_tokens is not None and max_tokens > config["tpr"]:
            raise HTTPException(status_code=400, detail=f"max_tokens {max_tokens} exceeds config cap of {config['tpr']}")
        if body.get("stream"):
            raise HTTPException(status_code=502, detail="Object of type CustomStreamWrapper is not JSON serializable")

        messages = body.get("messages") or []
        blocked_word = config.get("blocked_word")
        last = messages[-1] if messages else {}
        if blocked_word and last.get("role") == "user" and blocked_word in _text(last.get("content")).lower():
            return JSONResponse(content={"blocked": True, "reason": f"Input contains '{blocked_word}'"})

        message = scripted_reply(body)
        prompt_tokens = sum(len(_text(m.get("content")).split()) for m in messages)
        completion_tokens = len(_text(message.get("content")).split()) or 5
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock/scripted-1",
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                "message": message,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model_used": "mock/scripted-1",
        }

    @app.post("/llm/{full_name}/embeddings")
    async def embeddings(full_name: str, request: Request):
        body = await request.json()
        preflight(full_name, "embedding")
        inputs = body.get("input")
        inputs = [inputs] if isinstance(inputs, str) else list(inputs or [])
        data = []
        for index, text in enumerate(inputs):
            # Deterministic bag-of-letters vector: similar texts get similar vectors.
            vector = [0.0] * EMBEDDING_SIZE
            for character in str(text).lower():
                if character.isalnum():
                    vector[ord(character) % EMBEDDING_SIZE] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            data.append({"object": "embedding", "index": index, "embedding": [value / norm for value in vector]})
        return {"object": "list", "data": data, "model": "mock/embed-1", "usage": {"prompt_tokens": 0, "total_tokens": 0},
                "model_used": "mock/embed-1"}

    @app.get("/catalog/global_models")
    async def global_models():
        # Present only so the tests can show this route is never requested.
        return [{"model_name": "mock/scripted-1", "api_key": "sk-THIS-MUST-NEVER-BE-READ"}]

    return app


app = create_app()
