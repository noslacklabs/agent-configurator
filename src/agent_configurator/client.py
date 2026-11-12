"""HTTP client for the LLM Configurator (coworkAI, port 8001).

The configurator is read-only for this project and has no authentication, so
this client is deliberately narrow. It can reach exactly three things:

* ``GET  /configs``                    list configs (no secrets in the reply)
* ``POST /llm/{full_name}/chat``       inference, the configurator holds the key
* ``POST /llm/{full_name}/embeddings`` embeddings, same idea

Every request goes through ``_check_route``, which refuses anything else. In
particular ``/catalog/global_models`` (returns provider keys in plain text) and
the config write routes can never be requested from here.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_TIMEOUT_SECONDS = 120.0
# Waits between attempts after a 429. The configurator counts requests in a
# sliding 60 second window, so short waits only help when the window is nearly
# clear. The last value is repeated if more retries are configured.
DEFAULT_RATE_LIMIT_BACKOFF = (2.0, 5.0, 10.0)

_NAME = r"[A-Za-z0-9_-]+"
_ALLOWED_ROUTES = (
    ("GET", re.compile(r"^/configs$")),
    ("POST", re.compile(rf"^/llm/{_NAME}/chat$")),
    ("POST", re.compile(rf"^/llm/{_NAME}/embeddings$")),
)


class ConfiguratorError(RuntimeError):
    """Base class for every failure talking to the LLM Configurator."""


class ConfiguratorUnavailableError(ConfiguratorError):
    """The configurator could not be reached at all."""


class ConfigNotFoundError(ConfiguratorError):
    """The config does not exist, or exists but is not active."""


class RateLimitedError(ConfiguratorError):
    """The config's requests-per-minute limit was hit and retries ran out."""


class GuardrailBlockedError(ConfiguratorError):
    """A guardrail refused the request or the reply.

    The configurator reports this with HTTP 200 and ``{"blocked": true}``, so
    it has to be detected in the body.
    """

    def __init__(self, reason: str):
        super().__init__(f"Blocked by guardrail: {reason}")
        self.reason = reason


class RequestRejectedError(ConfiguratorError):
    """The configurator rejected the request (HTTP 400), e.g. max_tokens above the cap."""


class ProviderError(ConfiguratorError):
    """The provider call failed after the configurator's own retries and fallbacks."""


def base_url_from_env() -> str:
    return (os.getenv("LLM_CONFIGURATOR_URL") or DEFAULT_BASE_URL).rstrip("/")


def _check_route(method: str, path: str) -> None:
    for allowed_method, pattern in _ALLOWED_ROUTES:
        if method == allowed_method and pattern.match(path):
            return
    raise ConfiguratorError(
        f"Refusing to call {method} {path}: only GET /configs and "
        "POST /llm/<name>/chat|embeddings are allowed"
    )


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)
    return str(detail)


class ConfiguratorClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        rate_limit_retries: int | None = None,
        rate_limit_backoff: tuple[float, ...] = DEFAULT_RATE_LIMIT_BACKOFF,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = (base_url or base_url_from_env()).rstrip("/")
        self.timeout = timeout
        if rate_limit_retries is None:
            rate_limit_retries = int(os.getenv("LLM_CONFIGURATOR_429_RETRIES", "3"))
        self.rate_limit_retries = max(0, rate_limit_retries)
        self.rate_limit_backoff = rate_limit_backoff or (0.0,)
        self._transport = transport
        self._async_transport = async_transport

    # -- request plumbing -------------------------------------------------

    def _wait(self, attempt: int) -> float:
        return self.rate_limit_backoff[min(attempt, len(self.rate_limit_backoff) - 1)]

    @staticmethod
    def _headers(agent_id: str | None) -> dict[str, str]:
        return {"X-Agent-Id": agent_id} if agent_id else {}

    def _interpret(self, response: httpx.Response, full_name: str | None) -> Any:
        """Turn a configurator reply into data or a specific error. 429 is handled by the caller."""
        status = response.status_code
        if status == 404:
            raise ConfigNotFoundError(
                f"LLM config '{full_name}' was not found or is not active in the LLM Configurator"
            )
        if status == 400:
            raise RequestRejectedError(f"LLM Configurator rejected the request: {_detail(response)}")
        if status == 502:
            raise ProviderError(f"Provider call failed for config '{full_name}': {_detail(response)}")
        if status >= 400:
            raise ConfiguratorError(f"LLM Configurator returned HTTP {status}: {_detail(response)}")
        try:
            data = response.json()
        except ValueError as exc:
            raise ConfiguratorError("LLM Configurator returned a reply that is not JSON") from exc
        if isinstance(data, dict) and data.get("blocked") is True:
            raise GuardrailBlockedError(str(data.get("reason") or "no reason given"))
        return data

    def _rate_limited(self, response: httpx.Response, full_name: str | None) -> RateLimitedError:
        return RateLimitedError(
            f"LLM config '{full_name}' hit its requests-per-minute limit ({_detail(response)}). "
            "Multi-agent runs make several calls per turn: raise the config's rpm in the LLM Configurator."
        )

    def _request(self, method: str, path: str, *, json: Any = None, agent_id: str | None = None,
                 full_name: str | None = None) -> Any:
        _check_route(method, path)
        attempt = 0
        with httpx.Client(base_url=self.base_url, timeout=self.timeout, transport=self._transport) as http:
            while True:
                try:
                    response = http.request(method, path, json=json, headers=self._headers(agent_id))
                except httpx.HTTPError as exc:
                    raise ConfiguratorUnavailableError(
                        f"Could not reach the LLM Configurator at {self.base_url}: {exc}"
                    ) from exc
                if response.status_code != 429:
                    return self._interpret(response, full_name)
                if attempt >= self.rate_limit_retries:
                    raise self._rate_limited(response, full_name)
                time.sleep(self._wait(attempt))
                attempt += 1

    async def _arequest(self, method: str, path: str, *, json: Any = None, agent_id: str | None = None,
                        full_name: str | None = None) -> Any:
        _check_route(method, path)
        attempt = 0
        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout, transport=self._async_transport
        ) as http:
            while True:
                try:
                    response = await http.request(method, path, json=json, headers=self._headers(agent_id))
                except httpx.HTTPError as exc:
                    raise ConfiguratorUnavailableError(
                        f"Could not reach the LLM Configurator at {self.base_url}: {exc}"
                    ) from exc
                if response.status_code != 429:
                    return self._interpret(response, full_name)
                if attempt >= self.rate_limit_retries:
                    raise self._rate_limited(response, full_name)
                await asyncio.sleep(self._wait(attempt))
                attempt += 1

    # -- public API -------------------------------------------------------

    def list_configs(self, operation: str | None = None) -> list[dict[str, Any]]:
        """Active configs, optionally only those that have ``operation`` (e.g. "embedding")."""
        rows = self._request("GET", "/configs")
        active = [row for row in rows if isinstance(row, dict) and row.get("status") == "active"]
        if operation:
            active = [row for row in active if operation in (row.get("operations_present") or [])]
        return active

    def config_names(self, operation: str | None = None) -> list[str]:
        return sorted(row["full_name"] for row in self.list_configs(operation) if row.get("full_name"))

    def max_tokens_cap(self, full_name: str) -> int | None:
        """The config's tokens-per-request cap (``restrictions.tpr``), if it has one."""
        for row in self.list_configs():
            if row.get("full_name") == full_name:
                cap = (row.get("restrictions") or {}).get("tpr")
                return int(cap) if cap else None
        return None

    @staticmethod
    def _chat_body(body: dict[str, Any]) -> dict[str, Any]:
        # The configurator passes model= itself, so a "model" key raises a 500
        # there. Streaming is not supported by its chat handler either.
        return {key: value for key, value in body.items() if key not in ("model", "stream")}

    def chat(self, full_name: str, body: dict[str, Any], agent_id: str | None = None) -> dict[str, Any]:
        return self._request(
            "POST", f"/llm/{full_name}/chat", json=self._chat_body(body), agent_id=agent_id, full_name=full_name
        )

    async def achat(self, full_name: str, body: dict[str, Any], agent_id: str | None = None) -> dict[str, Any]:
        return await self._arequest(
            "POST", f"/llm/{full_name}/chat", json=self._chat_body(body), agent_id=agent_id, full_name=full_name
        )

    @staticmethod
    def _vectors(data: dict[str, Any]) -> list[list[float]]:
        rows = sorted(data.get("data") or [], key=lambda row: row.get("index", 0))
        return [row["embedding"] for row in rows]

    def embeddings(self, full_name: str, inputs: list[str], agent_id: str | None = None) -> list[list[float]]:
        data = self._request(
            "POST", f"/llm/{full_name}/embeddings", json={"input": inputs}, agent_id=agent_id, full_name=full_name
        )
        return self._vectors(data)

    async def aembeddings(self, full_name: str, inputs: list[str], agent_id: str | None = None) -> list[list[float]]:
        data = await self._arequest(
            "POST", f"/llm/{full_name}/embeddings", json={"input": inputs}, agent_id=agent_id, full_name=full_name
        )
        return self._vectors(data)
