"""Hand-off between Langflow and the Agent Evaluator.

The Langflow API key lives only here (from ``.env``) and in the evaluator's
encrypted secret store. It is never written into a flow and never sent to the
browser.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

# A block's type in saved flow JSON is its registry key. On the canvas that is
# "ext:coworkai:CoworkAIAgentComponent@extra"; in flows built in code it is the block's
# short name ("CoworkAIAgent"). Both are matched by these markers.
AGENT_MARKERS = ("CoworkAIAgent",)
ORCHESTRATOR_MARKERS = ("CoworkAIOrchestrator", "OrchestratorAgentComponent")
LLM_MARKERS = ("ConfiguredLLM", "ConfiguredEmbeddings")
EVALUATOR_NAME_LIMIT = 100


class HandoffError(RuntimeError):
    """A failure the page should show as-is. ``status`` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


@dataclass
class Settings:
    langflow_url: str = "http://127.0.0.1:7860"
    langflow_api_key: str = ""
    evaluator_api_url: str = "http://127.0.0.1:8002/api"
    evaluator_ui_url: str = "http://127.0.0.1:5174"
    configurator_url: str = "http://127.0.0.1:8001"
    timeout_seconds: float = 150.0

    @classmethod
    def from_env(cls) -> Settings:
        defaults = cls()
        return cls(
            langflow_url=os.getenv("LANGFLOW_URL", defaults.langflow_url).rstrip("/"),
            langflow_api_key=os.getenv("LANGFLOW_API_KEY", ""),
            evaluator_api_url=os.getenv("EVALUATOR_API_URL", defaults.evaluator_api_url).rstrip("/"),
            evaluator_ui_url=os.getenv("EVALUATOR_UI_URL", defaults.evaluator_ui_url).rstrip("/"),
            configurator_url=os.getenv("LLM_CONFIGURATOR_URL", defaults.configurator_url).rstrip("/"),
            timeout_seconds=float(os.getenv("HANDOFF_TIMEOUT_SECONDS", defaults.timeout_seconds)),
        )


# -- reading a flow -------------------------------------------------------


def _field(node: dict[str, Any], name: str) -> Any:
    template = ((node.get("data") or {}).get("node") or {}).get("template") or {}
    field = template.get(name)
    return field.get("value") if isinstance(field, dict) else None


def _node_type(node: dict[str, Any]) -> str:
    return str((node.get("data") or {}).get("type") or "")


def _is(node: dict[str, Any], markers: tuple[str, ...]) -> bool:
    node_type = _node_type(node)
    return any(marker in node_type for marker in markers)


def describe_flow(flow: dict[str, Any]) -> dict[str, Any]:
    """What the page needs to know about a flow, read from its saved JSON."""
    data = flow.get("data") or {}
    nodes = [node for node in data.get("nodes") or [] if isinstance(node, dict)]
    edges = [edge for edge in data.get("edges") or [] if isinstance(edge, dict)]
    by_id = {node.get("id"): node for node in nodes}

    agents = [
        {
            "id": node.get("id"),
            "name": (_field(node, "agent_name") or "Agent").strip() or "Agent",
            "role": (_field(node, "agent_role") or "").strip(),
        }
        for node in nodes
        if _is(node, AGENT_MARKERS)
    ]

    orchestrator = None
    orchestrator_node = next((node for node in nodes if _is(node, ORCHESTRATOR_MARKERS)), None)
    if orchestrator_node is not None:
        connected = []
        for edge in edges:
            handle = ((edge.get("data") or {}).get("targetHandle") or {})
            if edge.get("target") == orchestrator_node.get("id") and handle.get("fieldName") == "agents":
                source = by_id.get(edge.get("source"))
                if source is not None and _is(source, AGENT_MARKERS):
                    connected.append((_field(source, "agent_name") or "Agent").strip() or "Agent")
        orchestrator = {
            "id": orchestrator_node.get("id"),
            "name": (_field(orchestrator_node, "agent_name") or "Orchestrator").strip(),
            "pattern": _field(orchestrator_node, "pattern") or "Supervisor",
            "agents": connected,
        }

    llm_names = sorted({
        str(_field(node, "llm_name")).strip()
        for node in nodes
        if _is(node, LLM_MARKERS) and _field(node, "llm_name")
    })
    problems = []
    if not any(_is(node, ("ChatInput",)) for node in nodes):
        problems.append("no Chat Input block")
    if not any(_is(node, ("ChatOutput",)) for node in nodes):
        problems.append("no Chat Output block")
    if any(_is(node, LLM_MARKERS) and not _field(node, "llm_name") for node in nodes):
        problems.append("a Configured LLM block has no config selected")

    return {
        "id": str(flow.get("id")),
        "name": flow.get("name") or "Untitled flow",
        "description": flow.get("description") or "",
        "updated_at": flow.get("updated_at"),
        "agents": agents,
        "orchestrator": orchestrator,
        "llm_configs": llm_names,
        "uses_coworkai_blocks": any(
            _is(node, AGENT_MARKERS + ORCHESTRATOR_MARKERS + LLM_MARKERS) for node in nodes
        ),
        "problems": problems,
    }


def _evaluator_name(flow_name: str, agent: str | None = None) -> str:
    name = f"{flow_name} / {agent}" if agent else flow_name
    name = name.strip()
    if len(name) < 2:
        name = f"Flow {name}".strip()
    return name[:EVALUATOR_NAME_LIMIT]


# -- service --------------------------------------------------------------


class HandoffService:
    def __init__(
        self,
        settings: Settings,
        *,
        langflow_transport: httpx.AsyncBaseTransport | None = None,
        evaluator_transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.settings = settings
        self._langflow_transport = langflow_transport
        self._evaluator_transport = evaluator_transport

    def _langflow(self) -> httpx.AsyncClient:
        if not self.settings.langflow_api_key:
            raise HandoffError(
                "LANGFLOW_API_KEY is not set. Create an API key in Langflow (Settings > Langflow API Keys) "
                "or run scripts/create_langflow_key.py, then put it in .env.",
                status=503,
            )
        return httpx.AsyncClient(
            base_url=self.settings.langflow_url,
            headers={"x-api-key": self.settings.langflow_api_key},
            timeout=30,
            transport=self._langflow_transport,
        )

    def _evaluator(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.settings.evaluator_api_url,
            timeout=self.settings.timeout_seconds,
            transport=self._evaluator_transport,
        )

    @staticmethod
    def _detail(response: httpx.Response) -> str:
        try:
            detail = response.json().get("detail", response.text)
        except (ValueError, AttributeError):
            detail = response.text
        return str(detail)[:600]

    async def _get_flows(self) -> list[dict[str, Any]]:
        try:
            async with self._langflow() as langflow:
                response = await langflow.get(
                    "/api/v1/flows/",
                    params={"get_all": "true", "components_only": "false", "remove_example_flows": "true"},
                )
        except httpx.HTTPError as exc:
            raise HandoffError(f"Could not reach Langflow at {self.settings.langflow_url}: {exc}") from exc
        if response.status_code in (401, 403):
            raise HandoffError("Langflow refused LANGFLOW_API_KEY. Create a new key and update .env.", status=503)
        if response.status_code >= 400:
            raise HandoffError(f"Langflow returned HTTP {response.status_code}: {self._detail(response)}")
        flows = response.json()
        return [flow for flow in flows if isinstance(flow, dict) and not flow.get("is_component")]

    async def _evaluator_agents(self) -> list[dict[str, Any]]:
        try:
            async with self._evaluator() as evaluator:
                response = await evaluator.get("/agents")
        except httpx.HTTPError as exc:
            raise HandoffError(
                f"Could not reach the Agent Evaluator at {self.settings.evaluator_api_url}: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise HandoffError(f"Agent Evaluator returned HTTP {response.status_code}: {self._detail(response)}")
        return [
            agent for agent in response.json().get("agents", [])
            if agent.get("connection_type") == "langflow"
        ]

    @staticmethod
    def _registrations(
        agents: list[dict[str, Any]], flow_id: str, describe: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        found = []
        for agent in agents:
            config = agent.get("config") or {}
            if str(config.get("flow_id")) != flow_id:
                continue
            if describe is not None and config.get("solo_agent"):
                current = {describe["orchestrator"]["id"]: {"solo_agent": config["solo_agent"]}} if describe.get(
                    "orchestrator") else None
                if config.get("tweaks") != current:
                    continue  # points at an Orchestrator block that is no longer in the flow
            found.append({
                "agent_id": agent.get("id"),
                "name": agent.get("name"),
                "agent": config.get("solo_agent"),
                "status": agent.get("status"),
                "last_tested_at": agent.get("last_tested_at"),
            })
        return found

    async def list_flows(self) -> dict[str, Any]:
        flows = await self._get_flows()
        evaluator_error = None
        try:
            agents = await self._evaluator_agents()
        except HandoffError as exc:
            agents, evaluator_error = [], str(exc)
        described = []
        for flow in flows:
            item = describe_flow(flow)
            item["registered"] = self._registrations(agents, item["id"], item)
            item["langflow_url"] = f"{self.settings.langflow_url}/flow/{item['id']}"
            described.append(item)
        described.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return {
            "flows": described,
            "evaluator_url": self.settings.evaluator_ui_url,
            "evaluator_error": evaluator_error,
        }

    def _agent_config(self, flow: dict[str, Any], solo_agent: str | None) -> dict[str, Any]:
        config: dict[str, Any] = {
            "base_url": self.settings.langflow_url,
            "flow_id": flow["id"],
            "allow_private_network": True,
            "timeout_seconds": self.settings.timeout_seconds - 10,
            "source": "agent-configurator",
            "flow_name": flow["name"],
        }
        if solo_agent:
            # Tool metrics for one agent should see that agent's tools, not "called agent X".
            config.update({
                "solo_agent": solo_agent,
                "tweaks": {flow["orchestrator"]["id"]: {"solo_agent": solo_agent}},
                "include_agent_calls": False,
            })
        return config

    async def _register(
        self,
        evaluator: httpx.AsyncClient,
        existing: list[dict[str, Any]],
        flow: dict[str, Any],
        solo_agent: str | None,
        *,
        test: bool,
    ) -> dict[str, Any]:
        config = self._agent_config(flow, solo_agent)
        name = _evaluator_name(flow["name"], solo_agent)
        result: dict[str, Any] = {"name": name, "scope": "agent" if solo_agent else "flow", "agent": solo_agent}

        match = next(
            (
                agent for agent in existing
                if str((agent.get("config") or {}).get("flow_id")) == flow["id"]
                and (agent.get("config") or {}).get("solo_agent") == solo_agent
                and (agent.get("config") or {}).get("base_url") == config["base_url"]
                # The tweak names the Orchestrator block by id. If that block was replaced, the old
                # registration no longer reaches it and a new one is needed.
                and (agent.get("config") or {}).get("tweaks") == config.get("tweaks")
            ),
            None,
        )

        async def run_test(agent_id: str) -> dict[str, Any]:
            response = await evaluator.post(f"/agents/{agent_id}/test", json={})
            if response.status_code < 400:
                return {"ok": True, "message": "Connection successful", "output": response.json().get("output")}
            return {"ok": False, "message": self._detail(response)}

        if match is not None:
            outcome = await run_test(match["id"]) if test else {"ok": None, "message": "Not tested"}
            if outcome["ok"] is not False:
                return {**result, "agent_id": match["id"], "name": match.get("name") or name, "created": False,
                        "test": outcome}
            # The stored registration no longer works (for example the key was replaced): make a new one.

        response = await evaluator.post(
            "/agents",
            json={
                "name": name,
                "connection_type": "langflow",
                "config": config,
                "secrets": {"api_key": self.settings.langflow_api_key},
            },
        )
        if response.status_code == 422 and "langflow" in self._detail(response).lower():
            raise HandoffError(
                "The Agent Evaluator does not know the 'langflow' connection type. "
                "Update Agent-Evaluation to a version that has backend/adapters/langflow.py.",
                status=502,
            )
        if response.status_code >= 400:
            raise HandoffError(f"Agent Evaluator refused the registration: {self._detail(response)}")
        agent_id = response.json()["id"]
        outcome = await run_test(agent_id) if test else {"ok": None, "message": "Not tested"}
        return {**result, "agent_id": agent_id, "created": True, "test": outcome}

    async def evaluate(self, flow_id: str, *, include_agents: bool = True, test: bool = True) -> dict[str, Any]:
        flows = await self._get_flows()
        raw = next((flow for flow in flows if str(flow.get("id")) == flow_id), None)
        if raw is None:
            raise HandoffError(f"Flow {flow_id} was not found in Langflow.", status=404)
        flow = describe_flow(raw)
        existing = await self._evaluator_agents()

        targets: list[str | None] = [None]
        if include_agents and flow["orchestrator"]:
            targets.extend(dict.fromkeys(flow["orchestrator"]["agents"]))

        registered = []
        try:
            async with self._evaluator() as evaluator:
                for solo_agent in targets:
                    registered.append(await self._register(evaluator, existing, flow, solo_agent, test=test))
        except httpx.HTTPError as exc:
            raise HandoffError(
                f"Could not reach the Agent Evaluator at {self.settings.evaluator_api_url}: {exc}"
            ) from exc
        return {
            "flow": flow,
            "registered": registered,
            "evaluator_url": self.settings.evaluator_ui_url,
            "all_ok": all(item["test"]["ok"] is not False for item in registered),
        }

    async def health(self) -> dict[str, Any]:
        async def probe(url: str, path: str, transport=None) -> dict[str, Any]:
            try:
                async with httpx.AsyncClient(timeout=5, transport=transport) as client:
                    response = await client.get(f"{url}{path}")
                return {"url": url, "ok": response.status_code < 500}
            except httpx.HTTPError as exc:
                return {"url": url, "ok": False, "error": str(exc) or type(exc).__name__}

        evaluator_root = self.settings.evaluator_api_url.removesuffix("/api")
        return {
            "langflow": await probe(self.settings.langflow_url, "/health", self._langflow_transport),
            "evaluator": await probe(evaluator_root, "/health", self._evaluator_transport),
            # GET /configs is one of the three configurator routes this project is allowed to call.
            "llm_configurator": await probe(self.settings.configurator_url, "/configs"),
            "langflow_api_key_set": bool(self.settings.langflow_api_key),
        }
