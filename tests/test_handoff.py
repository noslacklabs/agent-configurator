from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from agent_configurator.handoff.app import create_app
from agent_configurator.handoff.service import HandoffError, HandoffService, Settings, describe_flow
from tests.flow_builders import orchestrated_flow, single_agent_flow

KEY = "lf-secret-key"
TEAM = [
    {"agent_name": "Researcher", "agent_role": "Finds facts."},
    {"agent_name": "Writer", "agent_role": "Writes text."},
]


def saved(graph, flow_id: str, name: str) -> dict:
    flow = graph.dump(name=name)
    return {"id": flow_id, "name": name, "description": "", "is_component": False,
            "updated_at": "2026-11-13T10:00:00", "data": flow["data"]}


@pytest.fixture(scope="module")
def flows():
    single, _ = single_agent_flow("demo_main", "", agent_name="Helper")
    team, _ = orchestrated_flow(TEAM, "Router", "demo_main", "")
    empty, _ = single_agent_flow("", "", agent_name="Helper")
    return [
        saved(single, "flow-single", "Support bot"),
        saved(team, "flow-team", "Content team"),
        saved(empty, "flow-empty", "Unfinished"),
    ]


class FakeLangflow:
    def __init__(self, flows, accept_key=KEY):
        self.flows = flows
        self.accept_key = accept_key
        self.keys_seen = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.keys_seen.append(request.headers.get("x-api-key"))
        if request.headers.get("x-api-key") != self.accept_key:
            return httpx.Response(403, json={"detail": "Invalid API key"})
        if request.url.path == "/api/v1/flows/":
            return httpx.Response(200, json=self.flows)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"detail": "not found"})


class FakeEvaluator:
    """The parts of the Agent Evaluator API the service uses, with an in-memory store."""

    def __init__(self, failing_agent_ids=(), knows_langflow=True):
        self.agents = []
        self.created = []
        self.tested = []
        self.failing = set(failing_agent_ids)
        self.knows_langflow = knows_langflow

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/api/agents":
            public = [{k: v for k, v in agent.items() if k != "secrets"} for agent in self.agents]
            return httpx.Response(200, json={"agents": public})
        if request.method == "POST" and path == "/api/agents":
            body = json.loads(request.content)
            if not self.knows_langflow:
                return httpx.Response(422, json={"detail": "Input should be 'rest_api', ... not 'langflow'"})
            agent = {"id": f"ev-{len(self.agents) + 1}", "status": "untested", "last_tested_at": None, **body}
            self.agents.append(agent)
            self.created.append(body)
            return httpx.Response(201, json={k: v for k, v in agent.items() if k != "secrets"})
        if request.method == "POST" and path.endswith("/test"):
            agent_id = path.split("/")[-2]
            self.tested.append(agent_id)
            if agent_id in self.failing:
                return httpx.Response(400, json={"detail": "Connection test failed: Check the Langflow API key."})
            return httpx.Response(200, json={"ok": True, "output": "It works."})
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"detail": "not found"})


def make_service(flows, evaluator=None, langflow=None, api_key=KEY):
    langflow = langflow or FakeLangflow(flows)
    evaluator = evaluator or FakeEvaluator()
    service = HandoffService(
        Settings(langflow_api_key=api_key),
        langflow_transport=httpx.MockTransport(langflow),
        evaluator_transport=httpx.MockTransport(evaluator),
    )
    return service, langflow, evaluator


# -- reading flows --------------------------------------------------------


def test_describe_single_agent_flow(flows):
    info = describe_flow(flows[0])
    assert info["name"] == "Support bot"
    assert [agent["name"] for agent in info["agents"]] == ["Helper"]
    assert info["orchestrator"] is None
    assert info["llm_configs"] == ["demo_main"]
    assert info["problems"] == []


def test_describe_orchestrated_flow(flows):
    info = describe_flow(flows[1])
    assert info["orchestrator"]["id"] == "CoworkAIOrchestrator-cwk01"
    assert info["orchestrator"]["pattern"] == "Router"
    assert sorted(info["orchestrator"]["agents"]) == ["Researcher", "Writer"]


def test_describe_reports_a_missing_llm_selection(flows):
    assert "a Configured LLM block has no config selected" in describe_flow(flows[2])["problems"]


async def test_list_flows_marks_what_is_already_registered(flows):
    service, _, evaluator = make_service(flows)
    await service.evaluate("flow-single")
    listing = await service.list_flows()
    by_id = {flow["id"]: flow for flow in listing["flows"]}
    assert [item["name"] for item in by_id["flow-single"]["registered"]] == ["Support bot"]
    assert by_id["flow-team"]["registered"] == []
    assert listing["evaluator_error"] is None


async def test_list_flows_still_works_when_the_evaluator_is_down(flows):
    def down(request):
        raise httpx.ConnectError("connection refused")

    service, _, _ = make_service(flows, evaluator=down)
    listing = await service.list_flows()
    assert len(listing["flows"]) == 3
    assert "Could not reach the Agent Evaluator" in listing["evaluator_error"]


# -- registering ----------------------------------------------------------


async def test_single_flow_registers_one_evaluator_agent(flows):
    service, _, evaluator = make_service(flows)
    result = await service.evaluate("flow-single")
    assert result["all_ok"] is True
    assert [(item["name"], item["scope"], item["created"]) for item in result["registered"]] == [
        ("Support bot", "flow", True)
    ]
    body = evaluator.created[0]
    assert body["connection_type"] == "langflow"
    assert body["secrets"] == {"api_key": KEY}
    assert body["config"]["flow_id"] == "flow-single"
    assert body["config"]["allow_private_network"] is True
    assert "tweaks" not in body["config"]
    assert evaluator.tested == ["ev-1"]


async def test_orchestrated_flow_also_registers_each_agent_with_a_solo_tweak(flows):
    service, _, evaluator = make_service(flows)
    result = await service.evaluate("flow-team")
    names = [item["name"] for item in result["registered"]]
    assert names[0] == "Content team"
    assert sorted(names[1:]) == ["Content team / Researcher", "Content team / Writer"]
    solo = next(body for body in evaluator.created if body["config"].get("solo_agent") == "Writer")
    assert solo["config"]["tweaks"] == {"CoworkAIOrchestrator-cwk01": {"solo_agent": "Writer"}}
    assert solo["config"]["include_agent_calls"] is False
    assert len(evaluator.tested) == 3


async def test_include_agents_false_registers_only_the_flow(flows):
    service, _, evaluator = make_service(flows)
    result = await service.evaluate("flow-team", include_agents=False)
    assert len(result["registered"]) == 1
    assert len(evaluator.created) == 1


async def test_second_click_reuses_the_registration(flows):
    service, _, evaluator = make_service(flows)
    await service.evaluate("flow-team")
    again = await service.evaluate("flow-team")
    assert [item["created"] for item in again["registered"]] == [False, False, False]
    assert len(evaluator.created) == 3
    assert len(evaluator.tested) == 6


async def test_a_registration_that_stopped_working_is_replaced(flows):
    evaluator = FakeEvaluator(failing_agent_ids={"ev-1"})
    service, _, _ = make_service(flows, evaluator=evaluator)
    first = await service.evaluate("flow-single")
    assert first["all_ok"] is False
    second = await service.evaluate("flow-single")
    assert second["registered"][0]["created"] is True
    assert second["registered"][0]["agent_id"] == "ev-2"
    assert second["all_ok"] is True


async def test_unknown_flow_is_a_404(flows):
    service, _, _ = make_service(flows)
    with pytest.raises(HandoffError, match="was not found in Langflow") as caught:
        await service.evaluate("nope")
    assert caught.value.status == 404


async def test_missing_or_refused_langflow_key_is_explained(flows):
    service, _, _ = make_service(flows, api_key="")
    with pytest.raises(HandoffError, match="LANGFLOW_API_KEY is not set"):
        await service.list_flows()
    service, _, _ = make_service(flows, api_key="wrong")
    with pytest.raises(HandoffError, match="Langflow refused LANGFLOW_API_KEY"):
        await service.list_flows()


async def test_old_evaluator_without_the_langflow_type_is_explained(flows):
    service, _, _ = make_service(flows, evaluator=FakeEvaluator(knows_langflow=False))
    with pytest.raises(HandoffError, match="does not know the 'langflow' connection type"):
        await service.evaluate("flow-single")


# -- HTTP surface ---------------------------------------------------------


def test_api_never_returns_the_langflow_key(flows, tmp_path):
    service, _, _ = make_service(flows)
    client = TestClient(create_app(service, frontend_dir=tmp_path))
    listing = client.get("/api/flows")
    assert listing.status_code == 200
    registered = client.post("/api/flows/flow-team/evaluate", json={})
    assert registered.status_code == 200
    health = client.get("/api/health")
    for response in (listing, registered, health):
        assert KEY not in response.text
    assert health.json()["langflow_api_key_set"] is True


def test_api_errors_carry_the_message(flows, tmp_path):
    service, _, _ = make_service(flows)
    client = TestClient(create_app(service, frontend_dir=tmp_path))
    response = client.post("/api/flows/nope/evaluate")
    assert response.status_code == 404
    assert "was not found in Langflow" in response.json()["detail"]


def test_describe_reads_the_shipped_templates():
    """Templates use the canvas type names (ext:coworkai:...@extra), unlike flows built in code."""
    from pathlib import Path

    templates = Path(__file__).resolve().parent.parent / "templates"
    single = describe_flow({"id": "t1", **json.loads((templates / "single-qa-agent.json").read_text("utf-8"))})
    assert [agent["name"] for agent in single["agents"]] == ["Assistant"]
    assert single["uses_coworkai_blocks"] is True
    # The only setup step left to the user:
    assert single["problems"] == ["a Configured LLM block has no config selected"]

    team = describe_flow(
        {"id": "t2", **json.loads((templates / "supervisor-three-specialists.json").read_text("utf-8"))}
    )
    assert team["orchestrator"]["pattern"] == "Supervisor"
    assert sorted(team["orchestrator"]["agents"]) == ["Analyst", "Researcher", "Writer"]
    assert team["orchestrator"]["id"].startswith("ext:coworkai:OrchestratorAgentComponent@extra-")


def test_templates_contain_no_keys():
    from pathlib import Path

    for path in (Path(__file__).resolve().parent.parent / "templates").glob("*.json"):
        text = path.read_text("utf-8")
        assert "sk-" not in text, path.name
        flow = json.loads(text)
        for node in flow["data"]["nodes"]:
            for name, field in node["data"]["node"]["template"].items():
                if isinstance(field, dict):
                    assert not field.get("password"), f"{path.name}: {name} is a secret field"


async def test_replaced_orchestrator_block_gets_new_agent_registrations(flows):
    service, _, evaluator = make_service(flows)
    await service.evaluate("flow-team")
    # The user deletes the Orchestrator block and adds a new one: same flow, new block id.
    changed = json.loads(json.dumps(flows[1]).replace("CoworkAIOrchestrator-cwk01", "CoworkAIOrchestrator-new99"))
    service, _, _ = make_service([flows[0], changed, flows[2]], evaluator=evaluator)

    listing = await service.list_flows()
    team = next(flow for flow in listing["flows"] if flow["id"] == "flow-team")
    # The old per-agent registrations point at a block that is gone, so they are not shown as current.
    assert [item["name"] for item in team["registered"]] == ["Content team"]

    again = await service.evaluate("flow-team")
    created = {item["name"]: item["created"] for item in again["registered"]}
    assert created == {"Content team": False, "Content team / Researcher": True, "Content team / Writer": True}
    newest = [body for body in evaluator.created if body["config"].get("solo_agent")][-2:]
    assert all(list(body["config"]["tweaks"]) == ["CoworkAIOrchestrator-new99"] for body in newest)
