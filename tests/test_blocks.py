"""The blocks running inside real Langflow graphs, against the mock LLM Configurator."""

from __future__ import annotations

import pytest

from agent_configurator.trace import tool_calls_from_blocks
from tests.flow_builders import blocks, orchestrated_flow, run, single_agent_flow

TEAM = [
    {"agent_name": "Researcher", "agent_role": "Finds facts and dates."},
    {"agent_name": "Mathematician", "agent_role": "Does calculations."},
    {"agent_name": "Writer", "agent_role": "Writes the final text.", "add_calculator_tool": False,
     "add_current_date_tool": False},
]


def chat_calls(configurator, agent_id=None):
    calls = configurator.app.state.chat_calls
    return [call for call in calls if agent_id is None or call["agent_id"] == agent_id]


def offered_tools(call) -> list[str]:
    return [tool["function"]["name"] for tool in call["body"].get("tools", [])]


# -- Configured LLM -------------------------------------------------------


def test_dropdown_lists_active_chat_configs(configurator):
    block = blocks()["llm"]()
    config = block.update_build_config({"config_name": {"options": []}}, "", "config_name")
    assert "demo_main" in config["config_name"]["options"]
    assert "demo_disabled" not in config["config_name"]["options"]


def test_embeddings_dropdown_lists_only_embedding_configs(configurator):
    block = blocks()["embeddings"]()
    config = block.update_build_config({"config_name": {"options": []}}, "", "config_name")
    assert config["config_name"]["options"] == ["demo_main"]


def test_dropdown_keeps_selection_when_configurator_is_down(monkeypatch):
    monkeypatch.setenv("LLM_CONFIGURATOR_URL", "http://127.0.0.1:9")
    block = blocks()["llm"]()
    config = block.update_build_config({"config_name": {"options": ["kept"], "value": "kept"}}, "", "config_name")
    assert config["config_name"]["options"] == ["kept"]


def test_max_tokens_is_lowered_to_the_config_cap(configurator):
    block = blocks()["llm"]()
    block.set(config_name="demo_chatonly", max_tokens=5000)
    assert block.build_model().max_tokens == 1024


def test_no_key_fields_on_any_block():
    for kind in blocks().values():
        for item in kind.inputs:
            assert "key" not in item.name.lower(), f"{kind.__name__} has a key-like input: {item.name}"
            assert type(item).__name__ != "SecretStrInput", f"{kind.__name__}.{item.name} takes a secret"


# -- Agent ----------------------------------------------------------------


async def test_agent_answers_and_reports_its_name(configurator):
    graph, parts = single_agent_flow("demo_main", "hello there", agent_name="Helper")
    await run(graph)
    message = parts["agent"]._agent_result
    assert message.text == "Echo: hello there"
    assert {call["agent_id"] for call in chat_calls(configurator)} == {"Helper"}


async def test_agent_tool_calls_end_up_in_content_blocks(configurator):
    graph, parts = single_agent_flow("demo_main", "use get_current_date please", agent_name="Helper")
    await run(graph)
    message = parts["agent"]._agent_result
    calls = tool_calls_from_blocks([block.model_dump() for block in message.content_blocks])
    assert [call["name"] for call in calls] == ["get_current_date"]
    assert calls[0]["output"]  # the tool's real result, not a wrapper
    assert message.text.startswith("Final answer:")


async def test_agent_refuses_a_model_that_is_not_a_configured_llm(configurator):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    agent = blocks()["agent"]()
    agent.set(agent_name="Helper", model=FakeListChatModel(responses=["x"]))
    with pytest.raises(ValueError, match="connect a Configured LLM block"):
        agent._get_llm()


async def test_guardrail_block_surfaces_as_an_error(configurator):
    graph, _ = single_agent_flow("demo_guarded", "this is forbidden", agent_name="Helper")
    with pytest.raises(Exception, match="Blocked by guardrail"):
        await run(graph)


# -- Orchestrator ---------------------------------------------------------


def step_names(message) -> list[str]:
    return [block.name for block in message.content_blocks if block.type == "tool_use"]


async def test_supervisor_calls_the_agent_it_chose_and_reports_nested_tools(configurator):
    graph, parts = orchestrated_flow(
        TEAM, "Supervisor", "demo_main", "ask the Mathematician to use evaluate_expression"
    )
    await run(graph)
    message = parts["orchestrator"]._orchestration_result
    assert step_names(message) == ["Mathematician"]

    calls = tool_calls_from_blocks([block.model_dump() for block in message.content_blocks])
    assert [(call["name"], call.get("agent")) for call in calls] == [
        ("Mathematician", None),
        ("evaluate_expression", "Mathematician"),
    ]
    assert calls[0]["is_agent"] is True

    # The supervisor's LLM was offered the agents as tools, and each call is attributed.
    supervisor_calls = chat_calls(configurator, "Orchestrator")
    assert offered_tools(supervisor_calls[0]) == ["Researcher", "Mathematician", "Writer"]
    assert chat_calls(configurator, "Mathematician")
    assert not chat_calls(configurator, "Writer")


async def test_sequential_runs_agents_in_order_without_an_orchestrator_llm_call(configurator):
    graph, parts = orchestrated_flow(TEAM, "Sequential", "demo_main", "draft a note")
    await run(graph)
    message = parts["orchestrator"]._orchestration_result
    assert step_names(message) == ["Researcher", "Mathematician", "Writer"]
    assert message.text == "Echo: Echo: Echo: draft a note"
    assert not chat_calls(configurator, "Orchestrator")
    # Each agent received the previous agent's answer.
    inputs = [block.tool_input["input_value"] for block in message.content_blocks if block.type == "tool_use"]
    assert inputs == ["draft a note", "Echo: draft a note", "Echo: Echo: draft a note"]


async def test_router_runs_exactly_one_agent(configurator):
    graph, parts = orchestrated_flow(TEAM, "Router", "demo_main", "this one is for the writer")
    await run(graph)
    message = parts["orchestrator"]._orchestration_result
    assert step_names(message) == ["Writer"]
    assert len(chat_calls(configurator, "Orchestrator")) == 1
    assert not chat_calls(configurator, "Researcher")
    assert not chat_calls(configurator, "Mathematician")


async def test_parallel_runs_every_agent_then_merges(configurator):
    graph, parts = orchestrated_flow(TEAM, "Parallel + aggregate", "demo_main", "compare options")
    await run(graph)
    message = parts["orchestrator"]._orchestration_result
    assert sorted(step_names(message)) == ["Mathematician", "Researcher", "Writer"]
    merge_calls = chat_calls(configurator, "Orchestrator")
    assert len(merge_calls) == 1
    merged_prompt = merge_calls[0]["body"]["messages"][-1]["content"]
    for name in ("Researcher", "Mathematician", "Writer"):
        assert f"[{name}]" in merged_prompt


async def test_solo_agent_skips_the_pattern(configurator):
    graph, parts = orchestrated_flow(
        TEAM, "Supervisor", "demo_main", "use evaluate_expression", solo_agent="Mathematician"
    )
    await run(graph)
    message = parts["orchestrator"]._orchestration_result
    assert step_names(message) == ["Mathematician"]
    assert not chat_calls(configurator, "Orchestrator")
    calls = tool_calls_from_blocks([block.model_dump() for block in message.content_blocks])
    assert ("evaluate_expression", "Mathematician") in [(call["name"], call.get("agent")) for call in calls]


async def test_solo_agent_must_be_connected(configurator):
    graph, _ = orchestrated_flow(TEAM, "Supervisor", "demo_main", "hi", solo_agent="Nobody")
    with pytest.raises(Exception, match="solo agent 'Nobody' is not connected"):
        await run(graph)


async def test_duplicate_agent_names_are_rejected(configurator):
    team = [{"agent_name": "Twin"}, {"agent_name": "Twin"}]
    graph, _ = orchestrated_flow(team, "Sequential", "demo_main", "hi")
    with pytest.raises(Exception, match="agent names must be unique"):
        await run(graph)
