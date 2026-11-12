"""Build CoworkAI flows in code, for tests and for generating the starter templates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from lfx.components.input_output import ChatInput, ChatOutput
from lfx.graph import Graph

COMPONENTS_DIR = Path(__file__).resolve().parent.parent / "components" / "coworkai"


def load_block(module_name: str, class_name: str):
    """Import a block from components/coworkai the way a standalone file is loaded."""
    key = f"coworkai_blocks_{module_name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, COMPONENTS_DIR / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return getattr(sys.modules[key], class_name)


def blocks():
    return {
        "llm": load_block("configured_llm", "ConfiguredLLMComponent"),
        "embeddings": load_block("configured_embeddings", "ConfiguredEmbeddingsComponent"),
        "agent": load_block("agent", "CoworkAIAgentComponent"),
        "orchestrator": load_block("orchestrator", "OrchestratorAgentComponent"),
    }


def single_agent_flow(config_name: str = "", user_input: str = "", **agent_fields):
    """Chat Input -> Agent (on a Configured LLM) -> Chat Output."""
    kinds = blocks()
    chat_input = ChatInput(_id="ChatInput-cwk01")
    chat_input.set(input_value=user_input)
    llm = kinds["llm"](_id="ConfiguredLLM-cwk01")
    llm.set(config_name=config_name)
    agent = kinds["agent"](_id="CoworkAIAgent-cwk01")
    fields = {"agent_name": "Assistant", "agent_role": "Answers questions.", **agent_fields}
    agent.set(model=llm.build_model, input_value=chat_input.message_response, **fields)
    chat_output = ChatOutput(_id="ChatOutput-cwk01")
    chat_output.set(input_value=agent.message_response)
    return Graph(chat_input, chat_output), {"agent": agent, "llm": llm, "chat_output": chat_output}


def orchestrated_flow(
    agents: list[dict],
    pattern: str = "Supervisor",
    config_name: str = "",
    user_input: str = "",
    solo_agent: str = "",
):
    """Chat Input -> Orchestrator (with one Agent block per entry in ``agents``) -> Chat Output."""
    kinds = blocks()
    chat_input = ChatInput(_id="ChatInput-cwk01")
    chat_input.set(input_value=user_input)
    llm = kinds["llm"](_id="ConfiguredLLM-cwk01")
    llm.set(config_name=config_name)

    agent_blocks = []
    for index, fields in enumerate(agents, 1):
        agent = kinds["agent"](_id=f"CoworkAIAgent-cwk{index:02d}")
        agent.set(model=llm.build_model, **fields)
        agent_blocks.append(agent)

    orchestrator = kinds["orchestrator"](_id="CoworkAIOrchestrator-cwk01")
    orchestrator.set(
        model=llm.build_model,
        agents=[agent.as_agent for agent in agent_blocks],
        pattern=pattern,
        input_value=chat_input.message_response,
        solo_agent=solo_agent,
    )
    chat_output = ChatOutput(_id="ChatOutput-cwk01")
    chat_output.set(input_value=orchestrator.orchestrate)
    return Graph(chat_input, chat_output), {
        "orchestrator": orchestrator, "agents": agent_blocks, "llm": llm, "chat_output": chat_output,
    }


async def run(graph: Graph) -> None:
    async for _ in graph.async_start():
        pass
