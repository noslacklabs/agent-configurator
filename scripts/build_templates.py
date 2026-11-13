"""Regenerate the starter templates in templates/ from a running Langflow.

The block definitions are read from the Langflow server, so the templates carry
exactly what that server would put on the canvas. Run it again after changing a
block or upgrading Langflow:

    python scripts/build_templates.py [--url http://127.0.0.1:7860] [--env .env] [--out templates]

Every template leaves the LLM Config empty: picking one is the only setup step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
from lfx.graph.flow_builder.component import add_component, configure_component, sync_dropdown_selected_outputs
from lfx.graph.flow_builder.connect import add_connection
from lfx.graph.flow_builder.flow import empty_flow
from lfx.graph.flow_builder.layout import layout_flow

QA_INSTRUCTIONS = (
    "You are a helpful assistant. Answer the user's question clearly and briefly. "
    "Use your tools when they help. If you do not know, say so."
)

SPECIALISTS = [
    {
        "agent_name": "Researcher",
        "agent_role": "Finds and checks facts, dates and background on a topic. Call it first when the request needs information.",
        "system_prompt": "You are a researcher. Gather the facts needed for the task and report them as short, sourced bullet points. Do not write the final answer.",
    },
    {
        "agent_name": "Analyst",
        "agent_role": "Does calculations, comparisons and reasoning over facts it is given. Call it when numbers or trade-offs are involved.",
        "system_prompt": "You are an analyst. Work through the numbers and trade-offs in the material you are given, show the key steps, and state a conclusion.",
    },
    {
        "agent_name": "Writer",
        "agent_role": "Turns findings into the final, well-structured answer for the user. Call it last.",
        "system_prompt": "You are a writer. Turn the material you are given into a clear, well-organised answer for the user. Do not add facts that are not in the material.",
    },
]


def read_env(path: Path) -> dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def load_registry(url: str, api_key: str) -> dict[str, dict]:
    response = httpx.get(f"{url}/api/v1/all", headers={"x-api-key": api_key}, timeout=120)
    response.raise_for_status()
    registry: dict[str, dict] = {}
    for category, blocks in response.json().items():
        if isinstance(blocks, dict):
            for key, block in blocks.items():
                if isinstance(block, dict) and "template" in block:
                    registry[key] = {**block, "category": category}
    return registry


def find(registry: dict[str, dict], category: str, class_name: str) -> str:
    """The registry key of a block, e.g. the server's key for our Agent block."""
    for key, block in registry.items():
        if block.get("category") == category and (
            key == class_name or (block.get("metadata") or {}).get("component_class_name") == class_name
        ):
            return key
    raise SystemExit(f"Block {class_name} not found in category '{category}'. Is LANGFLOW_COMPONENTS_PATH set?")


class Builder:
    def __init__(self, registry: dict[str, dict], name: str, description: str):
        self.registry = registry
        self.flow = empty_flow(name=name, description=description)
        self.keys = {
            "chat_input": find(registry, "input_output", "ChatInput"),
            "chat_output": find(registry, "input_output", "ChatOutput"),
            "llm": find(registry, "coworkai", "ConfiguredLLMComponent"),
            "agent": find(registry, "coworkai", "CoworkAIAgentComponent"),
            "orchestrator": find(registry, "coworkai", "OrchestratorAgentComponent"),
        }

    def add(self, kind: str, **fields) -> str:
        node_id = add_component(self.flow, self.keys[kind], self.registry)["id"]
        if fields:
            configure_component(self.flow, node_id, fields)
        return node_id

    def connect(self, source: str, output: str, target: str, field: str) -> None:
        add_connection(self.flow, source, output, target, field, registry=self.registry)

    def finish(self) -> dict:
        sync_dropdown_selected_outputs(self.flow)
        layout_flow(self.flow)
        return self.flow


def single_agent(registry: dict[str, dict]) -> dict:
    b = Builder(
        registry,
        "Single Q&A agent",
        "One agent on a Configured LLM. Pick an LLM Config on the Configured LLM block, then chat in the Playground.",
    )
    chat_input = b.add("chat_input")
    llm = b.add("llm")
    agent = b.add("agent", agent_name="Assistant", agent_role="Answers questions.", system_prompt=QA_INSTRUCTIONS)
    chat_output = b.add("chat_output")
    b.connect(llm, "model_output", agent, "model")
    b.connect(chat_input, "message", agent, "input_value")
    b.connect(agent, "response", chat_output, "input_value")
    return b.finish()


def supervisor_team(registry: dict[str, dict]) -> dict:
    b = Builder(
        registry,
        "Supervisor with three specialists",
        "An Orchestrator in Supervisor mode with a Researcher, an Analyst and a Writer. Pick an LLM Config on the "
        "Configured LLM block. Change Pattern on the Orchestrator to try Sequential, Router or Parallel + aggregate.",
    )
    chat_input = b.add("chat_input")
    llm = b.add("llm")
    orchestrator = b.add("orchestrator", agent_name="Supervisor", pattern="Supervisor")
    chat_output = b.add("chat_output")
    b.connect(llm, "model_output", orchestrator, "model")
    for fields in SPECIALISTS:
        agent = b.add("agent", **fields)
        b.connect(llm, "model_output", agent, "model")
        b.connect(agent, "agent", orchestrator, "agents")
    b.connect(chat_input, "message", orchestrator, "input_value")
    b.connect(orchestrator, "response", chat_output, "input_value")
    return b.finish()


TEMPLATES = {
    "single-qa-agent.json": single_agent,
    "supervisor-three-specialists.json": supervisor_team,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--out", default="templates")
    args = parser.parse_args()

    api_key = read_env(Path(args.env)).get("LANGFLOW_API_KEY", "")
    if not api_key:
        print("LANGFLOW_API_KEY is missing. Run scripts/create_langflow_key.py first.", file=sys.stderr)
        return 1
    registry = load_registry(args.url.rstrip("/"), api_key)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for filename, build in TEMPLATES.items():
        flow = build(registry)
        (out / filename).write_text(json.dumps(flow, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out / filename}: {len(flow['data']['nodes'])} blocks, {len(flow['data']['edges'])} connections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
