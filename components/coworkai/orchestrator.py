import asyncio
import uuid

from agent_configurator.chat_model import ConfiguredChatModel
from agent_configurator.trace import as_report, slugify, unwrap_tool_output
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from lfx.custom.custom_component.component import Component
from lfx.io import DropdownInput, HandleInput, IntInput, MessageTextInput, MultilineInput, Output, StrInput
from lfx.schema.content_types import ToolContent
from lfx.schema.message import Message
from lfx.utils.constants import MESSAGE_SENDER_AI

SUPERVISOR = "Supervisor"
SEQUENTIAL = "Sequential"
ROUTER = "Router"
PARALLEL = "Parallel + aggregate"

DEFAULT_SUPERVISOR_PROMPT = (
    "You coordinate a team of agents. Each agent is available to you as a tool. Decide which agent should "
    "handle each part of the request, give it everything it needs in one message, and call agents again "
    "if their answers are incomplete. When the work is done, give the user the final answer yourself."
)
DEFAULT_AGGREGATE_PROMPT = (
    "Several agents answered the same request independently. Combine their answers into one answer for "
    "the user. Keep what is correct and useful, resolve contradictions, and do not mention the agents."
)


class OrchestratorAgentComponent(Component):
    display_name = "Orchestrator Agent"
    description = (
        "Runs several agents together. Connect the Agent output of each agent, pick a pattern, and "
        "every agent call shows as a step with the agent's name, its input and its output."
    )
    icon = "network"
    name = "CoworkAIOrchestrator"

    inputs = [
        StrInput(
            name="agent_name",
            display_name="Orchestrator Name",
            info="Sent to the usage log for the orchestrator's own LLM calls.",
            value="Orchestrator",
        ),
        HandleInput(
            name="model",
            display_name="Configured LLM",
            info="Connect a Configured LLM block. Sequential does not call it.",
            input_types=["LanguageModel"],
        ),
        HandleInput(
            name="agents",
            display_name="Agents",
            info="Connect the Agent output of one or more Agent blocks. For Sequential, connection order is run order.",
            input_types=["Tool"],
            is_list=True,
            required=True,
        ),
        DropdownInput(
            name="pattern",
            display_name="Pattern",
            info=(
                "Supervisor: the orchestrator's LLM decides which agents to call, possibly several times. "
                "Sequential: agents run in order, each receiving the previous answer. "
                "Router: one LLM call picks exactly one agent. "
                "Parallel + aggregate: all agents run at once, then one LLM call merges the answers."
            ),
            options=[SUPERVISOR, SEQUENTIAL, ROUTER, PARALLEL],
            value=SUPERVISOR,
        ),
        MessageTextInput(
            name="input_value",
            display_name="Input",
            info="The request to handle.",
        ),
        MultilineInput(
            name="system_prompt",
            display_name="Orchestrator Instructions",
            info="Optional. Replaces the default instructions for Supervisor and for the Parallel merge step.",
            value="",
        ),
        IntInput(
            name="max_iterations",
            display_name="Max Iterations",
            info="Supervisor only: the most LLM calls the orchestrator may make in one run.",
            value=10,
            advanced=True,
        ),
        StrInput(
            name="solo_agent",
            display_name="Solo Agent",
            info=(
                "Leave empty. When set to an agent's name, the pattern is skipped and the input goes straight "
                "to that one agent. The evaluator sets this per run to score a single agent inside the flow."
            ),
            value="",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Response", name="response", method="orchestrate"),
    ]

    # -- helpers ----------------------------------------------------------

    def _label(self) -> str:
        return (self.agent_name or "").strip() or "Orchestrator"

    def _llm(self) -> ConfiguredChatModel:
        model = self.model
        if not isinstance(model, ConfiguredChatModel):
            msg = f"{self._label()}: connect a Configured LLM block. The {self.pattern} pattern needs it."
            raise ValueError(msg)  # noqa: TRY004
        return model.with_agent_id(self._label())

    def _agent_tools(self) -> list:
        connected = self.agents
        if connected is None:
            connected = []
        if not isinstance(connected, list):
            connected = [connected]
        tools = []
        for item in connected:
            tools.extend(item if isinstance(item, list) else [item])
        tools = [tool for tool in tools if tool is not None]
        if not tools:
            msg = f"{self._label()}: connect at least one agent to the Agents port."
            raise ValueError(msg)
        # Langflow renames clashing tools by appending the block id, so compare the names people gave.
        names = [self._display(tool).lower() for tool in tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            msg = f"{self._label()}: agent names must be unique. Repeated: {', '.join(duplicates)}"
            raise ValueError(msg)
        return tools

    @staticmethod
    def _display(tool) -> str:
        return (tool.metadata or {}).get("coworkai_agent") or tool.name

    @staticmethod
    def _report(tool, task: str, raw) -> dict:
        """Normalise whatever a connected tool returned into an agent report."""
        value = unwrap_tool_output(raw)
        report = as_report(value)
        if report:
            return report
        label = (tool.metadata or {}).get("coworkai_agent") or tool.name
        return {"coworkai_agent": label, "input": task, "output": value if isinstance(value, str) else str(value),
                "tools_called": []}

    async def _call(self, tool, task: str) -> dict:
        result = await tool.ainvoke(
            {"type": "tool_call", "name": tool.name, "args": {"input_value": task}, "id": f"call_{uuid.uuid4().hex[:12]}"}
        )
        if isinstance(result, ToolMessage) and result.status == "error":
            msg = f"Agent '{self._display(tool)}' failed: {result.content}"
            raise RuntimeError(msg)
        return self._report(tool, task, result)

    @staticmethod
    def _step(tool_name: str, report: dict) -> ToolContent:
        return ToolContent(
            name=tool_name,
            tool_input={"input_value": report.get("input", "")},
            output=report,
            header={"title": f"Agent **{report.get('coworkai_agent', tool_name)}**", "icon": "Bot"},
        )

    # -- patterns ---------------------------------------------------------

    async def _run_solo(self, tools: list, task: str):
        wanted = (self.solo_agent or "").strip()
        by_name = {}
        for tool in tools:
            by_name[tool.name.lower()] = tool
            by_name[self._display(tool).lower()] = tool
            by_name[slugify(self._display(tool)).lower()] = tool
        tool = by_name.get(wanted.lower()) or by_name.get(slugify(wanted).lower())
        if tool is None:
            available = ", ".join(self._display(item) for item in tools)
            msg = f"{self._label()}: solo agent '{wanted}' is not connected. Connected agents: {available}"
            raise ValueError(msg)
        # "solo" lets the evaluator confirm the switch really took effect. If the tweak that sets
        # solo_agent stops matching this block, the flow would otherwise run its normal pattern
        # and be scored under one agent's name.
        report = {**(await self._call(tool, task)), "solo": True}
        return report["output"], [self._step(tool.name, report)]

    async def _run_sequential(self, tools: list, task: str):
        steps = []
        current = task
        for tool in tools:
            report = await self._call(tool, current)
            steps.append(self._step(tool.name, report))
            current = report["output"]
        return current, steps

    async def _run_router(self, tools: list, task: str):
        names = [tool.name for tool in tools]
        listing = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        system = (
            "You route a request to the one agent best suited to handle it.\n"
            f"Agents:\n{listing}\n"
            f"Reply with exactly one of: {', '.join(names)}\n"
            "Reply with the name only."
        )
        reply = await self._llm().ainvoke([SystemMessage(system), HumanMessage(task)])
        answer = (reply.content if isinstance(reply.content, str) else str(reply.content)).strip().lower()
        # Longest name first so "billing_support" is not mistaken for "billing".
        chosen = next((tool for tool in sorted(tools, key=lambda t: -len(t.name)) if tool.name.lower() in answer), None)
        if chosen is None:
            msg = f"{self._label()}: the router answered '{reply.content}', which is not one of: {', '.join(names)}"
            raise ValueError(msg)
        report = await self._call(chosen, task)
        return report["output"], [self._step(chosen.name, report)]

    async def _run_parallel(self, tools: list, task: str):
        reports = await asyncio.gather(*(self._call(tool, task) for tool in tools))
        steps = [self._step(tool.name, report) for tool, report in zip(tools, reports, strict=True)]
        answers = "\n\n".join(f"[{report['coworkai_agent']}]\n{report['output']}" for report in reports)
        system = (self.system_prompt or "").strip() or DEFAULT_AGGREGATE_PROMPT
        reply = await self._llm().ainvoke(
            [SystemMessage(system), HumanMessage(f"Request:\n{task}\n\nAnswers:\n{answers}")]
        )
        return (reply.content if isinstance(reply.content, str) else str(reply.content)), steps

    async def _run_supervisor(self, tools: list, task: str):
        system = (self.system_prompt or "").strip() or DEFAULT_SUPERVISOR_PROMPT
        max_calls = max(1, int(self.max_iterations or 10))
        graph = create_agent(model=self._llm(), tools=tools, system_prompt=system)
        # Each iteration is one model step plus one tool step in the graph.
        result = await graph.ainvoke({"messages": [HumanMessage(task)]}, config={"recursion_limit": 2 * max_calls + 1})

        by_name = {tool.name: tool for tool in tools}
        requested = {}
        steps = []
        final = ""
        for message in result["messages"]:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    requested[call["id"]] = call
                if isinstance(message.content, str) and message.content:
                    final = message.content
            elif isinstance(message, ToolMessage):
                call = requested.get(message.tool_call_id, {})
                name = message.name or call.get("name") or "tool"
                task_sent = (call.get("args") or {}).get("input_value", "")
                tool = by_name.get(name)
                report = self._report(tool, task_sent, message) if tool else {
                    "coworkai_agent": name, "input": task_sent, "output": str(message.content), "tools_called": []}
                steps.append(self._step(name, report))
        return final, steps

    # -- output -----------------------------------------------------------

    async def orchestrate(self) -> Message:
        task = self.input_value
        task = task if isinstance(task, str) else getattr(task, "text", str(task))
        tools = self._agent_tools()

        if (self.solo_agent or "").strip():
            text, steps = await self._run_solo(tools, task)
        elif self.pattern == SEQUENTIAL:
            text, steps = await self._run_sequential(tools, task)
        elif self.pattern == ROUTER:
            text, steps = await self._run_router(tools, task)
        elif self.pattern == PARALLEL:
            text, steps = await self._run_parallel(tools, task)
        else:
            text, steps = await self._run_supervisor(tools, task)

        session_id = self.graph.session_id if hasattr(self, "graph") else None
        message = Message(
            text=text,
            sender=MESSAGE_SENDER_AI,
            sender_name=self._label(),
            session_id=session_id,
            content_blocks=steps,
            properties={"icon": "Network", "state": "complete"},
        )
        self._orchestration_result = message
        self.status = message
        return message
