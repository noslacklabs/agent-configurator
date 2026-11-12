from agent_configurator.chat_model import ConfiguredChatModel
from agent_configurator.trace import build_report, slugify
from lfx.base.tools.component_tool import ComponentStructuredTool, _build_output_async_function
from lfx.base.tools.constants import TOOL_OUTPUT_NAME
from lfx.components.models_and_agents.agent import AgentComponent
from lfx.io import HandleInput, MultilineInput, Output, StrInput
from lfx.schema.data import Data
from pydantic import BaseModel, Field

# Inputs of Langflow's Agent that pick a provider and take its key. They are
# replaced by one port that only accepts a Configured LLM.
_PROVIDER_INPUTS = {"model", "api_key", "base_url_ibm_watsonx", "project_id"}


class AgentTask(BaseModel):
    input_value: str = Field(description="The task or question for this agent, with all the context it needs.")


class CoworkAIAgentComponent(AgentComponent):
    """Langflow's Agent, running on a Configured LLM and reporting its work to orchestrators.

    Tools, MCP, memory, max iterations and output format all come from the
    built-in Agent unchanged. This subclass depends on its internals, so it is
    tied to the pinned Langflow version and must be re-tested on upgrade.
    """

    display_name = "Agent"
    description = (
        "An agent that runs on a Configured LLM. Give it a name and a role, connect tools, and either "
        "chat with it directly or connect its Agent output to an Orchestrator."
    )
    icon = "bot"
    name = "CoworkAIAgent"

    inputs = [
        StrInput(
            name="agent_name",
            display_name="Agent Name",
            info="Shown on each step this agent takes, used as its tool name, and sent to the usage log.",
            value="Agent",
            required=True,
        ),
        MultilineInput(
            name="agent_role",
            display_name="Role / Description",
            info="One or two sentences on what this agent is for. Orchestrators use it to decide when to call it.",
            value="",
        ),
        HandleInput(
            name="model",
            display_name="Configured LLM",
            info="Connect a Configured LLM block.",
            input_types=["LanguageModel"],
            required=True,
        ),
        *[item for item in AgentComponent.inputs if item.name not in _PROVIDER_INPUTS],
    ]

    outputs = [
        *AgentComponent.outputs,
        Output(
            name="agent",
            display_name="Agent",
            info="Connect to an Orchestrator's Agents port.",
            method="as_agent",
            types=["Tool"],
        ),
    ]

    # -- model ------------------------------------------------------------

    def _label(self) -> str:
        return (self.agent_name or "").strip() or "Agent"

    def _get_llm(self):
        model = self.model
        if not isinstance(model, ConfiguredChatModel):
            msg = f"{self._label()}: connect a Configured LLM block to the Configured LLM port."
            raise ValueError(msg)  # noqa: TRY004
        return model.with_agent_id(self._label())

    async def update_build_config(self, build_config, field_value, field_name=None):
        # The built-in Agent refreshes its provider picker here. There is none.
        return build_config

    # -- running as part of an orchestration --------------------------------

    async def run_for_orchestrator(self) -> Data:
        """Run the agent and report the answer together with the tool calls it made."""
        task = self.input_value
        task_text = task if isinstance(task, str) else getattr(task, "text", str(task))
        message = await self.message_response()
        return Data(data=build_report(self._label(), task_text, message.text or "", message.content_blocks))

    def _agent_tool(self) -> ComponentStructuredTool:
        name = slugify(self._label())
        description = (self.agent_role or "").strip() or f"Ask the {self._label()} agent to handle a task."
        return ComponentStructuredTool(
            name=name,
            description=description,
            coroutine=_build_output_async_function(
                self, self.run_for_orchestrator, self.get_event_manager(), TOOL_OUTPUT_NAME
            ),
            source_component=self,
            args_schema=AgentTask,
            handle_tool_error=True,
            callbacks=self.get_langchain_callbacks(),
            tags=[name],
            metadata={
                "display_name": self._label(),
                "display_description": description,
                "source_id": self.get_id(),
                "coworkai_agent": self._label(),
            },
        )

    async def as_agent(self) -> ComponentStructuredTool:
        tool = self._agent_tool()
        self.status = f"Agent '{self._label()}' is ready for an Orchestrator."
        return tool

    async def _get_tools(self):
        # Tool Mode on this block gives the same tool as the Agent output.
        return [self._agent_tool()]
