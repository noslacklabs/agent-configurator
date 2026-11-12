"""LangChain chat model that talks to one LLM Configurator config by name.

Langflow's OpenAI-compatible block cannot be pointed at the configurator: the
route is ``/llm/{name}/chat`` and a ``model`` field in the body makes it fail.
This class posts the OpenAI-style body without ``model`` and holds no key. The
configurator makes the provider call.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import InvalidToolCall, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from agent_configurator.client import ConfiguratorClient, ConfiguratorError


class ConfiguredChatModel(BaseChatModel):
    """Chat model backed by ``POST /llm/{config_name}/chat`` on the LLM Configurator."""

    config_name: str
    """``full_name`` of the config in the LLM Configurator. The only identifier a flow stores."""
    base_url: str | None = None
    """Configurator base URL. Defaults to the ``LLM_CONFIGURATOR_URL`` setting."""
    agent_id: str | None = None
    """Sent as ``X-Agent-Id`` so the configurator's usage log can break usage down per agent."""
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float = 120.0
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "coworkai-llm-configurator"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"config_name": self.config_name, "temperature": self.temperature, "max_tokens": self.max_tokens}

    def with_agent_id(self, agent_id: str | None) -> ConfiguredChatModel:
        """A copy of this model that reports ``agent_id`` in the usage log."""
        return self.model_copy(update={"agent_id": agent_id or self.agent_id})

    # -- request / response translation ------------------------------------

    def _client(self) -> ConfiguratorClient:
        return ConfiguratorClient(self.base_url, timeout=self.timeout)

    def _body(self, messages: list[BaseMessage], stop: list[str] | None, kwargs: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": convert_to_openai_messages(messages)}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        if stop:
            body["stop"] = stop
        body.update(self.extra_body)
        # Anything bound with .bind()/.bind_tools() arrives here: tools, tool_choice, response_format...
        body.update({key: value for key, value in kwargs.items() if value is not None})
        return body

    def _result(self, data: dict[str, Any]) -> ChatResult:
        choices = data.get("choices") or []
        if not choices:
            raise ConfiguratorError(f"LLM config '{self.config_name}' returned no choices")
        message = choices[0].get("message") or {}

        tool_calls: list[ToolCall] = []
        invalid: list[InvalidToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            arguments = function.get("arguments")
            try:
                args = json.loads(arguments) if isinstance(arguments, str) and arguments.strip() else (arguments or {})
                if not isinstance(args, dict):
                    raise TypeError("tool arguments are not an object")
            except (ValueError, TypeError) as exc:
                invalid.append(InvalidToolCall(
                    name=function.get("name"), args=str(arguments), id=raw.get("id"), error=str(exc),
                    type="invalid_tool_call",
                ))
                continue
            tool_calls.append(ToolCall(name=function.get("name") or "", args=args, id=raw.get("id"), type="tool_call"))

        usage = data.get("usage") or {}
        usage_metadata = None
        if usage:
            prompt = int(usage.get("prompt_tokens") or 0)
            completion = int(usage.get("completion_tokens") or 0)
            usage_metadata = UsageMetadata(
                input_tokens=prompt,
                output_tokens=completion,
                total_tokens=int(usage.get("total_tokens") or prompt + completion),
            )

        content = message.get("content")
        ai_message = AIMessage(
            content=content if isinstance(content, (str, list)) else "",
            tool_calls=tool_calls,
            invalid_tool_calls=invalid,
            usage_metadata=usage_metadata,
            response_metadata={
                "config_name": self.config_name,
                "model_used": data.get("model_used"),
                "finish_reason": choices[0].get("finish_reason"),
            },
        )
        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output={"model_used": data.get("model_used"), "token_usage": usage},
        )

    # -- LangChain hooks ---------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        data = self._client().chat(self.config_name, self._body(messages, stop, kwargs), self.agent_id)
        return self._result(data)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        data = await self._client().achat(self.config_name, self._body(messages, stop, kwargs), self.agent_id)
        return self._result(data)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | dict[str, Any] | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice is not False:
            if tool_choice is True or tool_choice == "any":
                tool_choice = "required"
            elif isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted, **kwargs)
