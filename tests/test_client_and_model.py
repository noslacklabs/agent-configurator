from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agent_configurator.chat_model import ConfiguredChatModel
from agent_configurator.client import (
    ConfigNotFoundError,
    ConfiguratorClient,
    ConfiguratorError,
    ConfiguratorUnavailableError,
    GuardrailBlockedError,
    RateLimitedError,
    RequestRejectedError,
)
from agent_configurator.embeddings import ConfiguredEmbeddings


@tool
def get_weather(city: str) -> str:
    """Look up the weather for a city."""
    return f"Sunny in {city}"


# -- client ---------------------------------------------------------------


def test_lists_only_active_configs(configurator):
    client = ConfiguratorClient()
    names = client.config_names()
    assert "demo_main" in names
    assert "demo_disabled" not in names
    assert client.config_names("embedding") == ["demo_main"]
    assert client.max_tokens_cap("demo_chatonly") == 1024


def test_only_allowed_routes_are_ever_requested(configurator):
    client = ConfiguratorClient()
    client.config_names()
    client.chat("demo_main", {"messages": [{"role": "user", "content": "hi"}]})
    client.embeddings("demo_main", ["hi"])
    for method, path in configurator.app.state.requests:
        assert (method, path) == ("GET", "/configs") or (
            method == "POST" and path.startswith("/llm/") and path.rsplit("/", 1)[1] in ("chat", "embeddings")
        ), f"unexpected request {method} {path}"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/catalog/global_models"),
        ("POST", "/configs"),
        ("PUT", "/configs/demo_main"),
        ("DELETE", "/configs/demo_main"),
        ("GET", "/configs/demo_main"),
        ("POST", "/llm/demo_main/../../catalog/global_models"),
    ],
)
def test_other_routes_are_refused_before_any_request(configurator, method, path):
    client = ConfiguratorClient()
    with pytest.raises(ConfiguratorError, match="Refusing to call"):
        client._request(method, path)
    assert configurator.app.state.requests == []


def test_model_and_stream_are_never_sent(configurator):
    ConfiguratorClient().chat(
        "demo_main", {"model": "gpt-x", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    )
    body = configurator.app.state.chat_calls[-1]["body"]
    assert "model" not in body
    assert "stream" not in body


def test_unknown_and_disabled_configs_are_not_found(configurator):
    client = ConfiguratorClient()
    for name in ("nope", "demo_disabled"):
        with pytest.raises(ConfigNotFoundError):
            client.chat(name, {"messages": [{"role": "user", "content": "hi"}]})


def test_guardrail_block_is_an_error_not_an_empty_answer(configurator):
    with pytest.raises(GuardrailBlockedError, match="forbidden") as caught:
        ConfiguratorClient().chat("demo_guarded", {"messages": [{"role": "user", "content": "a forbidden word"}]})
    assert "forbidden" in caught.value.reason


def test_max_tokens_above_cap_is_rejected(configurator):
    with pytest.raises(RequestRejectedError, match="exceeds config cap"):
        ConfiguratorClient().chat(
            "demo_chatonly", {"max_tokens": 5000, "messages": [{"role": "user", "content": "hi"}]}
        )


def test_rate_limit_is_retried_then_reported(configurator):
    client = ConfiguratorClient(rate_limit_retries=2, rate_limit_backoff=(0.01,))
    body = {"messages": [{"role": "user", "content": "hi"}]}
    client.chat("demo_lowrpm", body)
    client.chat("demo_lowrpm", body)
    with pytest.raises(RateLimitedError, match="requests-per-minute"):
        client.chat("demo_lowrpm", body)
    attempts = [r for r in configurator.app.state.requests if r[1] == "/llm/demo_lowrpm/chat"]
    assert len(attempts) == 2 + 3  # two that passed, then one try and two retries


def test_unreachable_configurator_is_reported_clearly():
    client = ConfiguratorClient("http://127.0.0.1:9", timeout=2)
    with pytest.raises(ConfiguratorUnavailableError, match="Could not reach the LLM Configurator"):
        client.list_configs()


# -- chat model -----------------------------------------------------------


def test_plain_chat_sends_agent_id_and_reads_usage(configurator):
    model = ConfiguredChatModel(config_name="demo_main", agent_id="researcher", temperature=0.2)
    reply = model.invoke([SystemMessage("be brief"), HumanMessage("hello there")])
    assert reply.content == "Echo: hello there"
    assert reply.usage_metadata["total_tokens"] > 0
    assert reply.response_metadata["model_used"] == "mock/scripted-1"
    call = configurator.app.state.chat_calls[-1]
    assert call["agent_id"] == "researcher"
    assert call["body"]["temperature"] == 0.2
    assert [m["role"] for m in call["body"]["messages"]] == ["system", "user"]


def test_bind_tools_round_trip(configurator):
    model = ConfiguredChatModel(config_name="demo_main").bind_tools([get_weather])
    first = model.invoke([HumanMessage("please get weather for Paris")])
    assert isinstance(first, AIMessage)
    assert first.tool_calls[0]["name"] == "get_weather"
    assert first.tool_calls[0]["args"] == {"city": "please get weather for Paris"}

    sent = configurator.app.state.chat_calls[-1]["body"]
    assert sent["tools"][0]["function"]["name"] == "get_weather"

    result = get_weather.invoke(first.tool_calls[0])
    assert isinstance(result, ToolMessage)
    final = model.invoke([HumanMessage("please get weather for Paris"), first, result])
    assert final.content.startswith("Final answer: Sunny in")
    replayed = configurator.app.state.chat_calls[-1]["body"]["messages"]
    assert [m["role"] for m in replayed] == ["user", "assistant", "tool"]
    assert replayed[1]["tool_calls"][0]["function"]["name"] == "get_weather"


async def test_async_chat(configurator):
    reply = await ConfiguredChatModel(config_name="demo_main").ainvoke("ping")
    assert reply.content == "Echo: ping"


def test_streaming_falls_back_to_one_chunk(configurator):
    chunks = list(ConfiguredChatModel(config_name="demo_main").stream("ping"))
    assert "".join(chunk.content for chunk in chunks) == "Echo: ping"
    assert "stream" not in configurator.app.state.chat_calls[-1]["body"]


def test_with_agent_id_copies_the_model(configurator):
    base = ConfiguredChatModel(config_name="demo_main")
    named = base.with_agent_id("writer")
    assert base.agent_id is None
    named.invoke("hi")
    assert configurator.app.state.chat_calls[-1]["agent_id"] == "writer"


# -- embeddings -----------------------------------------------------------


def test_embeddings(configurator):
    embeddings = ConfiguredEmbeddings("demo_main", batch_size=2)
    vectors = embeddings.embed_documents(["alpha", "beta", "gamma"])
    assert len(vectors) == 3
    assert len(embeddings.embed_query("alpha")) == len(vectors[0])
    assert embeddings.embed_query("alpha") == vectors[0]


def test_embeddings_need_an_embedding_model(configurator):
    with pytest.raises(RequestRejectedError, match="no embedding-capable model"):
        ConfiguredEmbeddings("demo_chatonly").embed_query("alpha")
