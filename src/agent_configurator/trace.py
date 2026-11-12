"""Turn what an agent did into plain records the evaluator can score.

When Langflow runs an agent as a tool, the tool hands back only the agent's
final text, so the tool calls made inside that agent are lost. The CoworkAI
Agent block instead returns an *agent report*: a dict marked with
``coworkai_agent`` that carries the answer and the agent's own tool calls. The
functions here build and read those reports.
"""

from __future__ import annotations

import json
import re
from typing import Any

REPORT_MARKER = "coworkai_agent"


def slugify(name: str, fallback: str = "agent") -> str:
    """A tool-safe name (``^[a-zA-Z0-9_-]+$``) for an agent."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    return slug or fallback


def as_report(value: Any) -> dict[str, Any] | None:
    """Return ``value`` as an agent report if it is one (dict or JSON text), else None."""
    if isinstance(value, str):
        text = value.strip()
        if not (text.startswith("{") and REPORT_MARKER in text):
            return None
        try:
            value = json.loads(text)
        except ValueError:
            return None
    if isinstance(value, dict) and value.get(REPORT_MARKER):
        return value
    return None


def unwrap_tool_output(output: Any) -> Any:
    """The tool's actual result.

    Langflow 1.13 stores a serialised LangChain ``ToolMessage`` as the block's
    output: ``{"type": "tool", "content": "<text>", "artifact": <raw result>}``.
    The raw result is preferred, the text is the fallback.
    """
    if isinstance(output, dict) and output.get("type") == "tool" and "content" in output:
        artifact = output.get("artifact")
        return artifact if artifact is not None else output.get("content")
    content = getattr(output, "content", None)
    if content is not None and getattr(output, "type", None) == "tool":
        artifact = getattr(output, "artifact", None)
        return artifact if artifact is not None else content
    return output


def _block_fields(block: Any) -> dict[str, Any] | None:
    """Name, input and output of a ``tool_use`` content block, object or dict."""
    if isinstance(block, dict):
        if block.get("type") != "tool_use":
            return None
        return {
            "name": block.get("name"),
            "input": block.get("tool_input") if "tool_input" in block else block.get("input"),
            "output": unwrap_tool_output(block.get("output")),
            "error": block.get("error"),
        }
    if getattr(block, "type", None) != "tool_use":
        return None
    return {
        "name": getattr(block, "name", None),
        "input": getattr(block, "tool_input", None),
        "output": unwrap_tool_output(getattr(block, "output", None)),
        "error": getattr(block, "error", None),
    }


def _walk(blocks: Any):
    """Yield content blocks, looking inside the older nested ``contents`` layout too."""
    for block in blocks or []:
        nested = block.get("contents") if isinstance(block, dict) else getattr(block, "contents", None)
        is_tool = (block.get("type") if isinstance(block, dict) else getattr(block, "type", None)) == "tool_use"
        if nested and not is_tool:
            yield from _walk(nested)
        else:
            yield block


def tool_calls_from_blocks(blocks: Any, agent: str | None = None) -> list[dict[str, Any]]:
    """Flat list of tool calls from a message's content blocks.

    A call to a CoworkAI agent is kept as one entry (name = the agent) and is
    followed by the tool calls that agent made, each tagged with ``agent``.
    """
    calls: list[dict[str, Any]] = []
    for block in _walk(blocks):
        fields = _block_fields(block)
        if not fields or not fields["name"]:
            continue
        report = as_report(fields["output"])
        entry: dict[str, Any] = {
            "name": fields["name"],
            "input_parameters": fields["input"] or {},
            "output": report.get("output") if report else fields["output"],
        }
        if fields["error"]:
            entry["error"] = fields["error"]
        if agent:
            entry["agent"] = agent
        if report:
            entry["is_agent"] = True
        calls.append(entry)
        if report:
            for nested in report.get("tools_called") or []:
                calls.append({"agent": report[REPORT_MARKER], **nested} if "agent" not in nested else dict(nested))
    return calls


def build_report(agent: str, task: str, output: str, blocks: Any) -> dict[str, Any]:
    """The report an agent returns when it is run by an orchestrator or as a tool."""
    return {
        REPORT_MARKER: agent,
        "input": task,
        "output": output,
        "tools_called": tool_calls_from_blocks(blocks, agent=agent),
    }
