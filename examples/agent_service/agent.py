from __future__ import annotations

import inspect
import json
import os
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool as Tau2Tool
from tau2.registry import registry

from areal.experimental.agent_service.content import content_to_text, part_to_text
from areal.experimental.agent_service.types import (
    AgentRequest,
    AgentResponse,
    EventEmitter,
    Part,
)
from areal.utils import logging

logger = logging.getLogger("Tau2Agent")

TOOL_CALL_MEDIA = "application/x-tool-call"
TOOL_RESULT_MEDIA = "application/x-tool-result"


def _make_pydantic_tool(tau2_tool: Tau2Tool):
    fn = tau2_tool._func  # noqa: SLF001
    name = tau2_tool.name
    doc = tau2_tool.openai_schema["function"].get("description", name)

    async def _wrapper(**kwargs: Any) -> str:
        result = fn(**kwargs)
        if not isinstance(result, str):
            result = json.dumps(result, default=str)
        return result

    _wrapper.__name__ = name
    _wrapper.__qualname__ = name
    _wrapper.__doc__ = doc

    sig = inspect.signature(fn)
    params = [
        inspect.Parameter(
            pname,
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=param.default,
            annotation=param.annotation,
        )
        for pname, param in sig.parameters.items()
    ]
    _wrapper.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    if hasattr(fn, "__annotations__"):
        _wrapper.__annotations__ = {
            k: v for k, v in fn.__annotations__.items() if k != "return"
        }
    return _wrapper


def _think_tool_fn(thoughts: str) -> str:
    """Use this tool to think. Only use when necessary."""
    return "Your thoughts are recorded. Please continue your work."


class Tau2Agent:
    def __init__(self, config: dict | None = None, **kwargs: Any) -> None:
        config = config or {}
        tau2_cfg = config.get("tau2", {})
        agent_llm_cfg = config.get("agent_llm", {})

        self._domain = tau2_cfg.get("domain") or os.environ.get(
            "TAU2_DOMAIN", "airline"
        )
        add_thinking = tau2_cfg.get("add_thinking_tool", False)

        data_dir = tau2_cfg.get("data_dir") or os.environ.get("TAU2_DATA_DIR")
        if data_dir:
            os.environ["TAU2_DATA_DIR"] = data_dir

        env = self._build_environment()
        tau2_tools: list[Tau2Tool] = env.get_tools()
        if add_thinking:
            tau2_tools.append(Tau2Tool(_think_tool_fn))

        tools = [_make_pydantic_tool(t) for t in tau2_tools]
        system_prompt = env.get_policy()

        model_name = agent_llm_cfg.get("model", "openai:default")
        base_url = agent_llm_cfg.get("base_url")
        api_key = agent_llm_cfg.get("api_key", "unused")

        if base_url:
            model: Any = OpenAIChatModel(
                model_name.replace("openai:", ""),
                provider=OpenAIProvider(base_url=base_url, api_key=api_key),
            )
        else:
            model = model_name

        self._agent = Agent(model, system_prompt=system_prompt, tools=tools)

        logger.info(
            "Tau2Agent initialized (domain=%s, tools=%d, model=%s)",
            self._domain,
            len(tools),
            model_name,
        )

    def _build_environment(self) -> Environment:
        constructor = registry.get_env_constructor(self._domain)
        return constructor(**{"solo_mode": False})

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse:
        from pydantic_ai.messages import (
            ModelRequest,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
            UserPromptPart,
        )
        from pydantic_ai.messages import (
            ModelResponse as PAModelResponse,
        )

        message_history: list[ModelRequest | PAModelResponse] = []
        for msg in request.history:
            role = msg.role
            content_text = "".join(part_to_text(p) for p in msg.parts)

            if role == "user":
                message_history.append(
                    ModelRequest(parts=[UserPromptPart(content=content_text or "")])
                )
            elif role == "assistant" or role.startswith("agent"):
                tool_call_parts = [
                    p for p in msg.parts if p.media_type == TOOL_CALL_MEDIA
                ]
                if tool_call_parts:
                    parts = []
                    for p in tool_call_parts:
                        d = p.data or {}
                        parts.append(
                            ToolCallPart(
                                tool_name=d.get("name", ""),
                                args=d.get("arguments", ""),
                                tool_call_id=d.get("call_id", ""),
                            )
                        )
                    message_history.append(PAModelResponse(parts=parts))
                elif content_text:
                    message_history.append(
                        PAModelResponse(parts=[TextPart(content=content_text)])
                    )
            elif role == "tool":
                tool_result_parts = [
                    p for p in msg.parts if p.media_type == TOOL_RESULT_MEDIA
                ]
                for p in tool_result_parts:
                    d = p.data or {}
                    message_history.append(
                        ModelRequest(
                            parts=[
                                ToolReturnPart(
                                    tool_name=d.get("name", ""),
                                    content=d.get("result", ""),
                                    tool_call_id=d.get("call_id", ""),
                                )
                            ]
                        )
                    )
                if not tool_result_parts and content_text:
                    message_history.append(
                        ModelRequest(
                            parts=[
                                ToolReturnPart(
                                    tool_name="",
                                    content=content_text,
                                    tool_call_id="",
                                )
                            ]
                        )
                    )

        user_message = "".join(part_to_text(p) for p in request.message.parts)
        result = await self._agent.run(
            user_message,
            message_history=message_history,
        )

        final_text = str(result.output) if result.output else ""

        tool_calls: list[dict[str, Any]] = []
        for msg in result.new_messages():
            if not hasattr(msg, "parts"):
                continue
            for part in msg.parts:
                kind = getattr(part, "part_kind", "")
                if kind == "tool-call":
                    name = getattr(part, "tool_name", "")
                    args = getattr(part, "args", "")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    call_id = str(getattr(part, "tool_call_id", "") or "")
                    await emitter.emit_part(
                        Part(
                            data={
                                "name": name,
                                "arguments": str(args),
                                "call_id": call_id,
                            },
                            media_type=TOOL_CALL_MEDIA,
                        )
                    )
                    tool_calls.append({"name": name, "arguments": args})
                elif kind == "tool-return":
                    name = getattr(part, "tool_name", "")
                    content = str(getattr(part, "content", ""))
                    call_id = str(getattr(part, "tool_call_id", "") or "")
                    await emitter.emit_part(
                        Part(
                            data={"name": name, "result": content, "call_id": call_id},
                            media_type=TOOL_RESULT_MEDIA,
                        )
                    )

        if final_text:
            await emitter.emit_part(Part(text=final_text))

        return AgentResponse(
            output=[Part(text=final_text)],
            history_text=final_text,
            metadata={"tool_calls": tool_calls},
        )
