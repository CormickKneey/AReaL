"""Public types for the Agent Service protocol.

Inspired by Google A2A (Part as universal content atom), ACP (role-based
Message with MIME-typed parts), and OpenAI Responses (discriminated output
union).  Designed for extensibility: new content kinds are added as new
``Part`` instances (with appropriate ``media_type``), not new fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Part:
    """Universal content atom — carries one piece of content in any modality.

    Exactly one of ``text``, ``data``, ``file_url``, or ``file_bytes``
    should be set.  ``media_type`` describes the content format.

    Examples::

        # Plain text
        Part(text="Hello world")

        # HTML fragment
        Part(text="<div>rich content</div>", media_type="text/html")

        # Tool call (structured data with sentinel media_type)
        Part(
            data={"name": "search", "arguments": '{"q": "AI"}'},
            media_type="application/x-tool-call",
        )

        # Tool result
        Part(
            data={"name": "search", "result": '{"hits": 42}'},
            media_type="application/x-tool-result",
        )

        # Image by URL reference
        Part(file_url="https://example.com/cat.png", media_type="image/png")
    """

    text: str | None = None
    data: dict[str, Any] | None = None
    file_url: str | None = None
    file_bytes: bytes | None = None
    media_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One communication turn between user and agent.

    ``role`` follows A2A/ACP conventions: ``"user"``, ``"agent"``,
    ``"agent/{name}"`` (named sub-agent), or ``"tool"``.

    Examples::

        # Simple user message
        Message(role="user", parts=[Part(text="What is CRISPR?")])

        # Agent response with HTML output
        Message(
            role="agent",
            parts=[Part(text="<html>...</html>", media_type="text/html")],
        )

        # Multimodal user message (text + image)
        Message(
            role="user",
            parts=[
                Part(text="Describe this image"),
                Part(file_url="https://example.com/photo.jpg", media_type="image/jpeg"),
            ],
        )
    """

    role: str
    parts: list[Part]
    message_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRequest:
    """Inbound request to an agent worker.

    ``session_id`` groups multi-turn conversations.  ``history`` carries
    prior turns managed by the DataProxy.  ``config`` is an open dict
    for agent-specific parameters (LLM model, temperature, etc.).

    Examples::

        AgentRequest(
            message=Message(role="user", parts=[Part(text="Hello")]),
            session_id="sess-abc123",
        )

        AgentRequest(
            message=Message(role="user", parts=[Part(text="Follow up")]),
            session_id="sess-abc123",
            history=[
                Message(role="user", parts=[Part(text="Hello")]),
                Message(role="agent", parts=[Part(text="Hi there!")]),
            ],
            config={"model": "gpt-4o", "temperature": 0.7},
        )
    """

    message: Message
    session_id: str = ""
    history: list[Message] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, body: dict[str, Any]) -> "AgentRequest":
        """Construct from wire JSON (DataProxy → Worker).

        Handles both new-style (``message`` as Message dict) and legacy
        flat payloads (``message`` as str, ``session_key``, ``metadata``).
        """
        raw_message = body.get("message", "")
        if isinstance(raw_message, dict) and "role" in raw_message:
            message = Message(
                role=raw_message.get("role", "user"),
                parts=[
                    Part(**p) if isinstance(p, dict) else Part(text=str(p))
                    for p in raw_message.get("parts", [])
                ],
                message_id=raw_message.get("message_id", ""),
                metadata=raw_message.get("metadata", {}),
            )
        else:
            if isinstance(raw_message, str):
                parts = [Part(text=raw_message)] if raw_message else []
            elif isinstance(raw_message, list):
                parts = [Part(data=block) for block in raw_message]
            else:
                parts = [Part(text=str(raw_message))] if raw_message else []
            message = Message(role="user", parts=parts)

        raw_history = body.get("history", [])
        history: list[Message] = []
        for item in raw_history:
            if isinstance(item, dict) and "role" in item:
                if "parts" in item:
                    h_parts = [
                        Part(**p) if isinstance(p, dict) else Part(text=str(p))
                        for p in item["parts"]
                    ]
                else:
                    content = item.get("content", "")
                    h_parts = [Part(text=str(content))] if content else []
                history.append(
                    Message(
                        role=item.get("role", "user"),
                        parts=h_parts,
                        message_id=item.get("message_id", ""),
                        metadata=item.get("metadata", {}),
                    )
                )

        legacy_metadata = dict(body.get("metadata", {}) or {})
        config = dict(body.get("config", {}) or {})
        key_map = {
            "input": "input",
            "instructions": "instructions",
            "model": "model",
            "tools": "tools",
            "idempotencyKey": "idempotency_key",
        }
        for old_key, new_key in key_map.items():
            val = legacy_metadata.pop(old_key, None)
            if val is not None and new_key not in config:
                config[new_key] = val
        config.update(legacy_metadata)

        return cls(
            message=message,
            session_id=body.get("session_id", body.get("session_key", "")),
            history=history,
            config=config,
        )


@dataclass
class AgentResponse:
    """Outbound response from an agent worker.

    ``output`` carries the final content as a list of ``Part`` objects.
    ``history_text`` is an optional plain-text summary stored in session
    history (useful when ``output`` contains HTML or binary content).

    Examples::

        # Simple text response
        AgentResponse(output=[Part(text="The answer is 42.")])

        # HTML response with plain-text history
        AgentResponse(
            output=[Part(text="<div>...</div>", media_type="text/html")],
            history_text="The answer is 42.",
        )

        # Failed response
        AgentResponse(status="failed", error="Model timeout")

        # Agent needs more input
        AgentResponse(
            status="input_required",
            output=[Part(text="Which airport do you mean?")],
        )
    """

    status: str = "completed"
    output: list[Part] = field(default_factory=list)
    history_text: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EventEmitter(Protocol):
    """Callback interface for streaming events from agent to caller.

    All content flows through ``emit_part``; status changes use
    ``emit_status``.  This unified interface avoids adding a new method
    for every event kind.

    Examples::

        # Stream a text delta
        await emitter.emit_part(Part(text="The capital "))
        await emitter.emit_part(Part(text="is Paris."))

        # Emit a tool call
        await emitter.emit_part(Part(
            data={"name": "search", "arguments": '{"q": "weather"}'},
            media_type="application/x-tool-call",
        ))

        # Emit a tool result
        await emitter.emit_part(Part(
            data={"name": "search", "result": '{"temp": 22}'},
            media_type="application/x-tool-result",
        ))

        # Emit a status change
        await emitter.emit_status("working", "Searching the web...")
    """

    async def emit_part(self, part: Part) -> None: ...
    async def emit_status(self, status: str, message: str = "") -> None: ...


@runtime_checkable
class AgentRunnable(Protocol):
    """Minimal protocol for pluggable agent implementations.

    Agent classes are loaded at worker startup.  The framework handles
    session lifecycle and event streaming; the agent handles its own
    tool execution, memory, and LLM interaction.

    Examples::

        class MyAgent:
            async def run(
                self,
                request: AgentRequest,
                *,
                emitter: EventEmitter,
            ) -> AgentResponse:
                user_text = request.message.parts[0].text or ""
                await emitter.emit_part(Part(text=f"You said: {user_text}"))
                return AgentResponse(
                    output=[Part(text=f"You said: {user_text}")],
                )
    """

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse: ...
