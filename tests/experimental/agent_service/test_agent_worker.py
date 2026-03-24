from __future__ import annotations

from unittest.mock import patch

import pytest

from areal.experimental.agent_service.types import (
    AgentRequest,
    AgentResponse,
    AgentRunnable,
    EventEmitter,
    Part,
)
from areal.experimental.agent_service.worker.app import (
    _CollectingEmitter,
    create_worker_app,
)

httpx = pytest.importorskip("httpx")

TOOL_CALL_MEDIA = "application/x-tool-call"
TOOL_RESULT_MEDIA = "application/x-tool-result"


class _EchoAgent:
    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        text = request.message.parts[0].text or "" if request.message.parts else ""
        await emitter.emit_part(Part(text=f"echo: {text}"))
        return AgentResponse(
            output=[Part(text=f"echo: {text}")],
            history_text=f"echo: {text}",
            metadata={"history_len": len(request.history)},
        )


class _ToolAgent:
    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        await emitter.emit_part(
            Part(
                data={"name": "search", "arguments": '{"q": "test"}'},
                media_type=TOOL_CALL_MEDIA,
            )
        )
        await emitter.emit_part(
            Part(
                data={"name": "search", "result": "found it"},
                media_type=TOOL_RESULT_MEDIA,
            )
        )
        await emitter.emit_part(Part(text="Done"))
        return AgentResponse(output=[Part(text="Done")], history_text="Done")


class _FailAgent:
    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        raise RuntimeError("boom")


class _ConfigAgent:
    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        return AgentResponse(
            output=[Part(text="ok")],
            history_text="ok",
            metadata={
                "instructions": request.config.get("instructions", ""),
                "model": request.config.get("model", ""),
                "tool_count": len(request.config.get("tools", [])),
                "idempotency_key": request.config.get("idempotency_key", ""),
                "input_count": len(request.config.get("input", [])),
                "legacy": request.config.get("legacy"),
            },
        )


def _make_client(agent_cls):
    with patch(
        "areal.experimental.agent_service.worker.app.import_from_string",
        return_value=agent_cls,
    ):
        app = create_worker_app("mock.path")
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://worker")


class TestWorkerHealth:
    @pytest.mark.asyncio
    async def test_health(self):
        async with _make_client(_EchoAgent) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"


class TestWorkerRun:
    @pytest.mark.asyncio
    async def test_echo(self):
        async with _make_client(_EchoAgent) as client:
            resp = await client.post(
                "/run",
                json={"message": "hello", "session_key": "s1", "run_id": "r1"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["output"][0]["text"] == "echo: hello"
            assert any(e["type"] == "delta" for e in data["events"])

    @pytest.mark.asyncio
    async def test_history_forwarded(self):
        async with _make_client(_EchoAgent) as client:
            resp = await client.post(
                "/run",
                json={
                    "message": "hi",
                    "session_key": "s1",
                    "run_id": "r1",
                    "history": [{"role": "user", "content": "prev"}],
                },
            )
            assert resp.json()["metadata"]["history_len"] == 1

    @pytest.mark.asyncio
    async def test_config_fields_from_new_payload(self):
        async with _make_client(_ConfigAgent) as client:
            resp = await client.post(
                "/run",
                json={
                    "message": "hi",
                    "session_key": "s1",
                    "run_id": "r1",
                    "config": {
                        "input": [{"type": "message", "content": "hi"}],
                        "instructions": "be helpful",
                        "model": "remote-agent",
                        "tools": [{"type": "function", "name": "lookup"}],
                        "idempotency_key": "idem-1",
                    },
                },
            )
            assert resp.status_code == 200
            assert resp.json()["metadata"] == {
                "instructions": "be helpful",
                "model": "remote-agent",
                "tool_count": 1,
                "idempotency_key": "idem-1",
                "input_count": 1,
                "legacy": None,
            }

    @pytest.mark.asyncio
    async def test_legacy_metadata_payload_maps_to_config(self):
        async with _make_client(_ConfigAgent) as client:
            resp = await client.post(
                "/run",
                json={
                    "message": "hi",
                    "session_key": "s1",
                    "run_id": "r1",
                    "metadata": {
                        "input": [{"type": "message", "content": "hi"}],
                        "instructions": "legacy instructions",
                        "model": "legacy-model",
                        "tools": [{"type": "function", "name": "search"}],
                        "idempotencyKey": "legacy-idem",
                        "legacy": "still-here",
                    },
                },
            )
            assert resp.status_code == 200
            assert resp.json()["metadata"] == {
                "instructions": "legacy instructions",
                "model": "legacy-model",
                "tool_count": 1,
                "idempotency_key": "legacy-idem",
                "input_count": 1,
                "legacy": "still-here",
            }

    @pytest.mark.asyncio
    async def test_tool_events(self):
        async with _make_client(_ToolAgent) as client:
            resp = await client.post(
                "/run",
                json={"message": "go", "session_key": "s1", "run_id": "r1"},
            )
            types = [e["type"] for e in resp.json()["events"]]
            assert "tool_call" in types
            assert "tool_result" in types
            assert "delta" in types

    @pytest.mark.asyncio
    async def test_agent_failure(self):
        async with _make_client(_FailAgent) as client:
            resp = await client.post(
                "/run",
                json={"message": "x", "session_key": "s1", "run_id": "r1"},
            )
            assert resp.status_code == 500


class TestCollectingEmitter:
    @pytest.mark.asyncio
    async def test_collects_all_event_types(self):
        e = _CollectingEmitter()
        await e.emit_part(Part(text="hi"))
        await e.emit_part(
            Part(data={"name": "fn", "arguments": "{}"}, media_type=TOOL_CALL_MEDIA)
        )
        await e.emit_part(
            Part(data={"name": "fn", "result": "ok"}, media_type=TOOL_RESULT_MEDIA)
        )
        assert len(e.events) == 3


class TestAgentRunnableProtocol:
    def test_echo_satisfies(self):
        assert isinstance(_EchoAgent(), AgentRunnable)

    def test_plain_object_does_not(self):
        assert not isinstance(object(), AgentRunnable)
