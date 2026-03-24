from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from areal.utils import logging
from areal.utils.dynamic_import import import_from_string

from ..types import AgentRequest, AgentResponse, AgentRunnable, Part

logger = logging.getLogger("AgentWorker")

TOOL_CALL_MEDIA = "application/x-tool-call"
TOOL_RESULT_MEDIA = "application/x-tool-result"


class _CollectingEmitter:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def emit_part(self, part: Part) -> None:
        if part.media_type == TOOL_CALL_MEDIA and part.data:
            self.events.append(
                {
                    "type": "tool_call",
                    "name": part.data.get("name", ""),
                    "args": part.data.get("arguments", ""),
                    "call_id": part.data.get("call_id", ""),
                }
            )
        elif part.media_type == TOOL_RESULT_MEDIA and part.data:
            self.events.append(
                {
                    "type": "tool_result",
                    "name": part.data.get("name", ""),
                    "result": part.data.get("result", ""),
                    "call_id": part.data.get("call_id", ""),
                }
            )
        elif part.text is not None:
            self.events.append({"type": "delta", "text": part.text})

    async def emit_status(self, status: str, message: str = "") -> None:
        self.events.append({"type": "status", "status": status, "message": message})


def create_worker_app(
    agent_cls_path: str,
    **agent_kwargs: Any,
) -> FastAPI:
    app = FastAPI(title="AReaL Agent Worker")

    cls = import_from_string(agent_cls_path)
    agent: AgentRunnable = cls(**agent_kwargs)
    if not isinstance(agent, AgentRunnable):
        raise TypeError(
            f"Loaded class {agent_cls_path} does not satisfy AgentRunnable protocol "
            f"(missing async def run(request, *, emitter) method)"
        )
    logger.info("Agent loaded: %s", agent_cls_path)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/run")
    async def run(body: dict[str, Any]):
        request = AgentRequest.from_payload(body)

        emitter = _CollectingEmitter()

        try:
            response: AgentResponse = await agent.run(request, emitter=emitter)
        except Exception as exc:
            logger.exception("Agent run failed (session=%s)", request.session_id)
            return JSONResponse(
                {"error": {"message": str(exc), "type": type(exc).__name__}},
                status_code=500,
            )

        return {**asdict(response), "events": emitter.events}

    return app
