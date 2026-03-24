from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI

from areal.utils import logging

logger = logging.getLogger("AgentDataProxy")


@dataclass
class _SessionData:
    history: list[dict[str, Any]] = field(default_factory=list)
    last_active: float = field(default_factory=time.monotonic)


def create_data_proxy_app(
    worker_addr: str,
    session_timeout: int = 3600,
) -> FastAPI:
    sessions: dict[str, _SessionData] = {}
    http_client = httpx.AsyncClient(timeout=600.0)

    async def _reap_idle_sessions() -> None:
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            stale = [
                k for k, s in sessions.items() if now - s.last_active > session_timeout
            ]
            for k in stale:
                del sessions[k]
            if stale:
                logger.info("Reaped %d idle sessions", len(stale))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http_client = http_client
        app.state.reaper_task = asyncio.create_task(_reap_idle_sessions())
        try:
            yield
        finally:
            app.state.reaper_task.cancel()
            try:
                await app.state.reaper_task
            except asyncio.CancelledError:
                pass
            await http_client.aclose()

    app = FastAPI(title="AReaL Data Proxy", lifespan=lifespan)
    app.state.http_client = http_client

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "active_sessions": len(sessions),
            "worker_addr": worker_addr,
        }

    @app.post("/session/{session_key}/turn")
    async def turn(session_key: str, body: dict[str, Any]):
        session = sessions.get(session_key)
        if session is None:
            session = _SessionData()
            sessions[session_key] = session

        message = body.get("message", "")
        run_id = body.get("run_id", "")
        queue_mode = body.get("queue_mode", "collect")

        worker_request = {
            "message": message,
            "session_key": session_key,
            "run_id": run_id,
            "history": session.history.copy(),
            "queue_mode": queue_mode,
            "config": body.get("config", body.get("details", body.get("metadata", {}))),
        }

        resp = await app.state.http_client.post(
            f"{worker_addr}/run", json=worker_request
        )
        resp.raise_for_status()
        result = resp.json()

        session.history.append({"role": "user", "content": message})

        call_counter = 0
        for evt in result.get("events", []):
            if evt.get("type") == "tool_call":
                call_id = (
                    evt.get("call_id")
                    or f"call_{evt.get('name', '')}_{run_id}_{call_counter}"
                )
                call_counter += 1
                session.history.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": evt.get("name", ""),
                                    "arguments": evt.get("args", ""),
                                },
                            }
                        ],
                    }
                )
            elif evt.get("type") == "tool_result":
                result_call_id = evt.get("call_id") or (
                    f"call_{evt.get('name', '')}_{run_id}_{call_counter - 1}"
                    if call_counter > 0
                    else f"call_{evt.get('name', '')}_{run_id}_0"
                )
                session.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": result_call_id,
                        "content": evt.get("result", ""),
                    }
                )

        history_text = result.get("history_text")
        if history_text is None:
            output_parts = result.get("output", [])
            if output_parts:
                first_part = output_parts[0] if isinstance(output_parts, list) else {}
                if isinstance(first_part, dict):
                    history_text = first_part.get("text", "")
                else:
                    history_text = ""
        if history_text:
            session.history.append({"role": "assistant", "content": history_text})

        session.last_active = time.monotonic()
        return result

    @app.post("/session/{session_key}/close")
    async def close_session(session_key: str):
        sessions.pop(session_key, None)
        return {"status": "ok"}

    @app.get("/session/{session_key}/history")
    async def get_history(session_key: str):
        session = sessions.get(session_key)
        if session is None:
            return {"history": []}
        return {"history": session.history}

    return app
