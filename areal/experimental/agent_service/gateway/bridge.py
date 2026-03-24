from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from areal.utils import logging

from ..auth import DEFAULT_ADMIN_KEY, admin_headers, make_admin_dependency
from ..content import extract_message_content
from ..protocol import generate_run_id

logger = logging.getLogger("AgentBridge")


@dataclass(frozen=True)
class _OpenResponsesTurn:
    input_items: list[dict[str, Any]]
    instructions: str
    model: str
    user: str
    tools: list[dict[str, Any]]
    user_metadata: dict[str, Any]

    @classmethod
    def from_body(cls, body: dict[str, Any]) -> "_OpenResponsesTurn":
        return cls(
            input_items=list(body.get("input", [])),
            instructions=str(body.get("instructions", "")),
            model=str(body.get("model", "")),
            user=str(body.get("user", "")),
            tools=list(body.get("tools", [])),
            user_metadata=dict(body.get("metadata", {})),
        )

    @property
    def session_key(self) -> str:
        return OpenResponsesBridge._derive_session_key(self.user, self.model)

    @property
    def message(self):
        return OpenResponsesBridge._extract_message(self.input_items, self.instructions)

    def build_worker_payload(self, run_id: str, response_id: str) -> dict[str, Any]:
        return {
            "message": self.message,
            "run_id": run_id,
            "queue_mode": "collect",
            "config": {
                "input_items": self.input_items,
                "instructions": self.instructions,
                "model": self.model,
                "tools": self.tools,
                "idempotency_key": response_id,
                **self.user_metadata,
            },
        }

    def build_openresponses_payload(
        self,
        response_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        output_items: list[dict[str, Any]] = []

        output_parts = result.get("output", [])
        for part in output_parts:
            if not isinstance(part, dict):
                continue
            if part.get("text") is not None:
                output_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": part["text"]}],
                    }
                )

        for evt in result.get("events", []):
            if evt.get("type") == "tool_call":
                output_items.append(
                    {
                        "type": "function_call",
                        "name": evt.get("name", ""),
                        "arguments": evt.get("args", ""),
                    }
                )

        response_metadata = dict(result.get("metadata", {}) or {})
        status = result.get("status", "completed")
        error = result.get("error")
        if error:
            response_metadata["error"] = error

        return {
            "id": response_id,
            "object": "response",
            "status": status,
            "output": output_items,
            "model": self.model,
            "metadata": response_metadata,
        }


class AgentBridge(ABC):
    @abstractmethod
    async def handle_request(self, request: Request) -> Any: ...


class OpenResponsesBridge(AgentBridge):
    def __init__(self, router_addr: str, admin_key: str = DEFAULT_ADMIN_KEY) -> None:
        self._router_addr = router_addr
        self._auth_headers = admin_headers(admin_key)

    async def handle_request(self, request: Request) -> Any:
        turn = _OpenResponsesTurn.from_body(await request.json())

        if not turn.user:
            return JSONResponse(
                {
                    "error": {
                        "message": "'user' field is required for session affinity",
                        "type": "invalid_request",
                    }
                },
                status_code=400,
            )

        run_id = generate_run_id()
        response_id = f"resp-{uuid.uuid4().hex[:12]}"

        try:
            route_resp = await request.app.state.http_client.post(
                f"{self._router_addr}/route",
                json={"session_key": turn.session_key},
                headers=self._auth_headers,
            )
            route_resp.raise_for_status()
            data_proxy_addr = route_resp.json()["data_proxy_addr"]

            turn_resp = await request.app.state.http_client.post(
                f"{data_proxy_addr}/session/{turn.session_key}/turn",
                json=turn.build_worker_payload(run_id, response_id),
            )
            turn_resp.raise_for_status()
            result = turn_resp.json()
            return JSONResponse(turn.build_openresponses_payload(response_id, result))
        except Exception as exc:
            logger.error("OpenResponses request failed: %s", exc)
            return JSONResponse(
                {"error": {"message": str(exc), "type": "server_error"}},
                status_code=500,
            )

    @staticmethod
    def _extract_message(
        input_items: list[dict[str, Any]], instructions: str
    ) -> str | list[dict[str, Any]]:
        return extract_message_content(input_items, instructions)

    @staticmethod
    def _derive_session_key(user: str, model: str) -> str:
        if user:
            return f"agent:{model or 'default'}:{user}"
        return f"agent:{model or 'default'}:{uuid.uuid4().hex[:8]}"


def mount_bridge(
    app: FastAPI,
    bridge: OpenResponsesBridge,
    admin_key: str = DEFAULT_ADMIN_KEY,
) -> None:
    auth = make_admin_dependency(admin_key)

    @app.post("/v1/responses", dependencies=[Depends(auth)])
    async def responses_endpoint(request: Request):
        return await bridge.handle_request(request)
