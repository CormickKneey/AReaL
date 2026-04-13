"""AgentServiceController — orchestrates agent service micro-services via Guards.

Mirrors the architecture of
:class:`~areal.experimental.inference_service.controller.controller.GatewayInferenceController`:
the controller talks to RPCGuard workers over HTTP to fork/kill child
services, rather than managing processes directly.

Lifecycle::

    controller = AgentServiceController(config, guard_addrs=["http://g0:8090"])
    controller.initialize()
    # ... run traffic ...
    controller.scale_up(2)     # add 2 Worker+DataProxy pairs
    controller.scale_down(1)   # drain + remove 1 pair
    controller.destroy()
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

import requests

from areal.experimental.agent_service.controller.config import (
    AgentServiceControllerConfig,
)
from areal.utils import logging
from areal.utils.network import format_hostport

logger = logging.getLogger("AgentServiceController")


# ---------------------------------------------------------------------------
# Pair tracking
# ---------------------------------------------------------------------------


@dataclass
class _WorkerPair:
    """Internal bookkeeping for a single Worker+DataProxy pair."""

    pair_index: int
    guard_addr: str
    worker_host: str
    worker_port: int
    proxy_host: str
    proxy_port: int
    proxy_addr: str  # http://host:port
    worker_addr: str  # http://host:port


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class AgentServiceController:
    """Orchestrator for the Agent Service micro-service stack.

    Responsibilities:

    * Fork Router and Gateway on the first Guard.
    * Fork ``num_pairs`` Worker+DataProxy pairs across Guards.
    * Register DataProxy addresses with the Router.
    * Provide ``scale_up`` / ``scale_down`` for dynamic pair management.
    * Clean up all forked services on ``destroy()``.

    Parameters
    ----------
    config:
        Controller configuration.
    guard_addrs:
        HTTP addresses of RPCGuard workers, e.g. ``["http://10.0.0.1:8090"]``.
        Services are distributed across guards round-robin.
    """

    def __init__(
        self,
        config: AgentServiceControllerConfig,
        guard_addrs: list[str],
    ) -> None:
        if not guard_addrs:
            raise ValueError("At least one guard address is required.")

        self.config = config
        self.guard_addrs = guard_addrs

        # Service addresses (populated in initialize)
        self._router_addr: str = ""
        self._gateway_addr: str = ""

        # Pair management
        self._pairs: dict[int, _WorkerPair] = {}
        self._next_pair_index: int = 0

        # Track (guard_addr, role, worker_index) for cleanup
        self._forked_services: list[tuple[str, str, int]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Launch the full micro-service stack.

        Order: Router → Worker+DataProxy pairs → register → Gateway.
        """
        cfg = self.config
        guard_0 = self.guard_addrs[0]

        # Step 1: Fork Router on guard[0]
        router_cmd = [
            sys.executable,
            "-m",
            "areal.experimental.agent_service.router",
            "--admin-api-key",
            cfg.admin_key,
        ]
        router_host, router_port = self._fork_on_guard(
            guard_addr=guard_0,
            role="agent-router",
            worker_index=0,
            raw_cmd=router_cmd,
        )
        self._router_addr = f"http://{format_hostport(router_host, router_port)}"
        logger.info("Router: %s", self._router_addr)

        # Step 2: Fork Worker+DataProxy pairs
        self.scale_up(cfg.num_pairs)

        # Step 3: Fork Gateway on guard[0]
        gw_cmd = [
            sys.executable,
            "-m",
            "areal.experimental.agent_service.gateway",
            "--router-addr",
            self._router_addr,
            "--admin-api-key",
            cfg.admin_key,
        ]
        gw_host, gw_port = self._fork_on_guard(
            guard_addr=guard_0,
            role="agent-gateway",
            worker_index=0,
            raw_cmd=gw_cmd,
        )
        self._gateway_addr = f"http://{format_hostport(gw_host, gw_port)}"
        logger.info("Gateway: %s", self._gateway_addr)

    def destroy(self) -> None:
        """Tear down all services in reverse order."""
        for guard_addr, role, worker_index in reversed(self._forked_services):
            try:
                self._kill_forked_service(guard_addr, role, worker_index)
            except Exception:
                logger.error(
                    "Error killing forked service %s/%d: %s",
                    role,
                    worker_index,
                    traceback.format_exc(),
                )
        self._forked_services.clear()
        self._pairs.clear()
        self._router_addr = ""
        self._gateway_addr = ""

    def scale_up(self, count: int) -> list[int]:
        """Add *count* Worker+DataProxy pairs.

        Pairs are distributed across guards round-robin.
        Returns the pair indices that were created.
        """
        cfg = self.config
        created: list[int] = []

        for _ in range(count):
            pair_index = self._next_pair_index
            self._next_pair_index += 1

            guard_addr = self.guard_addrs[pair_index % len(self.guard_addrs)]

            # Fork Worker
            worker_cmd = [
                sys.executable,
                "-m",
                "areal.experimental.agent_service.worker",
                "--agent",
                cfg.agent_cls_path,
                "--log-level",
                cfg.log_level,
            ]
            worker_host, worker_port = self._fork_on_guard(
                guard_addr=guard_addr,
                role=f"agent-worker-{pair_index}",
                worker_index=pair_index,
                raw_cmd=worker_cmd,
            )
            worker_addr = f"http://{format_hostport(worker_host, worker_port)}"

            # Fork DataProxy
            proxy_cmd = [
                sys.executable,
                "-m",
                "areal.experimental.agent_service.data_proxy",
                "--worker-addr",
                worker_addr,
            ]
            proxy_host, proxy_port = self._fork_on_guard(
                guard_addr=guard_addr,
                role=f"agent-proxy-{pair_index}",
                worker_index=pair_index,
                raw_cmd=proxy_cmd,
            )
            proxy_addr = f"http://{format_hostport(proxy_host, proxy_port)}"

            pair = _WorkerPair(
                pair_index=pair_index,
                guard_addr=guard_addr,
                worker_host=worker_host,
                worker_port=worker_port,
                proxy_host=proxy_host,
                proxy_port=proxy_port,
                proxy_addr=proxy_addr,
                worker_addr=worker_addr,
            )
            self._pairs[pair_index] = pair
            created.append(pair_index)

            # Register DataProxy with Router
            self._register_proxy(proxy_addr)
            logger.info(
                "Pair %d: worker=%s proxy=%s", pair_index, worker_addr, proxy_addr
            )

        return created

    def scale_down(self, count: int) -> list[int]:
        """Remove *count* pairs (LIFO order).

        For each pair: unregister from Router → kill DataProxy → kill Worker.
        Returns the pair indices that were removed.
        """
        removed: list[int] = []
        # Remove newest pairs first
        indices = sorted(self._pairs.keys(), reverse=True)

        for pair_index in indices[:count]:
            pair = self._pairs.pop(pair_index)

            # Unregister from Router
            self._unregister_proxy(pair.proxy_addr)

            # Kill proxy then worker (proxy first so no traffic to dead worker)
            proxy_key = (pair.guard_addr, f"agent-proxy-{pair_index}", pair_index)
            worker_key = (pair.guard_addr, f"agent-worker-{pair_index}", pair_index)

            for guard_addr, role, wi in [proxy_key, worker_key]:
                try:
                    self._kill_forked_service(guard_addr, role, wi)
                    # Also remove from _forked_services tracking
                    entry = (guard_addr, role, wi)
                    if entry in self._forked_services:
                        self._forked_services.remove(entry)
                except Exception:
                    logger.warning(
                        "Failed to kill %s/%d: %s",
                        role,
                        wi,
                        traceback.format_exc(),
                    )

            removed.append(pair_index)
            logger.info("Removed pair %d", pair_index)

        return removed

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def router_addr(self) -> str:
        return self._router_addr

    @property
    def gateway_addr(self) -> str:
        return self._gateway_addr

    @property
    def pairs(self) -> dict[int, _WorkerPair]:
        return dict(self._pairs)

    # ------------------------------------------------------------------
    # Guard interaction helpers
    # ------------------------------------------------------------------

    def _fork_on_guard(
        self,
        guard_addr: str,
        role: str,
        worker_index: int,
        raw_cmd: list[str],
        health_path: str = "/health",
        env: dict[str, str] | None = None,
    ) -> tuple[str, int]:
        """Fork a process on a Guard via ``/alloc_ports`` + ``/fork``.

        Returns ``(host, port)`` and records the entry for cleanup.
        Mirrors ``GatewayInferenceController._fork_on_guard``.
        """
        # Allocate port
        resp = requests.post(
            f"{guard_addr}/alloc_ports",
            json={"count": 1},
            timeout=30,
        )
        resp.raise_for_status()
        port_data = resp.json()
        host = port_data["host"]
        port = port_data["ports"][0]

        cmd = list(raw_cmd) + ["--host", host, "--port", str(port)]

        merged_env = {**self.config.env, **(env or {})}

        fork_payload: dict[str, Any] = {
            "role": role,
            "worker_index": worker_index,
            "raw_cmd": cmd,
        }
        if merged_env:
            fork_payload["env"] = merged_env

        resp = requests.post(
            f"{guard_addr}/fork",
            json=fork_payload,
            timeout=30,
        )
        resp.raise_for_status()

        self._forked_services.append((guard_addr, role, worker_index))

        addr = f"http://{format_hostport(host, port)}"
        self._wait_for_service(f"{addr}{health_path}", role)

        return host, port

    def _kill_forked_service(
        self, guard_addr: str, role: str, worker_index: int
    ) -> None:
        """Kill a forked service via the Guard's ``/kill_forked_worker``."""
        try:
            resp = requests.post(
                f"{guard_addr}/kill_forked_worker",
                json={"role": role, "worker_index": worker_index},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Killed forked service %s/%d", role, worker_index)
            else:
                logger.warning(
                    "Failed to kill forked service %s/%d: %s",
                    role,
                    worker_index,
                    resp.text,
                )
        except requests.RequestException as exc:
            logger.error(
                "Error killing forked service %s/%d: %s", role, worker_index, exc
            )

    def _wait_for_service(
        self, url: str, name: str, timeout: float | None = None
    ) -> None:
        """Poll a health endpoint until it returns 200."""
        timeout = timeout or self.config.setup_timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    logger.info("%s healthy at %s", name, url)
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"{name} did not become healthy at {url} within {timeout}s")

    def _register_proxy(self, proxy_addr: str) -> None:
        """Register a DataProxy with the Router."""
        if not self._router_addr:
            return
        try:
            resp = requests.post(
                f"{self._router_addr}/register",
                json={"addr": proxy_addr},
                headers={"Authorization": f"Bearer {self.config.admin_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("Registered proxy %s with Router", proxy_addr)
        except Exception as exc:
            logger.warning("Failed to register proxy %s: %s", proxy_addr, exc)

    def _unregister_proxy(self, proxy_addr: str) -> None:
        """Unregister a DataProxy from the Router."""
        if not self._router_addr:
            return
        try:
            requests.post(
                f"{self._router_addr}/unregister",
                json={"addr": proxy_addr},
                headers={"Authorization": f"Bearer {self.config.admin_key}"},
                timeout=5,
            )
        except Exception as exc:
            logger.warning("Failed to unregister proxy %s: %s", proxy_addr, exc)
