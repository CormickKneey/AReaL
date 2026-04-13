"""Unit tests for AgentServiceController.

All Guard HTTP interactions are mocked — no real processes or servers.
Tests cover: initialize, destroy, scale_up, scale_down, and error handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from areal.experimental.agent_service.controller.config import (
    AgentServiceControllerConfig,
)
from areal.experimental.agent_service.controller.controller import (
    AgentServiceController,
)

CTRL = "areal.experimental.agent_service.controller.controller"


def _mock_alloc_ports_response(host: str, ports: list[int]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "success", "host": host, "ports": ports}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_fork_response(host: str, pid: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "success", "host": host, "pid": pid}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_kill_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "success"}
    resp.text = '{"status": "success"}'
    return resp


def _mock_register_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


def _mock_health_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok"}
    return resp


@pytest.fixture()
def config():
    return AgentServiceControllerConfig(
        agent_cls_path="my.Agent",
        admin_key="test-key",
        num_pairs=2,
        setup_timeout=1.0,
    )


class TestInit:
    def test_no_guards_raises(self, config):
        with pytest.raises(ValueError, match="(?i)at least one"):
            AgentServiceController(config=config, guard_addrs=[])

    def test_construction(self, config):
        ctrl = AgentServiceController(config=config, guard_addrs=["http://g0:8090"])
        assert ctrl.router_addr == ""
        assert ctrl.gateway_addr == ""
        assert ctrl.pairs == {}


class TestInitialize:
    @patch(f"{CTRL}.requests")
    def test_initialize_forks_router_pairs_gateway(self, mock_requests, config):
        """Initialize should fork: 1 Router + 2 Workers + 2 Proxies + 1 Gateway = 6 forks."""
        port_counter = iter(range(9001, 9100))

        def mock_post(url, **kwargs):
            if "/alloc_ports" in url:
                port = next(port_counter)
                return _mock_alloc_ports_response("10.0.0.1", [port])
            if "/fork" in url:
                return _mock_fork_response("10.0.0.1", 100)
            if "/register" in url:
                return _mock_register_response()
            if "/kill_forked_worker" in url:
                return _mock_kill_response()
            return MagicMock(status_code=404)

        def mock_get(url, **kwargs):
            return _mock_health_response()

        mock_requests.post = mock_post
        mock_requests.get = mock_get
        mock_requests.RequestException = Exception

        ctrl = AgentServiceController(config=config, guard_addrs=["http://g0:8090"])
        ctrl.initialize()

        # Router + Gateway forked
        assert "http://" in ctrl.router_addr
        assert "http://" in ctrl.gateway_addr

        # 2 pairs created
        assert len(ctrl.pairs) == 2
        assert 0 in ctrl.pairs
        assert 1 in ctrl.pairs

        # Total forked: router(1) + worker×2 + proxy×2 + gateway(1) = 6
        assert len(ctrl._forked_services) == 6


class TestScaleUp:
    @patch(f"{CTRL}.requests")
    def test_scale_up_adds_pairs(self, mock_requests, config):
        config.num_pairs = 0  # start with none

        port_counter = iter(range(9001, 9100))

        def mock_post(url, **kwargs):
            if "/alloc_ports" in url:
                port = next(port_counter)
                return _mock_alloc_ports_response("10.0.0.1", [port])
            if "/fork" in url:
                return _mock_fork_response("10.0.0.1", 200)
            if "/register" in url:
                return _mock_register_response()
            if "/kill_forked_worker" in url:
                return _mock_kill_response()
            return MagicMock(status_code=404)

        mock_requests.post = mock_post
        mock_requests.get = lambda url, **kw: _mock_health_response()
        mock_requests.RequestException = Exception

        ctrl = AgentServiceController(config=config, guard_addrs=["http://g0:8090"])
        ctrl.initialize()  # router + gateway, 0 pairs

        assert len(ctrl.pairs) == 0

        created = ctrl.scale_up(3)
        assert created == [0, 1, 2]
        assert len(ctrl.pairs) == 3

    @patch(f"{CTRL}.requests")
    def test_scale_up_round_robins_guards(self, mock_requests, config):
        config.num_pairs = 0

        guards_called: list[str] = []

        def mock_post(url, **kwargs):
            if "/alloc_ports" in url:
                guards_called.append(url.split("/alloc_ports")[0])
                return _mock_alloc_ports_response("10.0.0.1", [9001])
            if "/fork" in url:
                return _mock_fork_response("10.0.0.1", 100)
            if "/register" in url:
                return _mock_register_response()
            if "/kill_forked_worker" in url:
                return _mock_kill_response()
            return MagicMock(status_code=404)

        mock_requests.post = mock_post
        mock_requests.get = lambda url, **kw: _mock_health_response()
        mock_requests.RequestException = Exception

        ctrl = AgentServiceController(
            config=config,
            guard_addrs=["http://g0:8090", "http://g1:8091"],
        )
        ctrl.initialize()
        guards_called.clear()

        ctrl.scale_up(4)

        # Worker+Proxy for pair 0 on g0, pair 1 on g1, pair 2 on g0, pair 3 on g1
        # Each pair does 2 alloc_ports calls (worker + proxy)
        g0_calls = [g for g in guards_called if "g0" in g]
        g1_calls = [g for g in guards_called if "g1" in g]
        assert len(g0_calls) == 4  # pairs 0,2 × 2 allocs each
        assert len(g1_calls) == 4  # pairs 1,3 × 2 allocs each


class TestScaleDown:
    @patch(f"{CTRL}.requests")
    def test_scale_down_removes_newest_first(self, mock_requests, config):
        config.num_pairs = 3

        port_counter = iter(range(9001, 9100))

        def mock_post(url, **kwargs):
            if "/alloc_ports" in url:
                port = next(port_counter)
                return _mock_alloc_ports_response("10.0.0.1", [port])
            if "/fork" in url:
                return _mock_fork_response("10.0.0.1", 100)
            if "/register" in url:
                return _mock_register_response()
            if "/kill_forked_worker" in url:
                return _mock_kill_response()
            if "/unregister" in url:
                return _mock_register_response()
            return MagicMock(status_code=404)

        mock_requests.post = mock_post
        mock_requests.get = lambda url, **kw: _mock_health_response()
        mock_requests.RequestException = Exception

        ctrl = AgentServiceController(config=config, guard_addrs=["http://g0:8090"])
        ctrl.initialize()
        assert len(ctrl.pairs) == 3

        removed = ctrl.scale_down(2)
        assert set(removed) == {2, 1}  # newest first
        assert len(ctrl.pairs) == 1
        assert 0 in ctrl.pairs  # oldest survives


class TestDestroy:
    @patch(f"{CTRL}.requests")
    def test_destroy_clears_everything(self, mock_requests, config):
        config.num_pairs = 1

        port_counter = iter(range(9001, 9100))

        def mock_post(url, **kwargs):
            if "/alloc_ports" in url:
                port = next(port_counter)
                return _mock_alloc_ports_response("10.0.0.1", [port])
            if "/fork" in url:
                return _mock_fork_response("10.0.0.1", 100)
            if "/register" in url:
                return _mock_register_response()
            if "/kill_forked_worker" in url:
                return _mock_kill_response()
            return MagicMock(status_code=404)

        mock_requests.post = mock_post
        mock_requests.get = lambda url, **kw: _mock_health_response()
        mock_requests.RequestException = Exception

        ctrl = AgentServiceController(config=config, guard_addrs=["http://g0:8090"])
        ctrl.initialize()
        assert len(ctrl._forked_services) > 0

        ctrl.destroy()
        assert ctrl.router_addr == ""
        assert ctrl.gateway_addr == ""
        assert ctrl.pairs == {}
        assert ctrl._forked_services == []

    @patch(f"{CTRL}.requests")
    def test_destroy_tolerates_kill_errors(self, mock_requests, config):
        config.num_pairs = 0

        port_counter = iter(range(9001, 9100))
        kill_count = 0

        def mock_post(url, **kwargs):
            nonlocal kill_count
            if "/alloc_ports" in url:
                port = next(port_counter)
                return _mock_alloc_ports_response("10.0.0.1", [port])
            if "/fork" in url:
                return _mock_fork_response("10.0.0.1", 100)
            if "/kill_forked_worker" in url:
                kill_count += 1
                raise ConnectionError("Guard down")
            return MagicMock(status_code=404)

        mock_requests.post = mock_post
        mock_requests.get = lambda url, **kw: _mock_health_response()
        mock_requests.RequestException = Exception

        ctrl = AgentServiceController(config=config, guard_addrs=["http://g0:8090"])
        ctrl.initialize()  # router + gateway = 2 forked

        # Should not raise even though kills fail
        ctrl.destroy()
        assert kill_count == 2
        assert ctrl._forked_services == []
