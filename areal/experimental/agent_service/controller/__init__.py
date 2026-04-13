"""Agent Service Controller — orchestrator for agent micro-services.

Usage::

    from areal.experimental.agent_service.controller import (
        AgentServiceController,
        AgentServiceControllerConfig,
    )

    controller = AgentServiceController(
        config=AgentServiceControllerConfig(
            agent_cls_path="examples.agent_service.agent.Tau2Agent",
            num_pairs=2,
        ),
        guard_addrs=["http://guard0:8090", "http://guard1:8091"],
    )
    controller.initialize()
"""

from .config import AgentServiceControllerConfig
from .controller import AgentServiceController

__all__ = [
    "AgentServiceController",
    "AgentServiceControllerConfig",
]
