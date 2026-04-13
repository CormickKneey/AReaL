"""Controller-based demo: launch Agent Service via AgentServiceController.

This is the production-style launch pattern that mirrors how
``GatewayInferenceController`` manages the inference service stack.
A Guard process is started first, then the Controller forks Router,
Worker+DataProxy pairs, and Gateway onto the Guard via HTTP API.

Usage::

    # 1. Start a Guard (separate terminal or background)
    python -m areal.experimental.agent_service.guard \
        --experiment-name demo --trial-name run0 \
        --role agent-guard --worker-index 0 --port 8090

    # 2. Run this script
    python examples/agent_service/run_controller_demo.py \
        --guard-addr http://localhost:8090

    # Or with options
    python examples/agent_service/run_controller_demo.py \
        --guard-addr http://localhost:8090 \
        --num-pairs 2 --domain telecom --full

For a simpler one-click demo (without Guard/Controller), use run_demo.py.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

from areal.experimental.agent_service.controller import (
    AgentServiceController,
    AgentServiceControllerConfig,
)

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


async def _wait_healthy(url: str, timeout: float = 30.0) -> None:
    async with httpx.AsyncClient() as client:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"Service at {url} did not become healthy")


async def run_task(gateway_addr: str, task, domain: str, admin_key: str) -> float:
    """Run a single tau2 task through the gateway. Returns the reward."""
    from tau2.data_model.message import AssistantMessage, UserMessage
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    session_key = f"tau2-{domain}-{task.id}"
    print(f"\n  Task: {task.id}")
    print(f"  Scenario: {str(task.user_scenario)[:120]}...")

    scripted_messages = [
        str(task.user_scenario),
        "Yes, please go ahead and help me with that.",
        "Can you check the status of my request?",
        "Thank you, that's all I need.",
    ]

    tau2_messages = []
    error_occurred = False

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, msg in enumerate(scripted_messages, 1):
            resp = await client.post(
                f"{gateway_addr}/v1/responses",
                json={
                    "input": [{"type": "message", "content": msg}],
                    "model": "tau2-agent",
                    "user": session_key,
                },
                headers={"Authorization": f"Bearer {admin_key}"},
            )
            data = resp.json()

            tau2_messages.append(
                UserMessage(role="user", content=msg, turn_idx=len(tau2_messages))
            )

            if data.get("status") == "completed":
                agent_text = ""
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for block in item.get("content", []):
                            if block.get("type") == "output_text":
                                agent_text += block["text"]
                                print(f"    [Turn {i}] Agent: {block['text'][:150]}")
                    elif item.get("type") == "function_call":
                        print(f"    [Turn {i}] [tool] {item.get('name', '')}")

                tau2_messages.append(
                    AssistantMessage(
                        role="assistant",
                        content=agent_text or "(no response)",
                        turn_idx=len(tau2_messages),
                    )
                )
            elif data.get("error"):
                err = data["error"].get("message", "")[:100]
                print(f"    [Turn {i}] Error: {err}")
                tau2_messages.append(
                    AssistantMessage(
                        role="assistant",
                        content=f"Error: {err}",
                        turn_idx=len(tau2_messages),
                    )
                )
                error_occurred = True
                break

    reward = 0.0
    if not error_occurred:
        try:
            simulation = SimulationRun(
                id=f"demo-{task.id}",
                task_id=task.id,
                messages=tau2_messages,
                start_time="",
                end_time="",
                duration=0.0,
                termination_reason=TerminationReason.USER_STOP,
            )
            reward_info = evaluate_simulation(
                simulation=simulation,
                task=task,
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,
                domain=domain,
            )
            reward = reward_info.reward
        except Exception as e:
            print(f"    Eval error: {e}")

    print(f"    Reward: {reward:.3f}")
    return reward


async def run_demo(gateway_addr: str, domain: str, full: bool, admin_key: str) -> None:
    from tau2.registry import registry

    print(f"\n{'=' * 60}")
    print(f"  Tau2 Agent Service Demo (Controller mode) — domain: {domain}")
    print(f"{'=' * 60}")

    tasks = registry.get_tasks_loader(domain)(None)
    total = len(tasks)

    if not full:
        tasks = tasks[:1]
        print(f"  Running 1 task (use --full for all {total} tasks)")
    else:
        print(f"  Running all {total} tasks")

    rewards = []
    for task in tasks:
        reward = await run_task(gateway_addr, task, domain, admin_key=admin_key)
        rewards.append((task.id, reward))

    print(f"\n{'=' * 60}")
    print(f"  Results — {len(rewards)} task(s)")
    print(f"{'=' * 60}")
    for task_id, reward in rewards:
        print(f"  Task {task_id}: reward = {reward:.3f}")
    if rewards:
        avg = sum(r for _, r in rewards) / len(rewards)
        print(f"\n  Average reward: {avg:.3f}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tau2 Agent Service Demo (Controller mode)"
    )
    parser.add_argument(
        "--guard-addr",
        required=True,
        help="Guard HTTP address (e.g. http://localhost:8090)",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Config YAML path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--domain",
        choices=["airline", "retail", "telecom"],
        help="Override tau2.domain from config",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=1,
        help="Number of Worker+DataProxy pairs (default: 1)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all tasks (default: single task)",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    tau2_cfg = config.setdefault("tau2", {})
    domain = args.domain or tau2_cfg.get("domain", "airline")
    data_dir = tau2_cfg.get("data_dir") or os.environ.get("TAU2_DATA_DIR")
    if data_dir:
        os.environ["TAU2_DATA_DIR"] = data_dir
    admin_key = config.get("admin_key", "areal-agent-admin")

    child_env: dict[str, str] = {}
    if data_dir:
        child_env["TAU2_DATA_DIR"] = data_dir
    child_env["TAU2_DOMAIN"] = domain

    agent_llm_cfg = config.get("agent_llm", {})
    if agent_llm_cfg.get("model"):
        child_env["AGENT_LLM_MODEL"] = agent_llm_cfg["model"]
    if agent_llm_cfg.get("base_url"):
        child_env["AGENT_LLM_BASE_URL"] = agent_llm_cfg["base_url"]
    if agent_llm_cfg.get("api_key"):
        child_env["AGENT_LLM_API_KEY"] = agent_llm_cfg["api_key"]

    # --- Controller ---
    ctrl_config = AgentServiceControllerConfig(
        agent_cls_path="examples.agent_service.agent.Tau2Agent",
        admin_key=admin_key,
        num_pairs=args.num_pairs,
        env=child_env,
    )
    ctrl = AgentServiceController(
        config=ctrl_config,
        guard_addrs=[args.guard_addr],
    )

    try:
        print(f"Initializing with {args.num_pairs} pair(s) on {args.guard_addr} ...")
        ctrl.initialize()
        print(f"  Router:  {ctrl.router_addr}")
        print(f"  Gateway: {ctrl.gateway_addr}")
        print(f"  Pairs:   {len(ctrl.pairs)}")

        # Wait for Gateway to be fully ready
        asyncio.run(_wait_healthy(f"{ctrl.gateway_addr}/health"))

        # Run demo
        asyncio.run(
            run_demo(
                ctrl.gateway_addr,
                domain=domain,
                full=args.full,
                admin_key=admin_key,
            )
        )
    finally:
        print("\nShutting down ...")
        ctrl.destroy()
        print("Done.")


if __name__ == "__main__":
    main()
