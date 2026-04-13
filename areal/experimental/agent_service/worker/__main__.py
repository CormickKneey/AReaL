"""``python -m areal.experimental.agent_service.worker``

Standalone mode (default):
    Start Worker only — used by Controller to fork individual workers.

    python -m areal.experimental.agent_service.worker \
        --agent examples.agent_service.agent.Tau2Agent \
        --host 127.0.0.1 --port 9000

Legacy combined mode (--combined):
    Start Worker + DataProxy together and register with Router.
    Preserved for backward compatibility with manual startup.

    python -m areal.experimental.agent_service.worker --combined \
        --agent examples.agent_service.agent.Tau2Agent \
        --router-addr http://localhost:8081
"""

import argparse
import asyncio
import threading

import httpx
import uvicorn

from areal.utils.network import format_hostport

from ..auth import DEFAULT_ADMIN_API_KEY
from .app import create_worker_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Worker")
    parser.add_argument("--agent", required=True, help="Agent import path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--log-level", choices=["debug", "info", "warning", "error"], default="info"
    )

    combined = parser.add_argument_group("combined mode (legacy)")
    combined.add_argument(
        "--combined",
        action="store_true",
        help="Start Worker + DataProxy together (legacy mode)",
    )
    combined.add_argument("--router-addr", help="Router HTTP address (combined mode)")
    combined.add_argument("--proxy-port", type=int, default=9100)
    combined.add_argument("--admin-api-key", default=DEFAULT_ADMIN_API_KEY)
    args = parser.parse_args()

    if args.combined:
        _run_combined(args)
    else:
        app = create_worker_app(args.agent)
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


def _run_combined(args: argparse.Namespace) -> None:
    if not args.router_addr:
        raise SystemExit("--router-addr is required in --combined mode")

    from ..auth import admin_headers
    from ..data_proxy import create_data_proxy_app
    from ..data_proxy.config import DataProxyConfig

    worker_addr = f"http://{format_hostport(args.host, args.port)}"
    proxy_addr = f"http://{format_hostport(args.host, args.proxy_port)}"

    worker_app = create_worker_app(args.agent)
    proxy_config = DataProxyConfig(
        host=args.host,
        port=args.proxy_port,
        worker_addr=worker_addr,
    )
    proxy_app = create_data_proxy_app(proxy_config)

    def run_worker():
        uvicorn.run(
            worker_app, host=args.host, port=args.port, log_level=args.log_level
        )

    threading.Thread(target=run_worker, daemon=True).start()

    async def register():
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{args.router_addr}/register",
                json={"addr": proxy_addr},
                headers=admin_headers(args.admin_api_key),
            )

    asyncio.run(register())
    uvicorn.run(
        proxy_app, host=args.host, port=args.proxy_port, log_level=args.log_level
    )


if __name__ == "__main__":
    main()
