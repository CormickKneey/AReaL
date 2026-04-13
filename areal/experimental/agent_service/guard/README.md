# Agent Service Guard

Pure pass-through process supervisor backed by AReaL's shared
[guard infrastructure](../../../infra/rpc/guard/).

## Role

The guard is a **dumb process manager**: it allocates ports, forks child processes,
tracks PIDs, and cleans up on shutdown. It exposes only the base guard HTTP API:

| Route                      | Purpose                            |
| -------------------------- | ---------------------------------- |
| `GET /health`              | Health check + forked child count  |
| `POST /alloc_ports`        | Allocate free ports                |
| `POST /fork`               | Fork a child process               |
| `POST /kill_forked_worker` | Kill a specific child              |
| `POST /configure`          | Runtime configuration (hook-based) |

All orchestration logic (which services to launch, in what order, how to register them)
lives in the [AgentServiceController](../controller/controller.py).

## Usage

```bash
python -m areal.experimental.agent_service.guard \
    --experiment-name demo --trial-name run0 \
    --role agent-guard --worker-index 0
```

The guard is typically started by the scheduler (Local/Ray/Slurm) as part of
`AgentServiceController.initialize()`. You rarely need to start it manually.
