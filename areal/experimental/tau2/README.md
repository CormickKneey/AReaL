# TAU2 Agentic RL with AREAL

## Agentic Framework

docs: https://yuque.antfin.com/xrl-team/asystem-user-guide/ovrftp3f0kx2sh97

## Directory Structure

```bash
areal/experimental/tau2/
├── example_user.json   # chat history example of user for a task
├── example_agent.json  # chat history example of agent for a task
├── convert_dataset.py  # convert TAU2 data to huggingface dataset
├── tau2_env.py         # tau2 environment
└── agent.py            # agent, demonstrates how to interact with the TAU2 environment
```

## Quickstart

### Prepare Data

```bash
## 1. Download tau-bench data
export TAU2_BENCH_DIR=/path/to/tau2-bench
git clone https://github.com/sierra-research/tau2-bench.git $TAU2_BENCH_DIR
export TAU2_DATA_DIR=$TAU2_BENCH_DIR/data
export TAU2_OUTPUT_DIR=/path/to/output/dataset

## 2. Convert data
python areal/experimental/tau2/convert_dataset.py --data_dir $TAU2_DATA_DIR --output_dir $TAU2_OUTPUT_DIR --split train

## 3. install required packages
pip install git+https://github.com/sierra-research/tau2-bench.git@0ed2fd8d830a20657d89ae9c2efcc94838aa7129
pip install openai==2.7.1 openai-agents==0.5.0 transformers==4.56.1

## 4. login to wandb
wandb login --relogin --host=https://slurm.alipay.com

## 5. start locally!
export TAU2_DATA_DIR=$TAU2_BENCH_DIR/data

python3 -m areal.launcher.local \
examples/experimental/proxy/agent_grpo_proxy.py   \
--config  examples/experimental/proxy/agent.yaml  \
trial_name=20251111-02  \
train_dataset.batch_size=4  \
allocation_mode=sglang.d1p1t1+d1p1t1 \
stats_logger.wandb.mode=online \
train_dataset.path=/tau-data/   \
agent_module_path="areal.experimental.tau2.agent"  \
actor.path=Qwen/Qwen3-0.6B    \
experiment_name=tau2-agentic-rl \
actor.mb_spec.max_tokens_per_mb=32768 \
agent_custom_env.TAU2_USER_LLM_API_BASE=http://10.10.128.181:30000/v1 \
agent_custom_env.TAU2_USER_LLM_API_KEY="empty" \
agent_custom_env.TAU2_USER_LLM="openai/Qwen3-30B-A3B-Thinking"
```

## See Example Agent and User

| File               | Description                      |
| ------------------ | -------------------------------- |
| example_agent.json | Example agent (DeepSeek-R1-0528) |
| example_user.json  | Example user (Kimi-K2-Instruct)  |
