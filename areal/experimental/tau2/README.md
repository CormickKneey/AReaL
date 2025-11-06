# TAU2 Agentic RL with AREAL

## Agentic Framework

docs: https://yuque.antfin.com/xrl-team/asystem-user-guide/ovrftp3f0kx2sh97

## Directory Structure

````
examples/tau2/
├── example_user.json   # chat history example of user for a task
├── example_agent.json  # chat history example of agent for a task
├── convert_dataset.py  # convert TAU2 data to huggingface dataset
├── tau2_env.py         # tau2 environment
└── agent.py            # agent, demonstrates how to interact with the TAU2 environment

## Quickstart

### Prepare Data

```bash
## Download tau-bench data
export TAU2_BENCH_DIR=/path/to/tau2-bench
git clone -b feat/gym-only https://github.com/sierra-research/tau2-bench.git $TAU2_BENCH_DIR
export TAU2_DATA_DIR=$TAU2_BENCH_DIR/data
export TAU2_OUTPUT_DIR=/path/to/output/dataset

## Convert data
python areal/experimental/tau2/convert_dataset.py --data_dir $TAU2_DATA_DIR --output_dir $TAU2_OUTPUT_DIR --split train

## install required packages
pip install git+https://github.com/sierra-research/tau2-bench.git@15e68c887de4d586c4f718af3fc03581cb957dae
pip install openai==2.7.1 openai-agents==0.5.0 transformers==4.56.1

## login to wandb
wandb login --relogin --host=https://slurm.alipay.com

## start locally!
python3 -m areal.launcher.local examples/experimental/proxy/agent_grpo_proxy.py   --config  examples/math/gsm8k_grpo.yaml    trial_name=20251111-01  train_dataset.batch_size=4  allocation_mode=sglang.d1p1t1+d1p1t1 stats_logger.wandb.mode=online +train_dataset.dataset_path=$TAU2_OUTPUT_DIR +agent_module_path="areal.experimental.tau2.agent"
````

### Check Example Usage

```bash
python examples/tau2/example_usage.py
```

### See Example Agent and User

| File               | Description                      |
| ------------------ | -------------------------------- |
| example_agent.json | Example agent (DeepSeek-R1-0528) |
| example_user.json  | Example user (Kimi-K2-Instruct)  |
