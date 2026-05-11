# AgentRunner

> **⚠️ Experimental** — This project is under active development. APIs, config format, and CLI flags may change without notice.

Run a model that is already on the machine in a [vLLM](https://github.com/vllm-project/vllm) container and serve it on a specified port with an OpenAI-compatible API.

## Install

```bash
uv pip install -e ".[dev]"
```

Requires Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.

## Quick Start

```bash
# Serve a model on port 8000 (waits until the API is ready)
agent-runner run --model /path/to/my-model --port 8000

# Run in the background and return immediately
agent-runner run --model /path/to/my-model --port 8000 --detach --no-wait

# CPU-only (no GPU)
agent-runner run --model /path/to/my-model --port 8000 --no-gpu

# Check if the container is up and the API is healthy
agent-runner status

# Stream container logs
agent-runner logs --follow

# Stop the container
agent-runner stop
```

## How It Works

1. `agent-runner run` resolves the model path and pulls `vllm/vllm-openai:latest` if needed
2. A Docker container is started with the model directory mounted read-only at `/model`
3. vLLM serves the model on port 8000 inside the container, mapped to `--port` on the host
4. The served model ID is the directory name of the model path
5. The endpoint exposes a standard OpenAI-compatible API at `http://localhost:<port>/v1`

## Commands

| Command | Description |
|---------|-------------|
| `run` | Start a vLLM container for a model |
| `stop` | Stop and remove the container |
| `status` | Show container state and API health |
| `logs` | Print container logs |

## Options

### `run`

| Flag | Default | Description |
|------|---------|-------------|
| `--model`, `-m` | (required) | Path to the model directory |
| `--port`, `-p` | `8000` | Host port for the vLLM API |
| `--name`, `-n` | `agentrunner` | Docker container name |
| `--gpu/--no-gpu` | `--gpu` | Enable/disable GPU passthrough |
| `--dtype` | `auto` | Model dtype (`auto`, `float16`, `bfloat16`, `float32`) |
| `--max-model-len` | — | Override max context length |
| `--detach`, `-d` | `false` | Start in background |
| `--wait/--no-wait` | `--wait` | Wait for API to be ready (implies background start) |

Extra positional arguments are forwarded verbatim to vLLM.

## Using the API

Once running, the endpoint is OpenAI-compatible:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Works out of the box with [AgentTester](https://github.com/sroomberg/agenttester):

```yaml
# agent-tester.yaml
agents:
  my-model:
    command: 'agent-tester query http://localhost:8000 my-model {prompt}'
    host: localhost
    commit_style: manual
    timeout: 120
```

## Development

```bash
uv pip install -e ".[dev]"
ruff check src/ tests/
ruff format src/ tests/
pytest
```

## Docker

```bash
MODEL_PATH=/path/to/my-model docker compose run --rm agent-runner run --model /model --port 8000
```
