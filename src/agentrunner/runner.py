"""Docker management for vLLM model containers."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

VLLM_IMAGE = "vllm/vllm-openai:latest"
HEALTH_TIMEOUT = 300
HEALTH_INTERVAL = 3


@dataclass
class RunConfig:
    model_path: Path
    port: int = 8000
    name: str = "agentrunner"
    gpu: bool = True
    dtype: str = "auto"
    max_model_len: int | None = None
    extra_args: list[str] = field(default_factory=list)

    @property
    def model_id(self) -> str:
        return self.model_path.resolve().name

    @property
    def endpoint(self) -> str:
        return f"http://localhost:{self.port}"


def _docker(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["docker", *args]
    return subprocess.run(
        cmd,
        check=True,
        capture_output=capture,
        text=True,
    )


def _container_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    return name in result.stdout.splitlines()


def _wait_ready(endpoint: str, timeout: int = HEALTH_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{endpoint}/v1/models", timeout=5):
                return True
        except Exception:
            time.sleep(HEALTH_INTERVAL)
    return False


def start(config: RunConfig) -> None:
    """Start a vLLM container serving *config.model_path* on *config.port*."""
    model_path = config.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    if _container_exists(config.name):
        raise RuntimeError(
            f"Container '{config.name}' already exists. "
            "Run `agent-runner stop` first or use a different --name."
        )

    cmd = [
        "run",
        "--rm",
        "--name", config.name,
        "-p", f"{config.port}:8000",
        "-v", f"{model_path}:/model:ro",
    ]

    if config.gpu:
        cmd += ["--gpus", "all"]

    cmd += [
        VLLM_IMAGE,
        "--model", "/model",
        "--served-model-name", config.model_id,
        "--dtype", config.dtype,
        "--host", "0.0.0.0",
        "--port", "8000",
    ]

    if config.max_model_len is not None:
        cmd += ["--max-model-len", str(config.max_model_len)]

    cmd += config.extra_args

    _docker(*cmd)


def stop(name: str = "agentrunner") -> None:
    """Stop and remove the named container."""
    if not _container_exists(name):
        raise RuntimeError(f"No container named '{name}' found.")
    _docker("stop", name)


def status(name: str = "agentrunner") -> dict:
    """Return a dict with container state and API health."""
    result = subprocess.run(
        [
            "docker", "inspect", name,
            "--format", "{{json .State}}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"running": False, "api_healthy": False, "container": None}

    state = json.loads(result.stdout.strip())
    running = state.get("Running", False)
    api_healthy = False

    if running:
        port_result = subprocess.run(
            ["docker", "port", name, "8000"],
            capture_output=True,
            text=True,
        )
        port = 8000
        if port_result.returncode == 0:
            binding = port_result.stdout.strip().split(":")[-1]
            try:
                port = int(binding)
            except ValueError:
                pass
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=3):
                api_healthy = True
        except Exception:
            pass

    return {"running": running, "api_healthy": api_healthy, "container": state}


def wait_ready(config: RunConfig) -> bool:
    """Block until the vLLM API is reachable, or timeout. Returns True on success."""
    return _wait_ready(config.endpoint)


def logs(name: str = "agentrunner", follow: bool = False) -> None:
    """Stream or print container logs."""
    cmd = ["logs"]
    if follow:
        cmd.append("-f")
    cmd.append(name)
    _docker(*cmd)
