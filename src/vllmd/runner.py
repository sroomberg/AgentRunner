"""Docker management for vLLM model containers."""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

VLLM_IMAGE = "vllm/vllm-openai:latest"
HEALTH_TIMEOUT = 300
HEALTH_INTERVAL = 3

MANAGED_LABEL = "com.vllmd.managed"
MODEL_LABEL = "com.vllmd.model"
MODEL_PATH_LABEL = "com.vllmd.model_path"


@dataclass
class RunConfig:
    model_path: Path
    port: int = 8000
    name: str | None = None  # None → derived from model dir name
    gpu: bool = True
    dtype: str = "auto"
    max_model_len: int | None = None
    extra_args: list[str] = field(default_factory=list)

    @property
    def model_id(self) -> str:
        return self.model_path.resolve().name

    @property
    def container_name(self) -> str:
        return self.name or f"vllmd-{self.model_id}"

    @property
    def endpoint(self) -> str:
        return f"http://localhost:{self.port}"


def _docker(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
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


def _parse_host_port(ports_str: str) -> int | None:
    """Extract the host port from a Ports string like '0.0.0.0:8001->8000/tcp'."""
    m = re.search(r":(\d+)->8000", ports_str)
    return int(m.group(1)) if m else None


def _parse_labels(labels_str: str) -> dict[str, str]:
    """Parse Docker's comma-separated 'key=value,key=value' label string."""
    result: dict[str, str] = {}
    for part in labels_str.split(","):
        part = part.strip()
        if "=" in part:
            k, _, v = part.partition("=")
            result[k.strip()] = v.strip()
    return result


def build_docker_run_cmd(config: RunConfig) -> list[str]:
    model_path = config.model_path.resolve()
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        config.container_name,
        "-p",
        f"{config.port}:8000",
        "-v",
        f"{model_path}:/model:ro",
        f"--label={MANAGED_LABEL}=true",
        f"--label={MODEL_LABEL}={config.model_id}",
        f"--label={MODEL_PATH_LABEL}={model_path}",
    ]
    if config.gpu:
        cmd += ["--gpus", "all"]
    cmd += [
        VLLM_IMAGE,
        "--model",
        "/model",
        "--served-model-name",
        config.model_id,
        "--dtype",
        config.dtype,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    if config.max_model_len is not None:
        cmd += ["--max-model-len", str(config.max_model_len)]
    cmd += config.extra_args
    return cmd


def start(config: RunConfig) -> None:
    """Start a vLLM container (foreground, blocking)."""
    model_path = config.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if _container_exists(config.container_name):
        raise RuntimeError(
            f"Container '{config.container_name}' already exists. "
            "Stop it first or use a different --name."
        )
    subprocess.run(build_docker_run_cmd(config), check=True)


def stop(name: str) -> None:
    """Stop and remove a named container."""
    if not _container_exists(name):
        raise RuntimeError(f"No container named '{name}' found.")
    _docker("stop", name)


def stop_all() -> list[str]:
    """Stop all vllmd-managed containers. Returns list of stopped names."""
    containers = list_containers()
    stopped = []
    for c in containers:
        _docker("stop", c["name"])
        stopped.append(c["name"])
    return stopped


def list_containers() -> list[dict]:
    """Return info for all running vllmd-managed containers."""
    result = subprocess.run(
        [
            "docker",
            "ps",
            "--filter",
            f"label={MANAGED_LABEL}=true",
            "--format",
            "{{json .}}",
        ],
        capture_output=True,
        text=True,
    )
    containers = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        data = json.loads(line)
        labels = _parse_labels(data.get("Labels", ""))
        host_port = _parse_host_port(data.get("Ports", ""))
        model_id = labels.get(MODEL_LABEL, "?")
        model_path = labels.get(MODEL_PATH_LABEL, "?")
        containers.append(
            {
                "name": data["Names"],
                "model_id": model_id,
                "model_path": model_path,
                "port": host_port,
                "endpoint": f"http://localhost:{host_port}" if host_port else "?",
                "status": data.get("Status", "?"),
            }
        )
    return containers


def status(name: str) -> dict:
    """Return container state and API health for a named container."""
    result = subprocess.run(
        ["docker", "inspect", name, "--format", "{{json .State}}"],
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
            with contextlib.suppress(ValueError):
                port = int(binding)
        with contextlib.suppress(Exception):
            url = f"http://localhost:{port}/v1/models"
            with urllib.request.urlopen(url, timeout=3):
                api_healthy = True

    return {"running": running, "api_healthy": api_healthy, "container": state}


def wait_ready(config: RunConfig) -> bool:
    """Block until the vLLM API is reachable, or timeout."""
    return _wait_ready(config.endpoint)


def logs(name: str, follow: bool = False) -> None:
    """Stream or print container logs."""
    cmd = ["logs"]
    if follow:
        cmd.append("-f")
    cmd.append(name)
    _docker(*cmd)
