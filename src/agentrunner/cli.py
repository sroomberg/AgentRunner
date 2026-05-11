"""CLI entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .runner import RunConfig, logs, start, status, stop, wait_ready

app = typer.Typer(
    name="agent-runner",
    help="Run a local model via vLLM in a Docker container.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    model: Annotated[
        Path,
        typer.Option("--model", "-m", help="Path to the model directory on disk"),
    ],
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Host port to expose the vLLM API on"),
    ] = 8000,
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Docker container name"),
    ] = "agentrunner",
    gpu: Annotated[
        bool,
        typer.Option("--gpu/--no-gpu", help="Pass --gpus all to the container"),
    ] = True,
    dtype: Annotated[
        str,
        typer.Option("--dtype", help="Model dtype (auto, float16, bfloat16, float32)"),
    ] = "auto",
    max_model_len: Annotated[
        int | None,
        typer.Option("--max-model-len", help="Override max context length"),
    ] = None,
    detach: Annotated[
        bool,
        typer.Option(
            "--detach", "-d",
            help="Start container in background. Waits for the API to be ready, then returns.",
        ),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait/--no-wait",
            help="When used with --detach: wait for the API to be ready before returning (default: true).",
        ),
    ] = True,
    extra: Annotated[
        list[str] | None,
        typer.Argument(help="Extra args forwarded verbatim to vLLM"),
    ] = None,
) -> None:
    """Start a vLLM container serving MODEL on PORT."""
    config = RunConfig(
        model_path=model,
        port=port,
        name=name,
        gpu=gpu,
        dtype=dtype,
        max_model_len=max_model_len,
        extra_args=extra or [],
    )

    try:
        model_path = model.resolve()
    except Exception:
        model_path = model

    console.print(f"[bold]Starting vLLM container[/bold] '{name}'")
    console.print(f"  Model:    {model_path}")
    console.print(f"  Model ID: {config.model_id}")
    console.print(f"  Port:     {port}")
    console.print(f"  GPU:      {'yes' if gpu else 'no'}")
    console.print()

    docker_cmd = [
        "docker", "run", "--rm",
        "--name", name,
        "-p", f"{port}:8000",
        "-v", f"{model_path}:/model:ro",
        *(["--gpus", "all"] if gpu else []),
        "vllm/vllm-openai:latest",
        "--model", "/model",
        "--served-model-name", config.model_id,
        "--dtype", dtype,
        "--host", "0.0.0.0",
        "--port", "8000",
        *(["--max-model-len", str(max_model_len)] if max_model_len else []),
        *(extra or []),
    ]

    if detach:
        subprocess.Popen(
            docker_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if wait:
            console.print("[dim]Waiting for API to become ready…[/dim]")
            if wait_ready(config):
                console.print(f"[green]Ready.[/green] Endpoint: {config.endpoint}")
                console.print(f"  Model ID: [bold]{config.model_id}[/bold]")
            else:
                console.print(
                    "[yellow]Timed out waiting for API. "
                    "Container may still be loading.[/yellow]"
                )
                console.print(f"  Endpoint: {config.endpoint}")
        else:
            console.print(f"Container started in background. Endpoint: {config.endpoint}")
    else:
        try:
            subprocess.run(docker_cmd, check=True)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Docker exited with code {e.returncode}[/red]")
            raise typer.Exit(e.returncode) from e


@app.command(name="stop")
def stop_cmd(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Container name to stop"),
    ] = "agentrunner",
) -> None:
    """Stop and remove a running AgentRunner container."""
    try:
        stop(name)
        console.print(f"[green]Stopped container '{name}'.[/green]")
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Docker exited with code {e.returncode}[/red]")
        raise typer.Exit(e.returncode) from e


@app.command(name="status")
def status_cmd(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Container name to inspect"),
    ] = "agentrunner",
) -> None:
    """Show the status of a running AgentRunner container."""
    info = status(name)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    running_str = "[green]running[/green]" if info["running"] else "[red]stopped[/red]"
    api_str = "[green]healthy[/green]" if info["api_healthy"] else "[yellow]unreachable[/yellow]"

    table.add_row("Container", running_str)
    table.add_row("API", api_str)

    if info["container"]:
        table.add_row("Started", info["container"].get("StartedAt", "—"))

    console.print(table)

    if not info["running"]:
        raise typer.Exit(1)


@app.command(name="logs")
def logs_cmd(
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Container name"),
    ] = "agentrunner",
    follow: Annotated[
        bool,
        typer.Option("--follow", "-f", help="Follow log output"),
    ] = False,
) -> None:
    """Print logs from an AgentRunner container."""
    try:
        logs(name, follow=follow)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Docker exited with code {e.returncode}[/red]")
        raise typer.Exit(e.returncode) from e
