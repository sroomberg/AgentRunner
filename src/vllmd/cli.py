"""CLI entry point."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .runner import (
    RunConfig,
    build_docker_run_cmd,
    list_containers,
    logs,
    status,
    stop,
    stop_all,
    stop_by_port,
    wait_ready,
)
from .sessions.cli import session_app
from .vectordb.cli import db_app

app = typer.Typer(
    name="vllmd",
    help="Run local models via vLLM in Docker containers.",
    no_args_is_help=True,
)
app.add_typer(db_app, name="db")
app.add_typer(session_app, name="session")
console = Console()

_NAME_HELP = "Container name (default: vllmd-<model-dir-name>)"


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
        str | None,
        typer.Option("--name", "-n", help=_NAME_HELP),
    ] = None,
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
            "--detach",
            "-d",
            help="Start container in background; wait for API ready, then return.",
        ),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait/--no-wait",
            help="With --detach: wait for the API to be ready before returning.",
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

    model_path = model.resolve()
    console.print(f"[bold]Starting vLLM container[/bold] '{config.container_name}'")
    console.print(f"  Model:    {model_path}")
    console.print(f"  Model ID: {config.model_id}")
    console.print(f"  Port:     {port}")
    console.print(f"  GPU:      {'yes' if gpu else 'no'}")
    console.print()

    docker_cmd = build_docker_run_cmd(config)

    if detach:
        subprocess.Popen(
            docker_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
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
            console.print(f"Container started. Endpoint: {config.endpoint}")
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
        str | None,
        typer.Option("--name", "-n", help="Container name to stop"),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", "-p", help="Port of the container to stop"),
    ] = None,
    all_containers: Annotated[
        bool,
        typer.Option("--all", "-a", help="Stop all vllmd-managed containers"),
    ] = False,
) -> None:
    """Stop one or all vllmd containers."""
    if all_containers:
        stopped = stop_all()
        if stopped:
            for n in stopped:
                console.print(f"[green]Stopped '{n}'.[/green]")
        else:
            console.print("[yellow]No running vllmd containers found.[/yellow]")
        return

    if name is None and port is None:
        console.print("[red]Specify --name, --port, or --all.[/red]")
        raise typer.Exit(1)

    try:
        if port is not None:
            stopped_name = stop_by_port(port)
            console.print(f"[green]Stopped '{stopped_name}'.[/green]")
        else:
            stop(name)  # type: ignore[arg-type]
            console.print(f"[green]Stopped '{name}'.[/green]")
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Docker exited with code {e.returncode}[/red]")
        raise typer.Exit(e.returncode) from e


@app.command(name="ps")
def ps_cmd() -> None:
    """List all running vllmd-managed containers."""
    containers = list_containers()
    if not containers:
        console.print("[dim]No vllmd containers running.[/dim]")
        return

    table = Table("Name", "Model", "Port", "Endpoint", "Status", show_header=True)
    for c in containers:
        table.add_row(
            c["name"],
            c["model_id"],
            str(c["port"] or "?"),
            c["endpoint"],
            c["status"],
        )
    console.print(table)


@app.command(name="status")
def status_cmd(
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Container name (omit to show all)"),
    ] = None,
) -> None:
    """Show container and API status. Omit --name to show all containers."""
    if name is None:
        # Show summary table for all managed containers
        containers = list_containers()
        if not containers:
            console.print("[dim]No vllmd containers running.[/dim]")
            raise typer.Exit(1)

        table = Table("Name", "Model", "Port", "API", "Status", show_header=True)
        all_healthy = True
        for c in containers:
            info = status(c["name"])
            api_str = (
                "[green]healthy[/green]"
                if info["api_healthy"]
                else "[yellow]unreachable[/yellow]"
            )
            if not info["api_healthy"]:
                all_healthy = False
            table.add_row(
                c["name"],
                c["model_id"],
                str(c["port"] or "?"),
                api_str,
                c["status"],
            )
        console.print(table)
        if not all_healthy:
            raise typer.Exit(1)
        return

    info = status(name)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    running_str = "[green]running[/green]" if info["running"] else "[red]stopped[/red]"
    api_str = (
        "[green]healthy[/green]"
        if info["api_healthy"]
        else "[yellow]unreachable[/yellow]"
    )

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
        str | None,
        typer.Option(
            "--name", "-n", help="Container name (auto-resolved if only one is running)"
        ),
    ] = None,
    follow: Annotated[
        bool,
        typer.Option("--follow", "-f", help="Follow log output"),
    ] = False,
) -> None:
    """Print logs from a vllmd container."""
    if name is None:
        running = list_containers()
        if len(running) == 1:
            name = running[0]["name"]
        elif len(running) == 0:
            console.print("[yellow]No running vllmd containers found.[/yellow]")
            raise typer.Exit(1)
        else:
            console.print("[red]Multiple containers running — specify --name:[/red]")
            for c in running:
                console.print(f"  {c['name']}")
            raise typer.Exit(1)

    try:
        logs(name, follow=follow)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Docker exited with code {e.returncode}[/red]")
        raise typer.Exit(e.returncode) from e
