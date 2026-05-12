"""Unit tests for vllmd.runner (no Docker required)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from vllmd.runner import (
    MANAGED_LABEL,
    MODEL_LABEL,
    RunConfig,
    _container_exists,
    _parse_host_port,
    _parse_labels,
    _wait_ready,
    build_docker_run_cmd,
)


def test_run_config_model_id(tmp_path: Path) -> None:
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    cfg = RunConfig(model_path=model_dir, port=8000)
    assert cfg.model_id == "my-model"


def test_run_config_container_name_default(tmp_path: Path) -> None:
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    cfg = RunConfig(model_path=model_dir)
    assert cfg.container_name == "vllmd-my-model"


def test_run_config_container_name_explicit() -> None:
    cfg = RunConfig(model_path=Path("/models/foo"), name="custom-name")
    assert cfg.container_name == "custom-name"


def test_run_config_endpoint() -> None:
    cfg = RunConfig(model_path=Path("/models/foo"), port=9001)
    assert cfg.endpoint == "http://localhost:9001"


def test_build_docker_run_cmd_includes_label(tmp_path: Path) -> None:
    model_dir = tmp_path / "llama3"
    model_dir.mkdir()
    cfg = RunConfig(model_path=model_dir, port=8000)
    cmd = build_docker_run_cmd(cfg)
    assert f"--label={MANAGED_LABEL}=true" in cmd
    assert any(MODEL_LABEL in arg for arg in cmd)


def test_build_docker_run_cmd_no_gpu(tmp_path: Path) -> None:
    model_dir = tmp_path / "llama3"
    model_dir.mkdir()
    cfg = RunConfig(model_path=model_dir, port=8000, gpu=False)
    cmd = build_docker_run_cmd(cfg)
    assert "--gpus" not in cmd


def test_parse_host_port() -> None:
    assert _parse_host_port("0.0.0.0:8001->8000/tcp") == 8001
    assert _parse_host_port("0.0.0.0:9999->8000/tcp, :::9999->8000/tcp") == 9999
    assert _parse_host_port("") is None


def test_parse_labels() -> None:
    labels = _parse_labels("com.vllmd.managed=true,com.vllmd.model=llama3")
    assert labels["com.vllmd.managed"] == "true"
    assert labels["com.vllmd.model"] == "llama3"


def test_container_exists_false() -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="other-container\n")
        assert not _container_exists("vllmd-llama3")


def test_container_exists_true() -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="vllmd-llama3\n")
        assert _container_exists("vllmd-llama3")


def test_wait_ready_success() -> None:
    with patch("urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        assert _wait_ready("http://localhost:8000", timeout=5)


def test_wait_ready_timeout() -> None:
    with (
        patch("urllib.request.urlopen", side_effect=OSError("refused")),
        patch("time.sleep"),
    ):
        assert not _wait_ready("http://localhost:8000", timeout=1)
