"""Unit tests for agentrunner.runner (no Docker required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentrunner.runner import RunConfig, _container_exists, _wait_ready


def test_run_config_model_id(tmp_path: Path) -> None:
    model_dir = tmp_path / "my-model"
    model_dir.mkdir()
    cfg = RunConfig(model_path=model_dir, port=8000)
    assert cfg.model_id == "my-model"


def test_run_config_endpoint() -> None:
    cfg = RunConfig(model_path=Path("/models/foo"), port=9001)
    assert cfg.endpoint == "http://localhost:9001"


def test_container_exists_false() -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="other-container\n")
        assert not _container_exists("agentrunner")


def test_container_exists_true() -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="agentrunner\n")
        assert _container_exists("agentrunner")


def test_wait_ready_success() -> None:
    with patch("urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        assert _wait_ready("http://localhost:8000", timeout=5)


def test_wait_ready_timeout() -> None:
    with patch("urllib.request.urlopen", side_effect=OSError("refused")):
        with patch("time.sleep"):
            assert not _wait_ready("http://localhost:8000", timeout=1)
