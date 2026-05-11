"""Unit tests for agentrunner.sessions (no network required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentrunner.sessions.session import Message, Session


def test_session_create_and_save(tmp_path: Path) -> None:
    session = Session.create(
        "test-session",
        endpoint="http://localhost:8001",
        model_id="llama3",
        db_path=tmp_path / "vectordb",
    )
    session.save(tmp_path)
    assert (tmp_path / "test-session.json").exists()


def test_session_roundtrip(tmp_path: Path) -> None:
    session = Session.create(
        "roundtrip",
        endpoint="http://localhost:8001",
        model_id="mistral",
        db_path=tmp_path / "vectordb",
        system_prompt="You are helpful.",
    )
    session.messages.append(Message(role="user", content="hello"))
    session.messages.append(Message(role="assistant", content="hi there"))
    session.save(tmp_path)

    loaded = Session.load("roundtrip", tmp_path)
    assert loaded.id == "roundtrip"
    assert loaded.model_id == "mistral"
    assert loaded.system_prompt == "You are helpful."
    assert len(loaded.messages) == 2
    assert loaded.messages[0].role == "user"


def test_session_list(tmp_path: Path) -> None:
    for name in ("alpha", "beta", "gamma"):
        s = Session.create(name, endpoint="http://localhost:8000", model_id="m", db_path=tmp_path)
        s.save(tmp_path)
    sessions = Session.list_all(tmp_path)
    assert len(sessions) == 3
    assert {s.id for s in sessions} == {"alpha", "beta", "gamma"}


def test_session_load_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Session.load("nonexistent", tmp_path)


def test_session_clear_history(tmp_path: Path) -> None:
    session = Session.create("s", endpoint="http://localhost:8000", model_id="m", db_path=tmp_path)
    session.messages.append(Message(role="user", content="x"))
    session.clear_history()
    assert session.message_count() == 0


def test_session_delete(tmp_path: Path) -> None:
    session = Session.create("bye", endpoint="http://localhost:8000", model_id="m", db_path=tmp_path)
    session.save(tmp_path)
    assert (tmp_path / "bye.json").exists()
    session.delete(tmp_path)
    assert not (tmp_path / "bye.json").exists()


def test_chat_builds_messages(tmp_path: Path) -> None:
    """chat() should POST to the completions endpoint and persist the exchange."""
    session = Session.create(
        "chat-test",
        endpoint="http://localhost:8001",
        model_id="llama3",
        db_path=tmp_path / "vectordb",
        system_prompt="Be concise.",
    )

    fake_response = '{"choices":[{"message":{"content":"Hello!"}}]}'

    with patch("urllib.request.urlopen") as mock_open:
        mock_cm = MagicMock()
        mock_cm.__enter__ = lambda s: s
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_cm.read.return_value = fake_response.encode()
        mock_open.return_value = mock_cm

        from agentrunner.sessions.chat import chat
        reply = chat(session, "Hi model")

    assert reply == "Hello!"
    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[1].role == "assistant"
