"""S3 sync for the ChromaDB vector store directory."""

from __future__ import annotations

import subprocess
from pathlib import Path


def sync_push(db_path: Path, s3_uri: str) -> None:
    """Upload the local DB directory to S3."""
    _s3_sync(str(db_path), s3_uri.rstrip("/") + "/")


def sync_pull(s3_uri: str, db_path: Path) -> None:
    """Download the S3 DB to the local directory."""
    db_path.mkdir(parents=True, exist_ok=True)
    _s3_sync(s3_uri.rstrip("/") + "/", str(db_path))


def _s3_sync(src: str, dst: str) -> None:
    subprocess.run(
        ["aws", "s3", "sync", src, dst, "--delete"],
        check=True,
    )
