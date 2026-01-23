"""Persistent storage helpers for SSLogic stage outputs."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .utils import StructuredOutputError, coerce_json_dict


ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_stage_name(stage: str) -> str:
    """Convert an arbitrary stage label into a filesystem-safe name."""

    sanitized = _SAFE_NAME_PATTERN.sub("_", stage.strip().lower() or "stage")
    return sanitized.strip("_") or "stage"


def _ensure_artifact_root() -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_ROOT


@dataclass
class StageArtifact:
    """Reference to a stored stage output."""

    stage: str
    path: Path
    session_dir: Path
    content_hash: str
    log_path: Optional[Path] = None

    def read_text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def try_load_json(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Attempt to parse the artifact as JSON; returns (payload, error_message)."""

        text = self.read_text()
        if not text.strip():
            return None, "empty-output"
        try:
            payload = coerce_json_dict(text)
            return payload, None
        except StructuredOutputError as exc:  # type: ignore[misc]
            return None, str(exc)


def create_session_dir(prefix: str = "session") -> Path:
    """Create and return a new session directory under the artifacts root."""

    base_dir = _ensure_artifact_root()
    session_id = uuid.uuid4().hex
    session_dir = base_dir / prefix / session_id[:2] / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def persist_artifact(
    stage: str,
    output: str,
    *,
    log: Optional[str] = None,
    suffix: str = ".json",
    session_dir: Optional[Path] = None,
) -> StageArtifact:
    """Persist *output* and return a :class:`StageArtifact`.

    When *session_dir* is provided, all stage files for the session will be stored
    beneath that directory. Otherwise the content hash layout is used.
    """

    base_dir = session_dir if session_dir else _ensure_artifact_root()
    stage_name = _safe_stage_name(stage)
    data = output or ""
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()

    if session_dir:
        stage_dir = base_dir
        stage_dir.mkdir(parents=True, exist_ok=True)
    else:
        stage_dir = base_dir / digest[:2] / digest
        stage_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{stage_name}{suffix}" if suffix else stage_name
    artifact_path = stage_dir / filename
    artifact_path.write_text(data, encoding="utf-8")

    log_path = None
    if log:
        log_filename = f"{stage_name}.log"
        log_path = stage_dir / log_filename
        log_path.write_text(log, encoding="utf-8")

    return StageArtifact(
        stage=stage,
        path=artifact_path,
        session_dir=stage_dir,
        content_hash=digest,
        log_path=log_path,
    )
