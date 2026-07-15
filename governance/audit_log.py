"""
governance/audit_log.py
-----------------------
Append-only structured audit logger.

Every pipeline run writes one JSON line to `audit_log.jsonl` in the project
root (or a path supplied via the AUDIT_LOG_PATH environment variable).

Usage
-----
    from governance.audit_log import append_audit_record

    append_audit_record(lead_id="abc-123", record={"stage": "enrich", ...})

The file is opened in append mode on every write, so it survives crashes
without data loss and is safe for concurrent single-process writes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default log path — can be overridden by the AUDIT_LOG_PATH env var.
_DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "audit_log.jsonl"


def _log_path() -> Path:
    """Return the resolved path to the audit log file."""
    env_path = os.environ.get("AUDIT_LOG_PATH")
    return Path(env_path) if env_path else _DEFAULT_LOG_PATH


def append_audit_record(
    lead_id: str,
    record: dict[str, Any],
    *,
    log_path: Path | None = None,
) -> None:
    """
    Append a single structured JSON line to the audit log.

    Parameters
    ----------
    lead_id:
        The stable identifier for the lead this record describes.
        Injected at the top level so log parsers can filter by lead without
        deserialising the full payload.
    record:
        Arbitrary dict of pipeline data. Will be merged with injected fields
        (`lead_id`, `logged_at`). Nested Pydantic models should be passed as
        `model.model_dump()`.
    log_path:
        Override the destination file (useful in tests). Falls back to the
        path determined by :func:`_log_path`.
    """
    destination = log_path or _log_path()

    # Ensure the parent directory exists (e.g. if a custom path is used).
    destination.parent.mkdir(parents=True, exist_ok=True)

    entry: dict[str, Any] = {
        "lead_id": lead_id,
        "logged_at": datetime.now(tz=timezone.utc).isoformat(),
        **record,
    }

    with destination.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=_json_serialiser) + "\n")


def read_audit_records(
    lead_id: str | None = None,
    *,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Read all audit records, optionally filtered by lead_id.

    Parameters
    ----------
    lead_id:
        If supplied, only records matching this lead are returned.
    log_path:
        Override the source file (useful in tests).

    Returns
    -------
    list[dict]
        Parsed JSON objects, in the order they were written.
    """
    source = log_path or _log_path()
    if not source.exists():
        return []

    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Corrupt line — skip but don't crash the reader.
                continue
            if lead_id is None or obj.get("lead_id") == lead_id:
                records.append(obj)

    return records


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _json_serialiser(obj: Any) -> str:
    """
    Fallback serialiser for types not handled by the standard JSON encoder.
    Currently handles datetime objects; extend as needed.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
