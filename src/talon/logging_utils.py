"""Structured JSON logging with automatic secret redaction."""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

SENSITIVE_KEY_PARTS = frozenset(
    {
        "key",
        "apikey",
        "secret",
        "passphrase",
        "password",
        "token",
        "auth",
        "authorization",
        "signature",
        "bearer",
        "private",
    }
)

SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 [REDACTED]"),
    (re.compile(r"\b[0-9a-fA-F]{32,}\b"), "[REDACTED_HEX]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"), "[REDACTED_KEY]"),
)


def redact_text(text: str) -> str:
    """Scrub secret patterns from a string."""
    out = text
    for pattern, repl in SECRET_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def redact_value(val: Any) -> Any:
    """Recursively redact sensitive keys and string patterns."""
    if isinstance(val, str):
        return redact_text(val)
    if isinstance(val, dict):
        cleaned: dict[str, Any] = {}
        for k, v in val.items():
            key_str = str(k)
            parts = set(re.split(r"[\s_\-.]+", key_str.lower()))
            if parts & SENSITIVE_KEY_PARTS or key_str.lower() in SENSITIVE_KEY_PARTS:
                cleaned[key_str] = "[REDACTED]"
            else:
                cleaned[key_str] = redact_value(v)
        return cleaned
    if isinstance(val, (list, tuple)):
        return [redact_value(x) for x in val]
    return val


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON with secret scrubbing."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact_text(record.getMessage()),
        }
        extra_data = getattr(record, "structured_data", None)
        if isinstance(extra_data, dict):
            payload["data"] = redact_value(extra_data)
        return json.dumps(payload, sort_keys=True)


def get_logger(name: str = "talon", level: str = "INFO") -> logging.Logger:
    """Create or retrieve a configured structured logger without global mutation side effects."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
