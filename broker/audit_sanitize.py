"""
Redact secrets from structures before audit persistence (Phase 18).
"""

from __future__ import annotations

from typing import Any

_SENSITIVE_KEY_FRAGMENTS = (
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "token",
    "authorization",
    "credential",
)


def sanitize_for_audit(obj: Any, *, max_depth: int = 12) -> Any:
    """Recursively copy dict/list structures with sensitive keys redacted."""
    if max_depth <= 0:
        return "<max_depth>"
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if any(anyfrag in kl for anyfrag in _SENSITIVE_KEY_FRAGMENTS):
                out[k] = "<redacted>"
            else:
                out[k] = sanitize_for_audit(v, max_depth=max_depth - 1)
        return out
    if isinstance(obj, list):
        return [sanitize_for_audit(x, max_depth=max_depth - 1) for x in obj[:200]]
    if isinstance(obj, tuple):
        return tuple(sanitize_for_audit(x, max_depth=max_depth - 1) for x in obj[:200])
    return obj
