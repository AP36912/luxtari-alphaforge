"""
Time and calendar helpers.

Trading calendars and session masks prevent accidental inclusion of after-hours
bars when the strategy assumes RTH-only liquidity.
"""

from __future__ import annotations

from typing import Any


def align_to_session_index(frame: Any, calendar: str) -> Any:
    """Filter/reindex to exchange session (Phase 2+)."""
    raise NotImplementedError("Implemented with exchange calendars in Phase 2.")
