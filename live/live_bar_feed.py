"""Generic recent-bars feed hook (no vendor bridge)."""
from __future__ import annotations

from typing import Any, Protocol


class RecentBarsFeed(Protocol):
    def get_recent_bars(
        self, *, symbol: str, timeframe: str, count: int, include_incomplete: bool = False
    ) -> list[dict[str, Any]]:
        ...


class NullRecentBarsFeed:
    """Fail-closed feed used when no market-data adapter is configured."""

    def get_recent_bars(
        self, *, symbol: str, timeframe: str, count: int, include_incomplete: bool = False
    ) -> list[dict[str, Any]]:
        return []
