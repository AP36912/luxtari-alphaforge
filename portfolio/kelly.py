"""
Safe Kelly sizing: fractional Kelly with caps and guards for noisy inputs.

Full Kelly ``f*`` for binary outcomes (win/loss sizes fixed): ``(p b - q) / b``
with ``b = avg_win / avg_loss``, ``p`` win rate, ``q = 1-p``. The **safe** variant
uses a fraction of ``f*`` (e.g. 1/4 Kelly) and an absolute **cap** to avoid
over-leverage from estimation error.
"""

from __future__ import annotations


def safe_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    *,
    kelly_fraction: float = 0.25,
    cap: float = 0.25,
    floor: float = 0.0,
) -> float:
    """
    Return a **portfolio fraction** in ``[floor, cap]`` suitable for scaling signals.

    Args:
        win_rate: Historical or estimated probability of a winning trade, in ``(0,1)``.
        avg_win: Average win magnitude (positive).
        avg_loss: Average loss magnitude (positive number; not negative PnL).
        kelly_fraction: Multiply raw Kelly by this (e.g. ``0.25`` = quarter Kelly).
        cap: Hard ceiling on the returned fraction.
        floor: Minimum returned value (often ``0``).

    Returns:
        Non-negative fraction, capped. Returns ``floor`` if inputs are invalid or
        raw Kelly is non-positive.
    """
    if cap < 0 or kelly_fraction < 0 or kelly_fraction > 1:
        raise ValueError("cap and kelly_fraction must be valid (kelly_fraction in [0,1])")
    if avg_loss <= 0 or avg_win <= 0:
        return float(floor)
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError("win_rate must be in [0, 1]")

    p = float(win_rate)
    q = 1.0 - p
    b = avg_win / avg_loss
    if b <= 0:
        return float(floor)

    raw_kelly = (p * b - q) / b
    if raw_kelly <= 0:
        return float(floor)

    scaled = raw_kelly * kelly_fraction
    out = min(scaled, cap)
    return float(max(floor, out))
