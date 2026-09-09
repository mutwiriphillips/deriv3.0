"""
TICK features (spec Part 9). Operate on raw (epoch, price) tick tuples,
already sliced to time t by the caller — same no-look-ahead contract as
CandleSeries, just for the finer-grained tick stream.
"""
from __future__ import annotations


def consecutive_direction_run(ticks: list[tuple[int, float]]) -> tuple[str, int]:
    """
    Length of the current run of same-direction ticks, ending at the last tick.
    Returns ("UP"|"DOWN"|"FLAT", run_length). A single tick has no direction yet.
    """
    if len(ticks) < 2:
        return "FLAT", 0
    directions = []
    for i in range(1, len(ticks)):
        diff = ticks[i][1] - ticks[i - 1][1]
        directions.append("UP" if diff > 0 else "DOWN" if diff < 0 else "FLAT")

    last_dir = directions[-1]
    run = 0
    for d in reversed(directions):
        if d == last_dir and d != "FLAT":
            run += 1
        else:
            break
    return last_dir, run


def tick_imbalance(ticks: list[tuple[int, float]], window: int = 20) -> float | None:
    """(up_ticks - down_ticks) / total_ticks over the last `window` tick-to-tick moves."""
    if len(ticks) < 2:
        return None
    recent = ticks[-(window + 1):]
    up = down = 0
    for i in range(1, len(recent)):
        diff = recent[i][1] - recent[i - 1][1]
        if diff > 0:
            up += 1
        elif diff < 0:
            down += 1
    total = up + down
    if total == 0:
        return 0.0
    return (up - down) / total


def reversal_frequency(ticks: list[tuple[int, float]], window: int = 20) -> float | None:
    """Fraction of tick-to-tick moves that flip direction vs. the previous move."""
    if len(ticks) < 3:
        return None
    recent = ticks[-(window + 2):]
    diffs = [recent[i][1] - recent[i - 1][1] for i in range(1, len(recent))]
    signs = [1 if d > 0 else -1 if d < 0 else 0 for d in diffs]
    non_flat = [s for s in signs if s != 0]
    if len(non_flat) < 2:
        return None
    reversals = sum(1 for i in range(1, len(non_flat)) if non_flat[i] != non_flat[i - 1])
    return reversals / (len(non_flat) - 1)
