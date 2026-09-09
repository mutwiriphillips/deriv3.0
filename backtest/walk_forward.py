"""
Walk-forward validation (spec Part 19). Splits candle history into
chronological train/validate/test windows and rolls forward — data is never
shuffled, since the whole point is to catch a strategy that only worked in
one period.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.backtest.simulator import BacktestResult, run_backtest


@dataclass
class WalkForwardWindow:
    window_index: int
    train_range: tuple[int, int]      # candle indices [start, end)
    test_range: tuple[int, int]
    test_result: BacktestResult


@dataclass
class WalkForwardReport:
    windows: list[WalkForwardWindow] = field(default_factory=list)

    @property
    def performance_per_window(self) -> list[float | None]:
        return [w.test_result.realized_ev_per_stake for w in self.windows]

    @property
    def average_performance(self) -> float | None:
        vals = [v for v in self.performance_per_window if v is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def worst_window(self) -> float | None:
        vals = [v for v in self.performance_per_window if v is not None]
        return min(vals) if vals else None

    @property
    def best_window(self) -> float | None:
        vals = [v for v in self.performance_per_window if v is not None]
        return max(vals) if vals else None

    @property
    def stability_score(self) -> float | None:
        """
        1.0 = every window performed identically to the average; 0.0 = wildly
        inconsistent. Defined as 1 - (stdev / (|mean| + epsilon)), clamped to
        [0, 1] — a strategy profitable in only one window scores low here
        even if its overall average looks fine.
        """
        vals = [v for v in self.performance_per_window if v is not None]
        if len(vals) < 2:
            return None
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        stdev = variance ** 0.5
        score = 1 - (stdev / (abs(mean) + 1e-9))
        return max(0.0, min(1.0, score))


def make_walk_forward_windows(n_candles: int, n_windows: int, train_fraction: float = 0.7) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Splits [0, n_candles) into `n_windows` equal chronological chunks, each
    chunk itself split into a leading train slice and a trailing test slice.
    Returns a list of ((train_start, train_end), (test_start, test_end)).
    """
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    chunk_size = n_candles // n_windows
    if chunk_size < 2:
        raise ValueError("not enough candles to form the requested number of windows")

    windows = []
    for w in range(n_windows):
        start = w * chunk_size
        end = n_candles if w == n_windows - 1 else start + chunk_size
        train_end = start + int((end - start) * train_fraction)
        train_end = max(train_end, start + 1)
        windows.append(((start, train_end), (train_end, end)))
    return windows


def run_walk_forward(
    symbol,
    market_type,
    duration_s,
    timestamps,
    opens,
    highs,
    lows,
    closes,
    strategy_factory,
    duration_candles,
    payout_ratio,
    n_windows: int = 4,
    stake: float = 1.0,
    min_probability_edge: float = 0.0,
    min_expected_value: float = 0.0,
    warmup: int = 60,
) -> WalkForwardReport:
    """
    `strategy_factory(train_range)` is called once per window with that
    window's (start, end) train-slice indices, and must return a strategy
    instance ready to evaluate. For a stateless baseline (e.g. RandomStrategy),
    just ignore the argument and return a fixed instance; a strategy that
    needs to fit on training data can use train_range to slice the same
    closes/candles arrays the caller already has and fit from that slice.

    Only the *test* slice of each window is backtested and scored here; the
    train slice exists so a stateful strategy_factory has something to fit
    on. This function doesn't implicitly train anything itself — it stays a
    Strategy-protocol consumer, not a model-fitting framework.
    """
    n = len(closes)
    window_ranges = make_walk_forward_windows(n, n_windows)
    report = WalkForwardReport()

    for idx, (train_range, test_range) in enumerate(window_ranges):
        strategy = strategy_factory(train_range)
        test_start, test_end = test_range
        # Feed the backtester history up through the end of this window's test
        # slice, but rely on `warmup` + entry-loop bounds to keep it from
        # trading before test_start — a real "windowed" backtest would slice
        # more surgically; this is a reasonable first pass, not a claim of
        # perfect isolation from prior windows' data.
        result = run_backtest(
            symbol=symbol,
            market_type=market_type,
            duration_s=duration_s,
            timestamps=timestamps[:test_end],
            opens=opens[:test_end],
            highs=highs[:test_end],
            lows=lows[:test_end],
            closes=closes[:test_end],
            strategy=strategy,
            duration_candles=duration_candles,
            payout_ratio=payout_ratio,
            stake=stake,
            min_probability_edge=min_probability_edge,
            min_expected_value=min_expected_value,
            warmup=max(warmup, test_start),
        )
        report.windows.append(
            WalkForwardWindow(window_index=idx, train_range=train_range, test_range=test_range, test_result=result)
        )

    return report
