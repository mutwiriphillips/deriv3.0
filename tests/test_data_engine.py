import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

from app.data.quality import DataQualityScore, SymbolStreamState
from app.data.tick_ingestion import TickIngestor
from app.data.candles import aggregate_ticks_to_candles
from app.data.historical_store import fetch_and_store_candles, get_stored_range, store_candles
from app.data.candles import Candle


def make_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()
    return db_path


# --- quality.py ---

def test_normal_ticks_score_high_quality():
    state = SymbolStreamState(stale_after_s=30.0)
    r1 = state.evaluate(epoch=1000, price=1.085, received_monotonic=0.0)
    r2 = state.evaluate(epoch=1001, price=1.086, received_monotonic=1.0)
    assert r1.score == DataQualityScore.HIGH_QUALITY
    assert r2.score == DataQualityScore.HIGH_QUALITY


def test_duplicate_tick_detected():
    state = SymbolStreamState()
    state.evaluate(epoch=1000, price=1.085, received_monotonic=0.0)
    dup = state.evaluate(epoch=1000, price=1.085, received_monotonic=0.5)
    assert dup.is_duplicate
    assert dup.score == DataQualityScore.QUESTIONABLE


def test_out_of_order_tick_detected():
    state = SymbolStreamState()
    state.evaluate(epoch=1000, price=1.085, received_monotonic=0.0)
    ooo = state.evaluate(epoch=999, price=1.084, received_monotonic=0.5)
    assert ooo.is_out_of_order
    assert ooo.score == DataQualityScore.QUESTIONABLE


def test_stale_gap_detected():
    state = SymbolStreamState(stale_after_s=5.0)
    state.evaluate(epoch=1000, price=1.085, received_monotonic=0.0)
    stale = state.evaluate(epoch=1010, price=1.086, received_monotonic=20.0)
    assert stale.is_stale
    assert stale.score == DataQualityScore.ACCEPTABLE


def test_invalid_price_is_unusable():
    state = SymbolStreamState()
    r = state.evaluate(epoch=1000, price=-1.0, received_monotonic=0.0)
    assert r.score == DataQualityScore.UNUSABLE


def test_duplicate_does_not_corrupt_last_known_state():
    state = SymbolStreamState()
    state.evaluate(epoch=1000, price=1.085, received_monotonic=0.0)
    state.evaluate(epoch=1000, price=1.085, received_monotonic=0.5)  # duplicate
    ok = state.evaluate(epoch=1001, price=1.086, received_monotonic=1.0)
    assert ok.score == DataQualityScore.HIGH_QUALITY  # not treated as out-of-order


# --- tick_ingestion.py ---

def test_tick_ingestor_stores_ticks_with_quality_score(tmp_path):
    db_path = make_db(tmp_path)
    ingestor = TickIngestor(db_path)
    score = ingestor.ingest_tick("frxEURUSD", {"epoch": 1700000000, "quote": 1.085}, received_monotonic=0.0)
    assert score == DataQualityScore.HIGH_QUALITY

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT symbol, price, quality_score FROM ticks").fetchone()
    conn.close()
    assert row == ("frxEURUSD", 1.085, DataQualityScore.HIGH_QUALITY)


def test_tick_ingestor_stores_bad_tick_too(tmp_path):
    db_path = make_db(tmp_path)
    ingestor = TickIngestor(db_path)
    score = ingestor.ingest_tick("frxEURUSD", {"epoch": 1700000000, "quote": -5.0}, received_monotonic=0.0)
    assert score == DataQualityScore.UNUSABLE
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
    conn.close()
    assert count == 1  # bad ticks are stored, not dropped, so NO_TRADE reasoning stays reproducible


# --- candles.py ---

def test_aggregate_ticks_to_candles_basic():
    ticks = [(100, 1.0), (105, 1.2), (150, 0.9), (160, 2.0)]
    candles = aggregate_ticks_to_candles("R_100", ticks, duration_s=60)
    assert len(candles) == 2
    first = candles[0]
    assert first.timestamp == 60
    assert first.open == 1.0
    assert first.high == 1.2
    assert first.low == 1.0
    assert first.close == 1.2
    assert first.tick_count == 2
    assert candles[1].timestamp == 120
    assert candles[1].open == 0.9
    assert candles[1].close == 2.0


def test_aggregate_empty_ticks_returns_empty():
    assert aggregate_ticks_to_candles("R_100", [], duration_s=60) == []


# --- historical_store.py ---

def test_store_candles_dedupes(tmp_path):
    db_path = make_db(tmp_path)
    c = Candle(symbol="frxEURUSD", duration_s=60, timestamp=1000, open=1, high=1.1, low=0.9, close=1.05, tick_count=5)
    inserted_first = store_candles(db_path, [c])
    inserted_second = store_candles(db_path, [c])  # exact duplicate
    assert inserted_first == 1
    assert inserted_second == 0


def test_get_stored_range_empty(tmp_path):
    db_path = make_db(tmp_path)
    assert get_stored_range(db_path, "frxEURUSD", 60) == (None, None)


@pytest.mark.asyncio
async def test_fetch_and_store_candles_only_requests_missing_range(tmp_path):
    db_path = make_db(tmp_path)
    existing = Candle(symbol="frxEURUSD", duration_s=60, timestamp=1000, open=1, high=1.1, low=0.9, close=1.0, tick_count=1)
    store_candles(db_path, [existing])

    with patch("app.data.historical_store.DerivPublicClient") as MockClient:
        instance = MockClient.return_value
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=None)
        instance.ticks_history = AsyncMock(return_value=[
            {"epoch": 1060, "open": 1.0, "high": 1.2, "low": 0.95, "close": 1.1},
        ])

        inserted = await fetch_and_store_candles(db_path, "frxEURUSD", duration_s=60)

        # The request must start strictly after what we already have (1000 + 60),
        # proving we didn't re-request the range we already stored.
        called_kwargs = instance.ticks_history.call_args.kwargs
        assert called_kwargs["start"] == "1060"

    assert inserted == 1


@pytest.mark.asyncio
async def test_fetch_and_store_candles_treats_invalid_start_end_as_no_new_data(tmp_path):
    """
    Confirmed against a real production error: when `start` (last stored
    candle + duration_s) lands at or after the server's current latest
    completed candle, Deriv returns InvalidStartEnd. This is expected --
    nothing new has closed yet -- and must return 0, not raise.
    """
    from app.markets.ws_client import DerivApiError

    db_path = make_db(tmp_path)
    existing = Candle(symbol="frxEURUSD", duration_s=60, timestamp=1000, open=1, high=1.1, low=0.9, close=1.0, tick_count=1)
    store_candles(db_path, [existing])

    with patch("app.data.historical_store.DerivPublicClient") as MockClient:
        instance = MockClient.return_value
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=None)
        instance.ticks_history = AsyncMock(
            side_effect=DerivApiError({"code": "InvalidStartEnd", "message": "Start time must be before end time"})
        )

        inserted = await fetch_and_store_candles(db_path, "frxEURUSD", duration_s=60)

    assert inserted == 0  # did not raise


@pytest.mark.asyncio
async def test_fetch_and_store_candles_reraises_other_api_errors(tmp_path):
    """Only InvalidStartEnd is treated as benign -- any other API error must still propagate."""
    from app.markets.ws_client import DerivApiError

    db_path = make_db(tmp_path)
    existing = Candle(symbol="frxEURUSD", duration_s=60, timestamp=1000, open=1, high=1.1, low=0.9, close=1.0, tick_count=1)
    store_candles(db_path, [existing])

    with patch("app.data.historical_store.DerivPublicClient") as MockClient:
        instance = MockClient.return_value
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=None)
        instance.ticks_history = AsyncMock(
            side_effect=DerivApiError({"code": "SomeOtherError", "message": "something genuinely wrong"})
        )

        with pytest.raises(DerivApiError):
            await fetch_and_store_candles(db_path, "frxEURUSD", duration_s=60)


@pytest.mark.asyncio
async def test_fetch_and_store_candles_reraises_invalid_start_end_on_first_ever_fetch(tmp_path):
    """
    If InvalidStartEnd happens with no prior stored data (start=None), that's
    NOT the benign "nothing new yet" case -- something else is wrong, and it
    should still surface as an error rather than being silently swallowed.
    """
    from app.markets.ws_client import DerivApiError

    db_path = make_db(tmp_path)  # empty -- no existing candles, so start will be None

    with patch("app.data.historical_store.DerivPublicClient") as MockClient:
        instance = MockClient.return_value
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=None)
        instance.ticks_history = AsyncMock(
            side_effect=DerivApiError({"code": "InvalidStartEnd", "message": "Start time must be before end time"})
        )

        with pytest.raises(DerivApiError):
            await fetch_and_store_candles(db_path, "frxEURUSD", duration_s=60)
