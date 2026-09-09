import pytest

from app.risk.capital_ramp import CapitalRampManager, CapitalStage
from app.monitoring.trade_log import LifetimeStats


def make_stats(trade_count, wins, total_pnl):
    return LifetimeStats(trade_count=trade_count, wins=wins, win_rate=wins / trade_count if trade_count else None, total_pnl=total_pnl, latencies_ms=[])


def test_starts_at_demo_with_full_stake_fraction():
    manager = CapitalRampManager()
    assert manager.stage == CapitalStage.DEMO
    assert manager.stake_fraction() == pytest.approx(1.0)


def test_max_stake_for_stage_scales_correctly():
    manager = CapitalRampManager(stage=CapitalStage.MICRO_LIVE)
    assert manager.max_stake_for_stage(100.0) == pytest.approx(5.0)   # 5% of 100
    manager.stage = CapitalStage.SMALL_LIVE
    assert manager.max_stake_for_stage(100.0) == pytest.approx(25.0)
    manager.stage = CapitalStage.NORMAL_LIVE
    assert manager.max_stake_for_stage(100.0) == pytest.approx(100.0)


def test_normal_live_is_never_eligible_to_advance_further():
    manager = CapitalRampManager(stage=CapitalStage.NORMAL_LIVE)
    result = manager.evaluate_advancement_eligibility(make_stats(1000, 600, 500.0), break_even_probability=0.5)
    assert result.eligible is False
    assert "highest stage" in result.reason


def test_too_few_trades_blocks_advancement():
    manager = CapitalRampManager(stage=CapitalStage.MICRO_LIVE, min_trades_to_advance=50)
    result = manager.evaluate_advancement_eligibility(make_stats(10, 8, 50.0), break_even_probability=0.5556)
    assert result.eligible is False
    assert "10 trades" in result.reason


def test_negative_pnl_blocks_advancement_even_with_enough_trades():
    manager = CapitalRampManager(stage=CapitalStage.MICRO_LIVE, min_trades_to_advance=50)
    result = manager.evaluate_advancement_eligibility(make_stats(100, 40, -50.0), break_even_probability=0.5556)
    assert result.eligible is False
    assert "not positive" in result.reason


def test_statistically_unreliable_performance_blocks_advancement():
    manager = CapitalRampManager(stage=CapitalStage.MICRO_LIVE, min_trades_to_advance=50)
    # 56% win rate at 55.56% break-even, only 60 trades -- positive PnL but not statistically significant
    result = manager.evaluate_advancement_eligibility(make_stats(60, 34, 5.0), break_even_probability=0.5556)
    assert result.eligible is False


def test_strong_reliable_performance_is_eligible():
    manager = CapitalRampManager(stage=CapitalStage.MICRO_LIVE, min_trades_to_advance=50)
    # 65% win rate over 500 trades at 55.56% break-even -- genuinely strong, reliable edge
    result = manager.evaluate_advancement_eligibility(make_stats(500, 325, 1000.0), break_even_probability=0.5556)
    assert result.eligible is True


def test_advance_moves_to_next_stage_and_snapshots_trade_id():
    manager = CapitalRampManager(stage=CapitalStage.DEMO)
    manager.advance(current_latest_trade_id=42)
    assert manager.stage == CapitalStage.MICRO_LIVE
    assert manager.stage_entry_trade_id == 42


def test_advance_rejects_advancing_past_normal_live():
    manager = CapitalRampManager(stage=CapitalStage.NORMAL_LIVE)
    with pytest.raises(ValueError):
        manager.advance(current_latest_trade_id=100)


def test_demote_moves_down_one_stage():
    manager = CapitalRampManager(stage=CapitalStage.SMALL_LIVE)
    manager.demote(current_latest_trade_id=99, reason="drawdown limit hit")
    assert manager.stage == CapitalStage.MICRO_LIVE
    assert manager.stage_entry_trade_id == 99


def test_demote_from_demo_is_a_safe_noop():
    manager = CapitalRampManager(stage=CapitalStage.DEMO, stage_entry_trade_id=5)
    manager.demote(current_latest_trade_id=99, reason="whatever")
    assert manager.stage == CapitalStage.DEMO
    assert manager.stage_entry_trade_id == 5   # unchanged, since nothing happened


def test_full_ramp_sequence_advance_advance_demote():
    manager = CapitalRampManager(stage=CapitalStage.DEMO)
    manager.advance(current_latest_trade_id=10)
    assert manager.stage == CapitalStage.MICRO_LIVE
    manager.advance(current_latest_trade_id=60)
    assert manager.stage == CapitalStage.SMALL_LIVE
    manager.demote(current_latest_trade_id=80, reason="consecutive loss limit")
    assert manager.stage == CapitalStage.MICRO_LIVE
    assert manager.stage_entry_trade_id == 80


def test_status_reflects_current_state():
    manager = CapitalRampManager(stage=CapitalStage.SMALL_LIVE, stage_entry_trade_id=30, min_trades_to_advance=75)
    status = manager.status()
    assert status == {
        "stage": "SMALL_LIVE",
        "stake_fraction": 0.25,
        "stage_entry_trade_id": 30,
        "min_trades_to_advance": 75,
    }
