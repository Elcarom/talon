"""Comprehensive unit tests for the TALON Risk Engine, safety gates, and logging redaction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from talon.config import (
    LIVE_ACKNOWLEDGEMENT_PHRASE,
    ConfigError,
    DataConfig,
    ExecutionConfig,
    RiskConfig,
    load_config,
)
from talon.data import generate_synthetic_market
from talon.execution import LiveExecutionDisabledError, PaperExecutionAdapter
from talon.logging_utils import get_logger, redact_text, redact_value
from talon.risk import RiskEngine
from talon.types import BarSeries, ExecutionMode, MarketMatrix, PortfolioState, RiskActionKind


def _make_state(symbols: tuple[str, ...], equity: float = 100_000.0) -> PortfolioState:
    return PortfolioState(
        cash_usd=equity,
        equity_usd=equity,
        peak_equity_usd=equity,
        day_start_equity_usd=equity,
        positions={s: 0.0 for s in symbols},
        weights={s: 0.0 for s in symbols},
    )


def test_live_mode_blocked_by_default(tmp_path: Path) -> None:
    cfg = load_config(config_path=None, env={})
    assert cfg.mode == ExecutionMode.PAPER

    with pytest.raises(ConfigError, match="LIVE trading mode is locked"):
        load_config(config_path=None, env={"TALON_MODE": "live"})

    with pytest.raises(ConfigError, match="live_confirmed: true"):
        load_config(
            config_path=None,
            env={"TALON_MODE": "live", "TALON_LIVE_TRADING": LIVE_ACKNOWLEDGEMENT_PHRASE},
        )

    # Now create an explicit live config with live_confirmed: true
    live_cfg_file = tmp_path / "live.yaml"
    live_cfg_file.write_text("mode: live\nlive_confirmed: true\n", encoding="utf-8")
    unlocked = load_config(
        config_path=live_cfg_file,
        env={"TALON_MODE": "live", "TALON_LIVE_TRADING": LIVE_ACKNOWLEDGEMENT_PHRASE},
    )
    assert unlocked.mode == ExecutionMode.LIVE


def test_paper_adapter_blocks_unconfirmed_live_mode() -> None:
    with pytest.raises(LiveExecutionDisabledError):
        PaperExecutionAdapter(ExecutionConfig(), mode=ExecutionMode.LIVE, live_ack="")

    adapter = PaperExecutionAdapter(
        ExecutionConfig(),
        mode=ExecutionMode.LIVE,
        live_ack=LIVE_ACKNOWLEDGEMENT_PHRASE,
    )
    assert adapter.mode == ExecutionMode.LIVE


def test_config_validation_errors(tmp_path: Path) -> None:
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("initial_equity_usd: -500\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="initial_equity_usd"):
        load_config(bad_cfg)

    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nonexistent.yaml")

    with pytest.raises(ConfigError, match="Invalid execution mode"):
        load_config(None, env={"TALON_MODE": "invalid_mode"})


def test_risk_engine_max_drawdown_halt() -> None:
    market = generate_synthetic_market(DataConfig(num_bars=60, seed=1))
    risk_cfg = RiskConfig(max_drawdown_pct=0.15)
    engine = RiskEngine(risk_cfg)

    state = _make_state(market.symbols, equity=84_000.0)
    state.peak_equity_usd = 100_000.0  # 16% drawdown > 15% limit
    state.weights["BTC-USD"] = 0.25

    proposed = {s: 0.20 for s in market.symbols}
    report = engine.evaluate(int(market.timestamps[-1]), state, proposed, market)

    assert report.halted is True
    assert state.halted is True
    assert report.gross_leverage_approved == 0.0
    assert report.decisions["BTC-USD"].action == RiskActionKind.HALTED_FLATTEN
    assert report.decisions["BTC-USD"].approved_weight == 0.0
    assert report.decisions["ETH-USD"].action == RiskActionKind.VETOED


def test_risk_engine_daily_loss_circuit_breaker() -> None:
    market = generate_synthetic_market(DataConfig(num_bars=60, seed=2))
    risk_cfg = RiskConfig(daily_loss_limit_pct=0.04)
    engine = RiskEngine(risk_cfg)

    state = _make_state(market.symbols, equity=95_500.0)
    state.day_start_equity_usd = 100_000.0  # 4.5% daily loss > 4.0% limit
    state.weights["ETH-USD"] = -0.20

    proposed = {s: 0.15 for s in market.symbols}
    report = engine.evaluate(int(market.timestamps[-1]), state, proposed, market)

    assert report.halted is False
    assert report.halt_reason is not None and "DAILY_LOSS_LIMIT_BREACH" in report.halt_reason
    assert report.decisions["ETH-USD"].action == RiskActionKind.HALTED_FLATTEN
    assert report.decisions["ETH-USD"].approved_weight == 0.0


def test_risk_engine_asset_cap_and_gross_leverage() -> None:
    market = generate_synthetic_market(DataConfig(num_bars=60, seed=3))
    risk_cfg = RiskConfig(
        max_asset_weight=0.30,
        max_gross_leverage=1.00,
        portfolio_vol_target=0.99,  # high so we isolate asset cap + gross leverage
        max_pairwise_correlation=0.99,
    )
    engine = RiskEngine(risk_cfg)
    state = _make_state(market.symbols)

    # Propose 0.50 weight on all 5 symbols (total 2.50x gross)
    proposed = {s: 0.50 for s in market.symbols}
    report = engine.evaluate(int(market.timestamps[-1]), state, proposed, market)

    # First each symbol is capped to 0.30 (sum = 1.50), then scaled down to 1.00 gross (0.20 each)
    assert np.isclose(report.gross_leverage_approved, 1.00, atol=1e-6)
    for sym in market.symbols:
        dec = report.decisions[sym]
        assert dec.action == RiskActionKind.SCALED
        assert np.isclose(dec.approved_weight, 0.20, atol=1e-6)


def test_risk_engine_correlation_cluster_cap_and_vol_target() -> None:
    # Construct two nearly identical asset series so pairwise correlation > 0.99
    ts = np.arange(1000, 1000 + 50 * 86400, 86400, dtype=np.int64)
    rng = np.random.default_rng(7)
    rets = rng.normal(0.001, 0.04, size=50)  # High daily vol (~76% ann)
    closes_a = 100.0 * np.exp(np.cumsum(rets))
    closes_b = 50.0 * np.exp(np.cumsum(rets + rng.normal(0, 0.0005, size=50)))

    s_a = BarSeries("ASSET-A", ts, closes_a, closes_a * 1.01, closes_a * 0.99, closes_a, np.ones(50) * 1e6)
    s_b = BarSeries("ASSET-B", ts, closes_b, closes_b * 1.01, closes_b * 0.99, closes_b, np.ones(50) * 1e6)
    market = MarketMatrix(ts, ("ASSET-A", "ASSET-B"), {"ASSET-A": s_a, "ASSET-B": s_b})

    risk_cfg = RiskConfig(
        max_asset_weight=0.35,
        max_pairwise_correlation=0.80,
        correlated_cluster_cap=0.40,
        portfolio_vol_target=0.20,
        vol_lookback=30,
    )
    engine = RiskEngine(risk_cfg)
    state = _make_state(("ASSET-A", "ASSET-B"))

    proposed = {"ASSET-A": 0.30, "ASSET-B": 0.30}
    report = engine.evaluate(int(ts[-1]), state, proposed, market)

    assert report.ex_ante_vol_ann <= 0.20 + 1e-6
    for sym in ("ASSET-A", "ASSET-B"):
        reasons_joined = " ".join(report.decisions[sym].reasons)
        assert "SCALED_CORRELATION_CLUSTER" in reasons_joined
        assert "SCALED_PORTFOLIO_VOL" in reasons_joined


def test_risk_engine_stop_loss_and_rebalance_deadband() -> None:
    market = generate_synthetic_market(DataConfig(num_bars=60, seed=10))
    risk_cfg = RiskConfig(rebalance_deadband=0.03)
    engine = RiskEngine(risk_cfg)
    state = _make_state(market.symbols)

    # Set an active long position on BTC-USD with a stop price ABOVE current close -> stop triggered!
    btc_close = float(market.series["BTC-USD"].close[-1])
    state.weights["BTC-USD"] = 0.25
    state.stop_prices["BTC-USD"] = btc_close * 1.05

    # Set an active position on ETH-USD with a tiny proposed adjustment within the 3% deadband
    state.weights["ETH-USD"] = 0.20
    proposed = {s: 0.0 for s in market.symbols}
    proposed["BTC-USD"] = 0.25
    proposed["ETH-USD"] = 0.21  # delta = +0.01 < 0.03 deadband

    report = engine.evaluate(int(market.timestamps[-1]), state, proposed, market)

    assert report.decisions["BTC-USD"].approved_weight == 0.0
    assert any("FORCED_STOP_EXIT" in r for r in report.decisions["BTC-USD"].reasons)

    assert report.decisions["ETH-USD"].action == RiskActionKind.VETOED
    assert report.decisions["ETH-USD"].approved_weight == 0.20
    assert any("VETOED_REBALANCE_DEADBAND" in r for r in report.decisions["ETH-USD"].reasons)


def test_risk_engine_vetoes_nan_or_nonpositive_market_data() -> None:
    market = generate_synthetic_market(DataConfig(num_bars=40, seed=11))
    bad_closes = market.series["SOL-USD"].close.copy()
    bad_closes[-1] = float("nan")
    market.series["SOL-USD"] = BarSeries(
        symbol="SOL-USD",
        timestamps=market.timestamps,
        open=market.series["SOL-USD"].open,
        high=market.series["SOL-USD"].high,
        low=market.series["SOL-USD"].low,
        close=bad_closes,
        volume=market.series["SOL-USD"].volume,
    )

    engine = RiskEngine(RiskConfig())
    state = _make_state(market.symbols)
    proposed = {s: 0.20 for s in market.symbols}
    report = engine.evaluate(int(market.timestamps[-1]), state, proposed, market)

    assert report.decisions["SOL-USD"].action == RiskActionKind.VETOED
    assert "VETOED_INVALID_MARKET_DATA" in report.decisions["SOL-USD"].reasons


def test_secret_redaction_and_structured_logger() -> None:
    raw_msg = "Auth header Bearer secret_token_1234567890 and key AIzaSyD123456789012345678901234567890"
    scrubbed = redact_text(raw_msg)
    assert "secret_token_1234567890" not in scrubbed
    assert "AIzaSyD" not in scrubbed

    nested = {
        "api_key": "my-secret-key",
        "nested": {"user_password": "pw123", "safe_metric": 1.45},
    }
    redacted_dict = redact_value(nested)
    assert redacted_dict["api_key"] == "[REDACTED]"
    assert redacted_dict["nested"]["user_password"] == "[REDACTED]"
    assert redacted_dict["nested"]["safe_metric"] == 1.45

    logger = get_logger("talon_test")
    logger.info("Test log entry", extra={"structured_data": nested})
