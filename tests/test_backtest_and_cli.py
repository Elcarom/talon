"""Comprehensive tests for strategies, indicators, backtester causality, walk-forward, and CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from talon.backtest import run_backtest, run_walk_forward
from talon.cli import main as cli_main
from talon.config import DataConfig, TalonConfig, WalkForwardConfig, load_config
from talon.data import generate_synthetic_market, load_market_data
from talon.indicators import (
    multi_horizon_donchian_score,
    ols_cointegration_spread,
    realized_volatility_ann,
    wilder_atr,
)
from talon.metrics import compute_drawdown_series, compute_metrics
from talon.reporting import generate_reports
from talon.types import BarSeries


def test_bar_series_validation() -> None:
    ts = np.array([100, 200, 300], dtype=np.int64)
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    bs = BarSeries("BTC-USD", ts, arr, arr, arr, arr, arr)
    assert len(bs.slice_up_to(1).timestamps) == 2

    with pytest.raises(ValueError, match="Mismatched array lengths"):
        BarSeries("BTC-USD", ts, arr[:2], arr, arr, arr, arr)

    bad_ts = np.array([100, 300, 200], dtype=np.int64)
    with pytest.raises(ValueError, match="strictly increasing"):
        BarSeries("BTC-USD", bad_ts, arr, arr, arr, arr, arr)


def test_indicators_edge_cases() -> None:
    short_arr = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    assert realized_volatility_ann(short_arr, lookback=30) is None
    assert multi_horizon_donchian_score(short_arr) is None
    assert ols_cointegration_spread(short_arr, short_arr) is None

    market = generate_synthetic_market(DataConfig(num_bars=180, seed=42))
    btc = market.series["BTC-USD"]
    eth = market.series["ETH-USD"]

    rv = realized_volatility_ann(btc.close, lookback=30)
    assert rv is not None and rv > 0.0

    atr = wilder_atr(btc, period=14)
    assert atr is not None and atr > 0.0

    donchian = multi_horizon_donchian_score(btc.close)
    assert donchian is not None
    score, stop = donchian
    assert -1.0 <= score <= 1.0

    coint = ols_cointegration_spread(eth.close, btc.close, ols_window=60, z_window=20)
    assert coint is not None
    beta, z_score, half_life = coint
    assert np.isfinite(beta)
    assert np.isfinite(z_score)
    assert half_life > 0.0


def test_backtest_no_lookahead_causality() -> None:
    """Prove mathematically that future price mutations after bar T cannot affect equity up to bar T."""
    cfg = TalonConfig(data=DataConfig(num_bars=220, seed=99))
    market_a = generate_synthetic_market(cfg.data)
    market_b = generate_synthetic_market(cfg.data)

    cutoff = 160
    # Mutate all prices after `cutoff` in market_b by a massive shock
    for sym in market_b.symbols:
        s = market_b.series[sym]
        mut_open = s.open.copy()
        mut_high = s.high.copy()
        mut_low = s.low.copy()
        mut_close = s.close.copy()
        mut_open[cutoff + 1 :] *= 3.5
        mut_high[cutoff + 1 :] *= 3.5
        mut_low[cutoff + 1 :] *= 3.5
        mut_close[cutoff + 1 :] *= 3.5
        market_b.series[sym] = BarSeries(
            symbol=sym,
            timestamps=s.timestamps,
            open=mut_open,
            high=mut_high,
            low=mut_low,
            close=mut_close,
            volume=s.volume,
        )

    res_a = run_backtest(cfg, market_a)
    res_b = run_backtest(cfg, market_b)

    # Every single equity point up to `cutoff` must be bit-for-bit identical
    np.testing.assert_allclose(
        res_a.equity_curve[: cutoff + 1],
        res_b.equity_curve[: cutoff + 1],
        rtol=0.0,
        atol=1e-9,
    )


def test_walk_forward_and_reporting(tmp_path: Path) -> None:
    cfg = TalonConfig(
        data=DataConfig(num_bars=450, seed=42),
        walk_forward=WalkForwardConfig(train_bars=180, test_bars=75, step_bars=75),
    )
    market = load_market_data(cfg.data)
    bt_res = run_backtest(cfg, market)
    wf_res = run_walk_forward(cfg, market)

    assert len(wf_res.folds) >= 3
    assert bt_res.metrics.total_trades > 0
    assert bt_res.metrics.total_fees_usd > 0.0
    assert bt_res.metrics.total_slippage_usd > 0.0

    paths = generate_reports(cfg, bt_res, wf_res, output_dir=tmp_path)
    assert paths["report_md"].exists()
    assert paths["metrics_json"].exists()
    assert paths["equity_svg"].exists()
    assert paths["drawdown_svg"].exists()


def test_metrics_edge_cases() -> None:
    empty_dd = compute_drawdown_series(np.array([], dtype=np.float64))
    assert len(empty_dd) == 0

    single_m = compute_metrics(np.array([100_000.0], dtype=np.float64), [])
    assert single_m.sharpe_ratio == 0.0
    assert single_m.max_drawdown_pct == 0.0


def test_cli_commands(tmp_path: Path) -> None:
    cfg_file = tmp_path / "test_config.yaml"
    cfg = load_config(None)
    assert cfg.data.source == "synthetic"

    cfg_file.write_text(
        "mode: paper\n"
        "initial_equity_usd: 50000.0\n"
        "data:\n"
        "  source: synthetic\n"
        "  symbols: [BTC-USD, ETH-USD, SOL-USD]\n"
        "  num_bars: 360\n"
        "  seed: 42\n"
        "walk_forward:\n"
        "  train_bars: 160\n"
        "  test_bars: 60\n"
        "  step_bars: 60\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "reports"
    rc_bt = cli_main(["-c", str(cfg_file), "backtest", "-o", str(out_dir)])
    assert rc_bt == 0
    assert (out_dir / "backtest_report.md").exists()

    rc_wf = cli_main(["-c", str(cfg_file), "walk-forward"])
    assert rc_wf == 0

    rc_tick = cli_main(["-c", str(cfg_file), "paper-tick"])
    assert rc_tick == 0
