"""Event-driven causal backtester and walk-forward evaluation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from talon.config import TalonConfig
from talon.execution import PaperExecutionAdapter
from talon.indicators import realized_volatility_ann
from talon.metrics import PerformanceMetrics, compute_drawdown_series, compute_metrics
from talon.risk import RiskEngine
from talon.strategies import Strategy, build_default_strategies, combine_strategy_intents
from talon.types import (
    FillEvent,
    MarketMatrix,
    PortfolioRiskReport,
    PortfolioState,
    RiskActionKind,
    SignalIntent,
)


@dataclass(frozen=True)
class BacktestResult:
    """Complete record of a causal backtest simulation."""

    timestamps: NDArray[np.int64]
    equity_curve: NDArray[np.float64]
    drawdown_curve: NDArray[np.float64]
    fills: list[FillEvent]
    risk_reports: list[PortfolioRiskReport]
    metrics: PerformanceMetrics
    strategy_metrics: dict[str, PerformanceMetrics]


@dataclass(frozen=True)
class WalkForwardFold:
    """Single In-Sample / Out-of-Sample fold evaluation."""

    fold_index: int
    train_start_ts: int
    train_end_ts: int
    test_start_ts: int
    test_end_ts: int
    in_sample_metrics: PerformanceMetrics
    out_of_sample_metrics: PerformanceMetrics


@dataclass(frozen=True)
class WalkForwardSummary:
    """Aggregated walk-forward evaluation across all rolling folds."""

    folds: list[WalkForwardFold]
    oos_combined_metrics: PerformanceMetrics
    mean_is_sharpe: float
    mean_oos_sharpe: float
    sharpe_decay_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_folds": len(self.folds),
            "mean_is_sharpe": round(self.mean_is_sharpe, 3),
            "mean_oos_sharpe": round(self.mean_oos_sharpe, 3),
            "sharpe_decay_ratio": round(self.sharpe_decay_ratio, 3),
            "oos_combined_metrics": self.oos_combined_metrics.to_dict(),
            "folds": [
                {
                    "fold_index": f.fold_index,
                    "train_start_ts": int(f.train_start_ts),
                    "train_end_ts": int(f.train_end_ts),
                    "test_start_ts": int(f.test_start_ts),
                    "test_end_ts": int(f.test_end_ts),
                    "in_sample_sharpe": f.in_sample_metrics.sharpe_ratio,
                    "out_of_sample_sharpe": f.out_of_sample_metrics.sharpe_ratio,
                    "out_of_sample_return_pct": f.out_of_sample_metrics.total_return_pct,
                    "out_of_sample_max_dd_pct": f.out_of_sample_metrics.max_drawdown_pct,
                }
                for f in self.folds
            ],
        }


def _run_portfolio_loop(
    cfg: TalonConfig,
    market: MarketMatrix,
    strategies: list[Strategy],
    initial_equity: float | None = None,
) -> tuple[
    NDArray[np.float64],
    list[FillEvent],
    list[PortfolioRiskReport],
    float,
    int,
    int,
]:
    """Execute a strictly causal bar-by-bar simulation loop.

    At bar t (for t = 0 .. N-2):
    - Mark portfolio to close[t] (and reset daily start equity for next bar)
    - Slice market causally up to bar t: `market.slice_up_to(t)`
    - Generate strategy intents using only data <= t
    - Evaluate portfolio risk engine at t
    - Execute approved weight changes at bar t+1 OPEN (`open[t+1]`) with fees & slippage
    - Mark portfolio to bar t+1 CLOSE (`close[t+1]`)
    """
    n = market.num_bars
    start_eq = initial_equity if initial_equity is not None else cfg.initial_equity_usd
    state = PortfolioState(
        cash_usd=start_eq,
        equity_usd=start_eq,
        peak_equity_usd=start_eq,
        day_start_equity_usd=start_eq,
        positions={sym: 0.0 for sym in market.symbols},
        weights={sym: 0.0 for sym in market.symbols},
    )

    risk_engine = RiskEngine(cfg.risk, bars_per_year=cfg.bars_per_year)
    adapter = PaperExecutionAdapter(
        cfg.execution,
        mode=cfg.mode,
        bars_per_year=cfg.bars_per_year,
    )

    equity_curve = np.zeros(n, dtype=np.float64)
    equity_curve[0] = start_eq
    all_fills: list[FillEvent] = []
    risk_reports: list[PortfolioRiskReport] = []
    veto_count = 0
    scale_count = 0

    for t in range(n - 1):
        # 1. Causal slice up to bar t
        causal_slice = market.slice_up_to(t)
        close_prices_t = {sym: float(causal_slice.series[sym].close[-1]) for sym in market.symbols}

        # Mark to market at bar t close before decision
        adapter.mark_to_market(state, close_prices_t, apply_funding=(t > 0))
        equity_curve[t] = state.equity_usd

        # 2. Generate strategy signals from confirmed bars <= t
        intents_by_strat: dict[str, list[SignalIntent]] = {}
        for strat in strategies:
            intents_by_strat[strat.strategy_id] = strat.generate_signals(
                causal_slice,
                bars_per_year=cfg.bars_per_year,
            )

        proposed_weights, proposed_stops = combine_strategy_intents(
            market.symbols,
            strategies,
            intents_by_strat,
        )

        # 3. Risk Engine evaluation
        report = risk_engine.evaluate(
            timestamp=int(market.timestamps[t]),
            state=state,
            proposed_weights=proposed_weights,
            market=causal_slice,
        )
        risk_reports.append(report)

        for dec in report.decisions.values():
            if dec.action in (RiskActionKind.VETOED, RiskActionKind.HALTED_FLATTEN):
                if abs(dec.proposed_weight - dec.approved_weight) > 1e-6:
                    veto_count += 1
            elif dec.action == RiskActionKind.SCALED:
                scale_count += 1

        # 4. Execute approved orders at bar t+1 OPEN (eliminating look-ahead bias)
        open_prices_next = {sym: float(market.series[sym].open[t + 1]) for sym in market.symbols}
        daily_vols = {}
        for sym in market.symbols:
            rv = realized_volatility_ann(
                causal_slice.series[sym].close,
                lookback=20,
                bars_per_year=cfg.bars_per_year,
            )
            daily_vols[sym] = (rv / float(np.sqrt(cfg.bars_per_year))) if rv is not None else 0.02

        # Snapshot day_start_equity before next bar moves
        state.day_start_equity_usd = state.equity_usd

        fills = adapter.execute_decisions(
            timestamp=int(market.timestamps[t + 1]),
            state=state,
            decisions=report.decisions,
            exec_prices=open_prices_next,
            asset_daily_vols=daily_vols,
            proposed_stops=proposed_stops,
        )
        all_fills.extend(fills)

        # 5. Mark to market at bar t+1 CLOSE
        close_prices_next = {sym: float(market.series[sym].close[t + 1]) for sym in market.symbols}
        adapter.mark_to_market(state, close_prices_next, apply_funding=False)
        equity_curve[t + 1] = state.equity_usd

    return (
        equity_curve,
        all_fills,
        risk_reports,
        state.cumulative_funding_usd,
        veto_count,
        scale_count,
    )


def run_backtest(cfg: TalonConfig, market: MarketMatrix) -> BacktestResult:
    """Run full-sample causal backtest for the combined portfolio and each standalone strategy sleeve."""
    strategies = build_default_strategies(cfg.strategies)
    eq_curve, fills, reports, funding_usd, vetoes, scales = _run_portfolio_loop(
        cfg=cfg,
        market=market,
        strategies=strategies,
    )
    dd_curve = compute_drawdown_series(eq_curve)
    portfolio_metrics = compute_metrics(
        equity_curve=eq_curve,
        fills=fills,
        bars_per_year=cfg.bars_per_year,
        total_funding_usd=funding_usd,
        risk_veto_count=vetoes,
        risk_scale_count=scales,
    )

    # Also evaluate each strategy standalone for attribution
    strat_metrics: dict[str, PerformanceMetrics] = {}
    for single_strat in build_default_strategies(cfg.strategies):
        s_eq, s_fills, _, s_fund, s_veto, s_scale = _run_portfolio_loop(
            cfg=cfg,
            market=market,
            strategies=[single_strat],
        )
        strat_metrics[single_strat.strategy_id] = compute_metrics(
            equity_curve=s_eq,
            fills=s_fills,
            bars_per_year=cfg.bars_per_year,
            total_funding_usd=s_fund,
            risk_veto_count=s_veto,
            risk_scale_count=s_scale,
        )

    return BacktestResult(
        timestamps=market.timestamps.copy(),
        equity_curve=eq_curve,
        drawdown_curve=dd_curve,
        fills=fills,
        risk_reports=reports,
        metrics=portfolio_metrics,
        strategy_metrics=strat_metrics,
    )


def run_walk_forward(cfg: TalonConfig, market: MarketMatrix) -> WalkForwardSummary:
    """Run rolling walk-forward evaluation across contiguous Out-of-Sample (OOS) folds.

    To ensure enough warmup bars for indicators in each OOS fold, the OOS evaluation slices
    include trailing warmup context from the end of the training window while scoring strictly
    on the OOS test bars.
    """
    train_len = cfg.walk_forward.train_bars
    test_len = cfg.walk_forward.test_bars
    step_len = cfg.walk_forward.step_bars
    warmup_bars = 125

    n = market.num_bars
    folds: list[WalkForwardFold] = []
    oos_equity_segments: list[NDArray[np.float64]] = []
    oos_fills: list[FillEvent] = []
    oos_funding = 0.0
    oos_vetoes = 0
    oos_scales = 0

    current_oos_equity = cfg.initial_equity_usd
    fold_idx = 0
    start = 0

    while start + train_len + test_len <= n:
        train_end = start + train_len
        test_end = train_end + test_len

        # 1. Run In-Sample (Train) segment
        train_market = market.slice_window(start, train_end)
        is_strats = build_default_strategies(cfg.strategies)
        is_eq, is_fills, _, is_fund, is_v, is_s = _run_portfolio_loop(
            cfg=cfg,
            market=train_market,
            strategies=is_strats,
        )
        is_metrics = compute_metrics(
            equity_curve=is_eq,
            fills=is_fills,
            bars_per_year=cfg.bars_per_year,
            total_funding_usd=is_fund,
            risk_veto_count=is_v,
            risk_scale_count=is_s,
        )

        # 2. Run Out-of-Sample (Test) segment with warmup prefix
        oos_slice_start = max(0, train_end - warmup_bars)
        prefix_len = train_end - oos_slice_start
        test_with_warmup = market.slice_window(oos_slice_start, test_end)
        oos_strats = build_default_strategies(cfg.strategies)
        full_oos_eq, fold_fills, _, fold_fund, fold_v, fold_s = _run_portfolio_loop(
            cfg=cfg,
            market=test_with_warmup,
            strategies=oos_strats,
            initial_equity=current_oos_equity,
        )

        # Extract strictly the Out-of-Sample window [prefix_len:]
        raw_oos_slice = full_oos_eq[prefix_len:]
        if len(raw_oos_slice) > 0 and raw_oos_slice[0] > 0:
            norm_oos_eq = raw_oos_slice * (current_oos_equity / raw_oos_slice[0])
        else:
            norm_oos_eq = raw_oos_slice

        test_start_ts = int(market.timestamps[train_end])
        test_end_ts = int(market.timestamps[test_end - 1])
        strict_oos_fills = [f for f in fold_fills if f.timestamp >= test_start_ts]

        oos_metrics = compute_metrics(
            equity_curve=norm_oos_eq,
            fills=strict_oos_fills,
            bars_per_year=cfg.bars_per_year,
            total_funding_usd=fold_fund * (test_len / float(len(test_with_warmup.timestamps))),
            risk_veto_count=fold_v,
            risk_scale_count=fold_s,
        )

        if len(norm_oos_eq) > 0:
            current_oos_equity = float(norm_oos_eq[-1])
            if not oos_equity_segments:
                oos_equity_segments.append(norm_oos_eq)
            else:
                oos_equity_segments.append(norm_oos_eq[1:])

        oos_fills.extend(strict_oos_fills)
        oos_funding += oos_metrics.total_funding_usd
        oos_vetoes += fold_v
        oos_scales += fold_s

        folds.append(
            WalkForwardFold(
                fold_index=fold_idx,
                train_start_ts=int(market.timestamps[start]),
                train_end_ts=int(market.timestamps[train_end - 1]),
                test_start_ts=test_start_ts,
                test_end_ts=test_end_ts,
                in_sample_metrics=is_metrics,
                out_of_sample_metrics=oos_metrics,
            )
        )
        fold_idx += 1
        start += step_len

    combined_oos_eq = (
        np.concatenate(oos_equity_segments)
        if oos_equity_segments
        else np.array([cfg.initial_equity_usd], dtype=np.float64)
    )
    combined_metrics = compute_metrics(
        equity_curve=combined_oos_eq,
        fills=oos_fills,
        bars_per_year=cfg.bars_per_year,
        total_funding_usd=oos_funding,
        risk_veto_count=oos_vetoes,
        risk_scale_count=oos_scales,
    )

    mean_is = float(np.mean([f.in_sample_metrics.sharpe_ratio for f in folds])) if folds else 0.0
    mean_oos = (
        float(np.mean([f.out_of_sample_metrics.sharpe_ratio for f in folds])) if folds else 0.0
    )
    decay = (mean_oos / mean_is) if abs(mean_is) > 1e-6 else 0.0

    return WalkForwardSummary(
        folds=folds,
        oos_combined_metrics=combined_metrics,
        mean_is_sharpe=mean_is,
        mean_oos_sharpe=mean_oos,
        sharpe_decay_ratio=decay,
    )
