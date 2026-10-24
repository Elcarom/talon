"""Performance and risk-adjusted return metrics for backtesting and walk-forward evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from talon.types import FillEvent


@dataclass(frozen=True)
class PerformanceMetrics:
    """Comprehensive performance and risk metrics for a backtest run."""

    initial_equity_usd: float
    final_equity_usd: float
    total_return_pct: float
    cagr_pct: float
    annualized_vol_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    annualized_turnover: float
    hit_rate_pct: float
    profit_factor: float
    total_trades: int
    total_fees_usd: float
    total_slippage_usd: float
    total_funding_usd: float
    risk_veto_count: int
    risk_scale_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_drawdown_series(equity_curve: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute fractional drawdown series (in [0, 1]) from an equity curve."""
    if len(equity_curve) == 0:
        return np.array([], dtype=np.float64)
    peaks = np.maximum.accumulate(equity_curve)
    safe_peaks = np.where(peaks > 0, peaks, 1.0)
    dd = np.maximum(0.0, (safe_peaks - equity_curve) / safe_peaks)
    return dd


def compute_metrics(
    equity_curve: NDArray[np.float64],
    fills: list[FillEvent],
    bars_per_year: int = 365,
    total_funding_usd: float = 0.0,
    risk_veto_count: int = 0,
    risk_scale_count: int = 0,
) -> PerformanceMetrics:
    """Calculate standard institutional quantitative performance metrics."""
    if len(equity_curve) < 2:
        init_eq = float(equity_curve[0]) if len(equity_curve) == 1 else 0.0
        return PerformanceMetrics(
            initial_equity_usd=init_eq,
            final_equity_usd=init_eq,
            total_return_pct=0.0,
            cagr_pct=0.0,
            annualized_vol_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            calmar_ratio=0.0,
            annualized_turnover=0.0,
            hit_rate_pct=0.0,
            profit_factor=0.0,
            total_trades=len(fills),
            total_fees_usd=sum(f.fee_usd for f in fills),
            total_slippage_usd=sum(f.slippage_usd for f in fills),
            total_funding_usd=total_funding_usd,
            risk_veto_count=risk_veto_count,
            risk_scale_count=risk_scale_count,
        )

    init_eq = float(equity_curve[0])
    final_eq = float(equity_curve[-1])
    total_ret = (final_eq / init_eq) - 1.0 if init_eq > 0 else 0.0

    n_bars = len(equity_curve) - 1
    years = max(1e-6, float(n_bars) / float(bars_per_year))
    cagr = (final_eq / init_eq) ** (1.0 / years) - 1.0 if (init_eq > 0 and final_eq > 0) else -1.0

    # Bar-over-bar simple returns
    safe_prev = np.where(equity_curve[:-1] > 0, equity_curve[:-1], 1.0)
    rets = np.diff(equity_curve) / safe_prev

    mean_ret = float(np.mean(rets))
    std_ret = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    ann_vol = std_ret * float(np.sqrt(bars_per_year))

    sharpe = (mean_ret / std_ret) * float(np.sqrt(bars_per_year)) if std_ret > 1e-12 else 0.0

    downside = np.minimum(0.0, rets)
    downside_dev = float(np.sqrt(np.mean(downside**2)))
    sortino = (
        (mean_ret / downside_dev) * float(np.sqrt(bars_per_year))
        if downside_dev > 1e-12
        else 0.0
    )

    dd_series = compute_drawdown_series(equity_curve)
    max_dd = float(np.max(dd_series)) if len(dd_series) > 0 else 0.0
    calmar = (cagr / max_dd) if max_dd > 1e-9 else 0.0

    avg_equity = float(np.mean(equity_curve))
    total_traded_notional = float(sum(f.notional_usd for f in fills))
    ann_turnover = (
        (total_traded_notional / avg_equity) / years if avg_equity > 0 else 0.0
    )

    active_rets = rets[np.abs(rets) > 1e-9]
    if len(active_rets) > 0:
        wins = active_rets[active_rets > 0]
        losses = active_rets[active_rets < 0]
        hit_rate = float(len(wins)) / float(len(active_rets))
        gross_gains = float(np.sum(wins))
        gross_losses = float(np.abs(np.sum(losses)))
        profit_factor = (gross_gains / gross_losses) if gross_losses > 1e-12 else 0.0
    else:
        hit_rate = 0.0
        profit_factor = 0.0

    return PerformanceMetrics(
        initial_equity_usd=round(init_eq, 2),
        final_equity_usd=round(final_eq, 2),
        total_return_pct=round(total_ret * 100.0, 2),
        cagr_pct=round(cagr * 100.0, 2),
        annualized_vol_pct=round(ann_vol * 100.0, 2),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        max_drawdown_pct=round(max_dd * 100.0, 2),
        calmar_ratio=round(calmar, 3),
        annualized_turnover=round(ann_turnover, 2),
        hit_rate_pct=round(hit_rate * 100.0, 2),
        profit_factor=round(profit_factor, 3),
        total_trades=len(fills),
        total_fees_usd=round(float(sum(f.fee_usd for f in fills)), 2),
        total_slippage_usd=round(float(sum(f.slippage_usd for f in fills)), 2),
        total_funding_usd=round(total_funding_usd, 2),
        risk_veto_count=risk_veto_count,
        risk_scale_count=risk_scale_count,
    )
