"""Core domain models and data structures for TALON."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class ExecutionMode(StrEnum):
    """Execution mode for the trading worker."""

    PAPER = "paper"
    LIVE = "live"


class RiskActionKind(StrEnum):
    """Outcome of a risk engine evaluation for an order or portfolio."""

    APPROVED = "approved"
    SCALED = "scaled"
    VETOED = "vetoed"
    HALTED_FLATTEN = "halted_flatten"


@dataclass(frozen=True)
class BarSeries:
    """Confirmed historical OHLCV bars for a single instrument."""

    symbol: str
    timestamps: NDArray[np.int64]
    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]
    volume: NDArray[np.float64]

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if not (
            len(self.open) == n
            and len(self.high) == n
            and len(self.low) == n
            and len(self.close) == n
            and len(self.volume) == n
        ):
            raise ValueError(f"Mismatched array lengths in BarSeries for {self.symbol}")
        if n > 1 and not np.all(np.diff(self.timestamps) > 0):
            raise ValueError(f"Timestamps must be strictly increasing for {self.symbol}")

    def slice_up_to(self, end_idx: int) -> BarSeries:
        """Return a causal slice of confirmed bars up to and including end_idx."""
        stop = end_idx + 1
        return BarSeries(
            symbol=self.symbol,
            timestamps=self.timestamps[:stop],
            open=self.open[:stop],
            high=self.high[:stop],
            low=self.low[:stop],
            close=self.close[:stop],
            volume=self.volume[:stop],
        )


@dataclass(frozen=True)
class MarketMatrix:
    """Aligned multi-asset bar matrix across a shared timestamp index."""

    timestamps: NDArray[np.int64]
    symbols: tuple[str, ...]
    series: dict[str, BarSeries]

    @property
    def num_bars(self) -> int:
        return len(self.timestamps)

    def slice_up_to(self, end_idx: int) -> MarketMatrix:
        """Return a causal slice of all assets up to and including end_idx."""
        stop = end_idx + 1
        return MarketMatrix(
            timestamps=self.timestamps[:stop],
            symbols=self.symbols,
            series={sym: s.slice_up_to(end_idx) for sym, s in self.series.items()},
        )

    def slice_window(self, start_idx: int, end_idx: int) -> MarketMatrix:
        """Return a contiguous window [start_idx, end_idx) for walk-forward splits."""
        return MarketMatrix(
            timestamps=self.timestamps[start_idx:end_idx],
            symbols=self.symbols,
            series={
                sym: BarSeries(
                    symbol=sym,
                    timestamps=s.timestamps[start_idx:end_idx],
                    open=s.open[start_idx:end_idx],
                    high=s.high[start_idx:end_idx],
                    low=s.low[start_idx:end_idx],
                    close=s.close[start_idx:end_idx],
                    volume=s.volume[start_idx:end_idx],
                )
                for sym, s in self.series.items()
            },
        )


@dataclass(frozen=True)
class SignalIntent:
    """Target portfolio weight proposal emitted by a strategy for a single asset."""

    strategy_id: str
    symbol: str
    target_weight: float
    raw_score: float
    stop_price: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class OrderDecision:
    """Risk-vetted order instruction emitted by the RiskEngine."""

    symbol: str
    prior_weight: float
    proposed_weight: float
    approved_weight: float
    delta_weight: float
    action: RiskActionKind
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PortfolioRiskReport:
    """Summary of the RiskEngine evaluation for a single bar."""

    timestamp: int
    halted: bool
    halt_reason: str | None
    ex_ante_vol_ann: float
    gross_leverage_proposed: float
    gross_leverage_approved: float
    decisions: dict[str, OrderDecision]


@dataclass(frozen=True)
class FillEvent:
    """Simulated or paper execution fill for a single asset."""

    timestamp: int
    symbol: str
    side: str
    quantity: float
    exec_price: float
    notional_usd: float
    fee_usd: float
    slippage_usd: float
    reasons: tuple[str, ...] = ()


@dataclass
class PortfolioState:
    """Mutable portfolio state tracked by the execution simulator."""

    cash_usd: float
    equity_usd: float
    peak_equity_usd: float
    day_start_equity_usd: float
    positions: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    entry_prices: dict[str, float] = field(default_factory=dict)
    stop_prices: dict[str, float | None] = field(default_factory=dict)
    halted: bool = False
    halt_reason: str | None = None
    cumulative_fees_usd: float = 0.0
    cumulative_slippage_usd: float = 0.0
    cumulative_funding_usd: float = 0.0

    @property
    def current_drawdown_pct(self) -> float:
        if self.peak_equity_usd <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity_usd - self.equity_usd) / self.peak_equity_usd)

    @property
    def daily_loss_pct(self) -> float:
        if self.day_start_equity_usd <= 0.0:
            return 0.0
        return max(0.0, (self.day_start_equity_usd - self.equity_usd) / self.day_start_equity_usd)
