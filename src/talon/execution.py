"""Paper execution adapter and realistic transaction cost / slippage simulator."""

from __future__ import annotations

import numpy as np

from talon.config import LIVE_ACKNOWLEDGEMENT_PHRASE, ExecutionConfig
from talon.types import (
    ExecutionMode,
    FillEvent,
    OrderDecision,
    PortfolioState,
    RiskActionKind,
)


class LiveExecutionDisabledError(RuntimeError):
    """Raised when an operation attempts live execution without explicit dual authorization."""


class PaperExecutionAdapter:
    """Simulates realistic order fills with spread, volatility impact slippage, and fees.

    Strictly operates on simulated state. Any attempt to instantiate or call in LIVE mode
    without explicit authorization raises `LiveExecutionDisabledError`.
    """

    def __init__(
        self,
        cfg: ExecutionConfig,
        mode: ExecutionMode = ExecutionMode.PAPER,
        live_ack: str = "",
        bars_per_year: int = 365,
    ) -> None:
        if mode == ExecutionMode.LIVE and live_ack != LIVE_ACKNOWLEDGEMENT_PHRASE:
            raise LiveExecutionDisabledError(
                "Live execution path is disabled by default. Set TALON_LIVE_TRADING="
                f"{LIVE_ACKNOWLEDGEMENT_PHRASE} and provide a verified live config."
            )
        self._cfg = cfg
        self._mode = mode
        self._bars_per_year = bars_per_year

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    def mark_to_market(
        self,
        state: PortfolioState,
        prices: dict[str, float],
        apply_funding: bool = True,
    ) -> None:
        """Update portfolio equity, funding drag, peak equity, and current weights at `prices`."""
        pos_value = 0.0
        gross_notional = 0.0
        for sym, qty in state.positions.items():
            px = prices.get(sym, 0.0)
            if px > 0 and abs(qty) > 1e-12:
                val = qty * px
                pos_value += val
                gross_notional += abs(val)

        if apply_funding and gross_notional > 0.0 and self._cfg.annual_funding_drag_pct > 0.0:
            daily_funding_rate = self._cfg.annual_funding_drag_pct / float(self._bars_per_year)
            funding_cost = gross_notional * daily_funding_rate
            state.cash_usd -= funding_cost
            state.cumulative_funding_usd += funding_cost

        state.equity_usd = state.cash_usd + pos_value
        if state.equity_usd > state.peak_equity_usd:
            state.peak_equity_usd = state.equity_usd

        if state.equity_usd > 0:
            for sym, qty in state.positions.items():
                px = prices.get(sym, 0.0)
                state.weights[sym] = (qty * px) / state.equity_usd if px > 0 else 0.0
        else:
            for sym in state.positions:
                state.weights[sym] = 0.0

    def execute_decisions(
        self,
        timestamp: int,
        state: PortfolioState,
        decisions: dict[str, OrderDecision],
        exec_prices: dict[str, float],
        asset_daily_vols: dict[str, float],
        proposed_stops: dict[str, float | None],
    ) -> list[FillEvent]:
        """Execute approved weight transitions at `exec_prices` (bar t+1 open) with slippage & fees."""
        fills: list[FillEvent] = []
        if state.equity_usd <= 0:
            return fills

        for sym, dec in decisions.items():
            if dec.action == RiskActionKind.VETOED and abs(dec.delta_weight) <= 1e-9:
                continue

            mid_px = exec_prices.get(sym, 0.0)
            if mid_px <= 0 or not np.isfinite(mid_px):
                continue

            target_notional = dec.approved_weight * state.equity_usd
            current_qty = state.positions.get(sym, 0.0)
            current_notional = current_qty * mid_px
            delta_notional = target_notional - current_notional

            if abs(delta_notional) < 1.0:
                # Below $1 notional rounding threshold
                continue

            side = "buy" if delta_notional > 0 else "sell"
            direction = 1.0 if side == "buy" else -1.0

            # Total one-way slippage = half_spread + impact_coeff * daily_vol * |delta_weight|
            half_spread_frac = self._cfg.half_spread_bps / 10_000.0
            daily_vol = max(0.005, asset_daily_vols.get(sym, 0.02))
            impact_frac = self._cfg.impact_slippage_coeff * daily_vol * abs(dec.delta_weight)
            slip_frac = half_spread_frac + impact_frac

            fill_px = mid_px * (1.0 + direction * slip_frac)
            trade_qty = delta_notional / fill_px
            notional_usd = abs(trade_qty) * fill_px
            slippage_usd = abs(trade_qty) * abs(fill_px - mid_px)
            fee_usd = notional_usd * (self._cfg.taker_fee_bps / 10_000.0)

            # Update cash & positions:
            # In a unified margin / cash-settled representation, buying deducts trade_qty * fill_px + fee_usd
            state.cash_usd -= trade_qty * fill_px + fee_usd
            new_qty = current_qty + trade_qty
            if abs(dec.approved_weight) <= 1e-9 or abs(new_qty * mid_px) < 1e-3:
                state.positions[sym] = 0.0
                state.entry_prices.pop(sym, None)
                state.stop_prices[sym] = None
            else:
                state.positions[sym] = new_qty
                state.entry_prices[sym] = fill_px
                if proposed_stops.get(sym) is not None:
                    state.stop_prices[sym] = proposed_stops[sym]

            state.cumulative_fees_usd += fee_usd
            state.cumulative_slippage_usd += slippage_usd

            fills.append(
                FillEvent(
                    timestamp=timestamp,
                    symbol=sym,
                    side=side,
                    quantity=float(trade_qty),
                    exec_price=float(fill_px),
                    notional_usd=float(notional_usd),
                    fee_usd=float(fee_usd),
                    slippage_usd=float(slippage_usd),
                    reasons=dec.reasons,
                )
            )

        # Re-mark at mid execution prices after fills
        self.mark_to_market(state, exec_prices, apply_funding=False)
        return fills
