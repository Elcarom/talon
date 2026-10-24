"""Portfolio Risk Engine with absolute veto and scaling authority."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from talon.config import RiskConfig
from talon.types import (
    MarketMatrix,
    OrderDecision,
    PortfolioRiskReport,
    PortfolioState,
    RiskActionKind,
)


class RiskEngine:
    """Evaluates proposed target weights against portfolio-wide risk limits.

    Order of precedence:
    1. Max Drawdown Kill Switch (permanent portfolio halt + immediate flatten)
    2. Daily / Single-Bar Loss Circuit Breaker (flatten + veto new opens)
    3. Market Data Sanity Check (non-positive / NaN prices vetoed)
    4. Per-Asset Stop-Loss Breach Check (force close to 0.0 if stop breached)
    5. Per-Asset Notional Weight Cap
    6. Pairwise Return Correlation Cluster Cap
    7. Ex-Ante Portfolio Volatility Target Scaling
    8. Maximum Gross Leverage Scaling
    9. Rebalance Turnover Deadband (veto tiny weight adjustments to prevent fee churn)
    """

    def __init__(self, cfg: RiskConfig, bars_per_year: int = 365) -> None:
        self._cfg = cfg
        self._bars_per_year = bars_per_year

    def _compute_cov_and_corr(
        self,
        market: MarketMatrix,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        lb = self._cfg.vol_lookback
        if market.num_bars < lb + 1:
            return None

        n_sym = len(market.symbols)
        ret_matrix = np.zeros((lb, n_sym), dtype=np.float64)
        for j, sym in enumerate(market.symbols):
            closes = market.series[sym].close[-(lb + 1) :]
            if np.any(closes <= 0) or not np.all(np.isfinite(closes)):
                return None
            ret_matrix[:, j] = np.diff(np.log(closes))

        cov_daily = np.cov(ret_matrix, rowvar=False, ddof=1)
        if cov_daily.ndim == 0:
            cov_daily = np.array([[float(cov_daily)]], dtype=np.float64)
        cov_ann = cov_daily * float(self._bars_per_year)

        stds = np.sqrt(np.maximum(1e-12, np.diag(cov_daily)))
        outer_stds = np.outer(stds, stds)
        corr = np.clip(cov_daily / outer_stds, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        return cov_ann, corr

    def evaluate(
        self,
        timestamp: int,
        state: PortfolioState,
        proposed_weights: dict[str, float],
        market: MarketMatrix,
    ) -> PortfolioRiskReport:
        """Evaluate proposed portfolio weights and emit vetted OrderDecisions."""
        symbols = market.symbols
        gross_proposed = float(sum(abs(proposed_weights.get(s, 0.0)) for s in symbols))

        # 1. Check permanent Max Drawdown Kill Switch
        if state.halted or state.current_drawdown_pct >= self._cfg.max_drawdown_pct:
            reason = state.halt_reason or (
                f"MAX_DRAWDOWN_BREACH ({state.current_drawdown_pct * 100:.2f}% >= "
                f"{self._cfg.max_drawdown_pct * 100:.2f}%)"
            )
            state.halted = True
            state.halt_reason = reason
            decisions: dict[str, OrderDecision] = {}
            for sym in symbols:
                prior_w = state.weights.get(sym, 0.0)
                prop_w = proposed_weights.get(sym, 0.0)
                action = (
                    RiskActionKind.HALTED_FLATTEN
                    if abs(prior_w) > 1e-9
                    else RiskActionKind.VETOED
                )
                decisions[sym] = OrderDecision(
                    symbol=sym,
                    prior_weight=prior_w,
                    proposed_weight=prop_w,
                    approved_weight=0.0,
                    delta_weight=-prior_w,
                    action=action,
                    reasons=(reason,),
                )
            return PortfolioRiskReport(
                timestamp=timestamp,
                halted=True,
                halt_reason=reason,
                ex_ante_vol_ann=0.0,
                gross_leverage_proposed=gross_proposed,
                gross_leverage_approved=0.0,
                decisions=decisions,
            )

        # 2. Check Daily Loss Circuit Breaker
        if state.daily_loss_pct >= self._cfg.daily_loss_limit_pct:
            reason = (
                f"DAILY_LOSS_LIMIT_BREACH ({state.daily_loss_pct * 100:.2f}% >= "
                f"{self._cfg.daily_loss_limit_pct * 100:.2f}%)"
            )
            decisions = {}
            for sym in symbols:
                prior_w = state.weights.get(sym, 0.0)
                prop_w = proposed_weights.get(sym, 0.0)
                action = (
                    RiskActionKind.HALTED_FLATTEN
                    if abs(prior_w) > 1e-9
                    else RiskActionKind.VETOED
                )
                decisions[sym] = OrderDecision(
                    symbol=sym,
                    prior_weight=prior_w,
                    proposed_weight=prop_w,
                    approved_weight=0.0,
                    delta_weight=-prior_w,
                    action=action,
                    reasons=(reason,),
                )
            return PortfolioRiskReport(
                timestamp=timestamp,
                halted=False,
                halt_reason=reason,
                ex_ante_vol_ann=0.0,
                gross_leverage_proposed=gross_proposed,
                gross_leverage_approved=0.0,
                decisions=decisions,
            )

        working_w: dict[str, float] = {}
        reasons_map: dict[str, list[str]] = {sym: [] for sym in symbols}
        vetoed_symbols: set[str] = set()

        # 3 & 4. Data sanity check + Per-asset stop-loss check + Per-asset cap
        for sym in symbols:
            series = market.series.get(sym)
            prior_w = state.weights.get(sym, 0.0)
            prop_w = proposed_weights.get(sym, 0.0)

            if (
                series is None
                or len(series.close) == 0
                or not np.isfinite(series.close[-1])
                or series.close[-1] <= 0
                or not np.isfinite(prop_w)
            ):
                working_w[sym] = prior_w
                reasons_map[sym].append("VETOED_INVALID_MARKET_DATA")
                vetoed_symbols.add(sym)
                continue

            last_px = float(series.close[-1])
            stop_px = state.stop_prices.get(sym)
            if (
                stop_px is not None
                and abs(prior_w) > 1e-9
                and ((prior_w > 0 and last_px <= stop_px) or (prior_w < 0 and last_px >= stop_px))
            ):
                working_w[sym] = 0.0
                reasons_map[sym].append(f"FORCED_STOP_EXIT (px={last_px:.2f} vs stop={stop_px:.2f})")
                continue

            # 5. Per-Asset Exposure Cap
            if abs(prop_w) > self._cfg.max_asset_weight + 1e-9:
                capped = float(np.sign(prop_w) * self._cfg.max_asset_weight)
                working_w[sym] = capped
                reasons_map[sym].append(
                    f"SCALED_ASSET_CAP ({prop_w:+.3f} -> {capped:+.3f})"
                )
            else:
                working_w[sym] = float(prop_w)

        # 6. Pairwise Correlation Limit & 7. Portfolio Volatility Target
        cov_corr = self._compute_cov_and_corr(market)
        ex_ante_vol = 0.0

        if cov_corr is not None:
            cov_ann, corr = cov_corr
            n_sym = len(symbols)

            # Check pairwise correlation between same-direction positions
            for i in range(n_sym):
                for j in range(i + 1, n_sym):
                    s_i, s_j = symbols[i], symbols[j]
                    w_i, w_j = working_w[s_i], working_w[s_j]
                    if w_i * w_j > 0 and corr[i, j] > self._cfg.max_pairwise_correlation:
                        cluster_gross = abs(w_i) + abs(w_j)
                        if cluster_gross > self._cfg.correlated_cluster_cap + 1e-9:
                            scale = self._cfg.correlated_cluster_cap / cluster_gross
                            working_w[s_i] *= scale
                            working_w[s_j] *= scale
                            msg = (
                                f"SCALED_CORRELATION_CLUSTER ({s_i}/{s_j} "
                                f"corr={corr[i, j]:.2f}>{self._cfg.max_pairwise_correlation:.2f})"
                            )
                            reasons_map[s_i].append(msg)
                            reasons_map[s_j].append(msg)

            # Compute ex-ante portfolio annualized volatility
            w_vec = np.array([working_w[s] for s in symbols], dtype=np.float64)
            port_var = float(w_vec.T @ cov_ann @ w_vec)
            ex_ante_vol = float(np.sqrt(max(0.0, port_var)))

            if ex_ante_vol > self._cfg.portfolio_vol_target + 1e-9:
                vol_scale = self._cfg.portfolio_vol_target / ex_ante_vol
                for sym in symbols:
                    if abs(working_w[sym]) > 1e-9 and sym not in vetoed_symbols:
                        working_w[sym] *= vol_scale
                        reasons_map[sym].append(
                            f"SCALED_PORTFOLIO_VOL (vol={ex_ante_vol:.2f}>{self._cfg.portfolio_vol_target:.2f})"
                        )
                ex_ante_vol = self._cfg.portfolio_vol_target

        # 8. Maximum Gross Leverage Ceiling
        current_gross = sum(abs(working_w[s]) for s in symbols)
        if current_gross > self._cfg.max_gross_leverage + 1e-9:
            lev_scale = self._cfg.max_gross_leverage / current_gross
            for sym in symbols:
                if abs(working_w[sym]) > 1e-9 and sym not in vetoed_symbols:
                    working_w[sym] *= lev_scale
                    reasons_map[sym].append(
                        f"SCALED_GROSS_LEVERAGE (gross={current_gross:.2f}>{self._cfg.max_gross_leverage:.2f})"
                    )

        # 9. Rebalance Deadband (suppress fee churn on tiny adjustments unless closing to 0)
        decisions = {}
        for sym in symbols:
            prior_w = state.weights.get(sym, 0.0)
            prop_w = proposed_weights.get(sym, 0.0)
            target_w = working_w[sym]
            delta = target_w - prior_w

            is_full_close = abs(target_w) <= 1e-9 and abs(prior_w) > 1e-9
            if (
                sym not in vetoed_symbols
                and not is_full_close
                and 1e-9 < abs(delta) < self._cfg.rebalance_deadband
            ):
                target_w = prior_w
                delta = 0.0
                reasons_map[sym].append(
                    f"VETOED_REBALANCE_DEADBAND (|delta|<{self._cfg.rebalance_deadband:.3f})"
                )
                action = RiskActionKind.VETOED
            elif sym in vetoed_symbols:
                action = RiskActionKind.VETOED
            elif any(r.startswith("SCALED_") or r.startswith("FORCED_") for r in reasons_map[sym]):
                action = RiskActionKind.SCALED
            else:
                action = RiskActionKind.APPROVED

            decisions[sym] = OrderDecision(
                symbol=sym,
                prior_weight=prior_w,
                proposed_weight=prop_w,
                approved_weight=target_w,
                delta_weight=delta,
                action=action,
                reasons=tuple(reasons_map[sym]),
            )

        approved_gross = float(sum(abs(d.approved_weight) for d in decisions.values()))
        return PortfolioRiskReport(
            timestamp=timestamp,
            halted=False,
            halt_reason=None,
            ex_ante_vol_ann=ex_ante_vol,
            gross_leverage_proposed=gross_proposed,
            gross_leverage_approved=approved_gross,
            decisions=decisions,
        )
