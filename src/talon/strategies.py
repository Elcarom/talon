"""Deterministic quantitative trading strategies for TALON."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from talon.config import (
    CrossSectionalConfig,
    MeanReversionConfig,
    StrategiesConfig,
    TrendStrategyConfig,
)
from talon.indicators import (
    multi_horizon_donchian_score,
    ols_cointegration_spread,
    realized_volatility_ann,
    wilder_atr,
)
from talon.types import MarketMatrix, SignalIntent


class Strategy(ABC):
    """Abstract base interface for all TALON quantitative strategies."""

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Unique identifier for the strategy."""

    @property
    @abstractmethod
    def weight_allocation(self) -> float:
        """Portfolio sleeve weight allocation in [0, 1]."""

    @abstractmethod
    def generate_signals(
        self,
        market: MarketMatrix,
        bars_per_year: int = 365,
    ) -> list[SignalIntent]:
        """Compute causal target weights from a confirmed market matrix up to bar t."""


class VolTargetedTrendStrategy(Strategy):
    """Strategy 1: Multi-horizon Donchian ensemble trend following with volatility targeting."""

    def __init__(self, cfg: TrendStrategyConfig) -> None:
        self._cfg = cfg

    @property
    def strategy_id(self) -> str:
        return "vol_targeted_trend"

    @property
    def weight_allocation(self) -> float:
        return self._cfg.weight_allocation

    def generate_signals(
        self,
        market: MarketMatrix,
        bars_per_year: int = 365,
    ) -> list[SignalIntent]:
        if not self._cfg.enabled:
            return []

        intents: list[SignalIntent] = []
        n_assets = max(1, len(market.symbols))

        for sym in market.symbols:
            series = market.series[sym]
            donchian = multi_horizon_donchian_score(series.close)
            rv = realized_volatility_ann(series.close, lookback=30, bars_per_year=bars_per_year)
            atr_val = wilder_atr(series, period=14)

            if donchian is None or rv is None or rv <= 1e-6:
                intents.append(
                    SignalIntent(
                        strategy_id=self.strategy_id,
                        symbol=sym,
                        target_weight=0.0,
                        raw_score=0.0,
                        reason="warmup_insufficient_bars",
                    )
                )
                continue

            score, channel_stop = donchian
            if abs(score) < self._cfg.activation_threshold:
                intents.append(
                    SignalIntent(
                        strategy_id=self.strategy_id,
                        symbol=sym,
                        target_weight=0.0,
                        raw_score=score,
                        reason=f"chop_filter |score|={abs(score):.2f}<{self._cfg.activation_threshold:.2f}",
                    )
                )
                continue

            # Volatility-targeted sizing scaled across universe assets
            vol_scalar = self._cfg.vol_target_ann / rv
            raw_weight = (score * vol_scalar) / float(np.sqrt(n_assets))
            clipped_weight = float(
                np.clip(raw_weight, -self._cfg.max_asset_weight, self._cfg.max_asset_weight)
            )

            last_px = float(series.close[-1])
            stop_px: float | None = channel_stop
            if atr_val is not None:
                atr_stop = (
                    last_px - self._cfg.atr_stop_mult * atr_val
                    if score > 0
                    else last_px + self._cfg.atr_stop_mult * atr_val
                )
                if stop_px is not None:
                    stop_px = max(stop_px, atr_stop) if score > 0 else min(stop_px, atr_stop)
                else:
                    stop_px = atr_stop

            intents.append(
                SignalIntent(
                    strategy_id=self.strategy_id,
                    symbol=sym,
                    target_weight=clipped_weight,
                    raw_score=score,
                    stop_price=stop_px,
                    reason=f"trend_score={score:+.2f} rv={rv:.2f}",
                )
            )

        return intents


class CrossSectionalMomentumStrategy(Strategy):
    """Strategy 2: Volatility-adjusted cross-sectional relative strength (long leaders, short laggards)."""

    def __init__(self, cfg: CrossSectionalConfig) -> None:
        self._cfg = cfg

    @property
    def strategy_id(self) -> str:
        return "cross_sectional_momentum"

    @property
    def weight_allocation(self) -> float:
        return self._cfg.weight_allocation

    def generate_signals(
        self,
        market: MarketMatrix,
        bars_per_year: int = 365,
    ) -> list[SignalIntent]:
        if not self._cfg.enabled:
            return []

        min_len = max(self._cfg.fast_lookback, self._cfg.slow_lookback, 30) + 2
        if market.num_bars < min_len:
            return [
                SignalIntent(
                    strategy_id=self.strategy_id,
                    symbol=sym,
                    target_weight=0.0,
                    raw_score=0.0,
                    reason="warmup_insufficient_bars",
                )
                for sym in market.symbols
            ]

        ir_scores: dict[str, float] = {}
        vols: dict[str, float] = {}

        for sym in market.symbols:
            closes = market.series[sym].close
            rv = realized_volatility_ann(closes, lookback=30, bars_per_year=bars_per_year)
            if rv is None or rv <= 1e-6:
                continue
            # Use closes[-2] vs lookback to skip 1-bar short-term reversal noise
            c_ref = float(closes[-2])
            c_fast = float(closes[-2 - self._cfg.fast_lookback])
            c_slow = float(closes[-2 - self._cfg.slow_lookback])
            if c_fast <= 0 or c_slow <= 0:
                continue
            ret_fast = (c_ref / c_fast) - 1.0
            ret_slow = (c_ref / c_slow) - 1.0
            blended_ret = 0.5 * ret_fast + 0.5 * ret_slow
            ir_scores[sym] = blended_ret / rv
            vols[sym] = rv

        if len(ir_scores) < 2:
            return []

        vals = np.array(list(ir_scores.values()), dtype=np.float64)
        mean_ir = float(np.mean(vals))
        std_ir = float(np.std(vals, ddof=1))
        if std_ir <= 1e-9:
            return [
                SignalIntent(
                    strategy_id=self.strategy_id,
                    symbol=sym,
                    target_weight=0.0,
                    raw_score=0.0,
                    reason="zero_cross_sectional_dispersion",
                )
                for sym in market.symbols
            ]

        z_scores: dict[str, float] = {sym: (ir - mean_ir) / std_ir for sym, ir in ir_scores.items()}
        raw_weights: dict[str, float] = {}
        for sym in market.symbols:
            z = z_scores.get(sym, 0.0)
            if abs(z) >= self._cfg.z_entry_threshold:
                raw_weights[sym] = z / max(0.15, vols.get(sym, 0.30))
            else:
                raw_weights[sym] = 0.0

        gross = sum(abs(w) for w in raw_weights.values())
        scale = (self._cfg.max_gross_weight / gross) if gross > 1e-9 else 0.0

        intents: list[SignalIntent] = []
        for sym in market.symbols:
            z = z_scores.get(sym, 0.0)
            w = raw_weights.get(sym, 0.0) * scale
            intents.append(
                SignalIntent(
                    strategy_id=self.strategy_id,
                    symbol=sym,
                    target_weight=float(w),
                    raw_score=float(z),
                    reason=f"xsec_z={z:+.2f}",
                )
            )
        return intents


class CointegratedMeanReversionStrategy(Strategy):
    """Strategy 3: Ornstein-Uhlenbeck residual spread mean reversion on cointegrated pair."""

    def __init__(self, cfg: MeanReversionConfig) -> None:
        self._cfg = cfg
        self._active_dir: int = 0  # +1 = long spread (long A, short B), -1 = short spread

    @property
    def strategy_id(self) -> str:
        return "cointegrated_mean_reversion"

    @property
    def weight_allocation(self) -> float:
        return self._cfg.weight_allocation

    def generate_signals(
        self,
        market: MarketMatrix,
        bars_per_year: int = 365,
    ) -> list[SignalIntent]:
        if not self._cfg.enabled:
            return []

        leg_a = self._cfg.pair_leg_a
        leg_b = self._cfg.pair_leg_b
        if leg_a not in market.series or leg_b not in market.series:
            return []

        res = ols_cointegration_spread(
            market.series[leg_a].close,
            market.series[leg_b].close,
            ols_window=self._cfg.ols_window,
            z_window=self._cfg.z_window,
        )
        if res is None:
            return [
                SignalIntent(self.strategy_id, leg_a, 0.0, 0.0, reason="warmup_insufficient_bars"),
                SignalIntent(self.strategy_id, leg_b, 0.0, 0.0, reason="warmup_insufficient_bars"),
            ]

        beta, z_score, half_life = res

        # Stationary regime filter: bail out if half-life breaks outside tradable range or stop_z breached
        stationary = self._cfg.min_half_life <= half_life <= self._cfg.max_half_life
        if not stationary or abs(z_score) >= self._cfg.stop_z:
            self._active_dir = 0
            reason = (
                f"regime_break hl={half_life:.1f}"
                if not stationary
                else f"stop_z_breach z={z_score:+.2f}"
            )
            return [
                SignalIntent(self.strategy_id, leg_a, 0.0, z_score, reason=reason),
                SignalIntent(self.strategy_id, leg_b, 0.0, -z_score, reason=reason),
            ]

        # Check exit / entry transitions
        if self._active_dir == 1 and z_score >= -self._cfg.exit_z or self._active_dir == -1 and z_score <= self._cfg.exit_z:
            self._active_dir = 0

        if self._active_dir == 0:
            if z_score <= -self._cfg.entry_z:
                self._active_dir = 1  # Spread oversold: Long Leg A, Short Leg B
            elif z_score >= self._cfg.entry_z:
                self._active_dir = -1  # Spread overbought: Short Leg A, Long Leg B

        if self._active_dir == 0:
            return [
                SignalIntent(
                    self.strategy_id,
                    leg_a,
                    0.0,
                    z_score,
                    reason=f"flat z={z_score:+.2f} hl={half_life:.1f}",
                ),
                SignalIntent(
                    self.strategy_id,
                    leg_b,
                    0.0,
                    -z_score,
                    reason=f"flat z={z_score:+.2f} hl={half_life:.1f}",
                ),
            ]

        # Scale leg weights respecting beta hedge ratio and max_leg_weight
        clipped_beta = float(np.clip(beta, 0.25, 2.50))
        norm = max(1.0, clipped_beta)
        w_a = float(self._active_dir) * (self._cfg.max_leg_weight / norm)
        w_b = -float(self._active_dir) * (self._cfg.max_leg_weight * clipped_beta / norm)

        desc = f"ou_spread z={z_score:+.2f} beta={clipped_beta:.2f} hl={half_life:.1f}"
        return [
            SignalIntent(self.strategy_id, leg_a, w_a, z_score, reason=desc),
            SignalIntent(self.strategy_id, leg_b, w_b, -z_score, reason=desc),
        ]


def build_default_strategies(cfg: StrategiesConfig) -> list[Strategy]:
    """Instantiate all enabled strategies from configuration."""
    strategies: list[Strategy] = []
    if cfg.vol_targeted_trend.enabled:
        strategies.append(VolTargetedTrendStrategy(cfg.vol_targeted_trend))
    if cfg.cross_sectional_momentum.enabled:
        strategies.append(CrossSectionalMomentumStrategy(cfg.cross_sectional_momentum))
    if cfg.cointegrated_mean_reversion.enabled:
        strategies.append(CointegratedMeanReversionStrategy(cfg.cointegrated_mean_reversion))
    return strategies


def combine_strategy_intents(
    symbols: tuple[str, ...],
    strategies: list[Strategy],
    intents_by_strategy: dict[str, list[SignalIntent]],
) -> tuple[dict[str, float], dict[str, float | None]]:
    """Aggregate weighted strategy sleeve proposals into net portfolio target weights."""
    combined_weights: dict[str, float] = {sym: 0.0 for sym in symbols}
    stop_prices: dict[str, float | None] = {sym: None for sym in symbols}

    total_alloc = sum(s.weight_allocation for s in strategies)
    norm_factor = (1.0 / total_alloc) if total_alloc > 1e-9 else 1.0

    for strat in strategies:
        alloc = strat.weight_allocation * norm_factor
        for intent in intents_by_strategy.get(strat.strategy_id, []):
            if intent.symbol in combined_weights:
                combined_weights[intent.symbol] += alloc * intent.target_weight
                if intent.stop_price is not None and stop_prices[intent.symbol] is None:
                    stop_prices[intent.symbol] = intent.stop_price

    return combined_weights, stop_prices
