"""Causal technical and statistical indicators operating strictly on confirmed bars."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from talon.types import BarSeries


def realized_volatility_ann(
    closes: NDArray[np.float64],
    lookback: int = 30,
    bars_per_year: int = 365,
) -> float | None:
    """Compute annualized realized volatility of log returns over the trailing lookback."""
    if len(closes) < lookback + 1:
        return None
    window = closes[-(lookback + 1) :]
    if np.any(window <= 0) or not np.all(np.isfinite(window)):
        return None
    log_rets = np.diff(np.log(window))
    if len(log_rets) < 2:
        return None
    std = float(np.std(log_rets, ddof=1))
    return float(std * np.sqrt(bars_per_year))


def wilder_atr(series: BarSeries, period: int = 14) -> float | None:
    """Compute Wilder's Average True Range (ATR) on confirmed bars."""
    n = len(series.close)
    if n < period + 1:
        return None
    highs = series.high[1:]
    lows = series.low[1:]
    prev_closes = series.close[:-1]
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)),
    )
    atr_val = float(np.mean(tr[:period]))
    alpha = 1.0 / period
    for val in tr[period:]:
        atr_val = atr_val * (1.0 - alpha) + float(val) * alpha
    return atr_val


def multi_horizon_donchian_score(
    closes: NDArray[np.float64],
    lookbacks: tuple[int, ...] = (10, 20, 40, 80, 120),
) -> tuple[float, float | None] | None:
    """Compute normalized Donchian ensemble trend score in [-1, +1] and trailing stop midpoint.

    Requires full history for the longest lookback so no slice is silently dropped.
    """
    max_lb = max(lookbacks)
    if len(closes) < max_lb + 1:
        return None

    active_states: list[int] = []
    active_stops: list[float] = []

    for lb in lookbacks:
        state = 0
        stop = 0.0
        # Replay slice over trailing 2*lb bars (or full array) to establish ratcheted midpoint state
        start_idx = max(lb, len(closes) - 2 * lb)
        for i in range(start_idx, len(closes)):
            prior = closes[i - lb : i]
            hi = float(np.max(prior))
            lo = float(np.min(prior))
            mid = 0.5 * (hi + lo)
            c = float(closes[i])

            if state == 1:
                stop = max(stop, mid)
                if c < stop:
                    state = 0
            elif state == -1:
                stop = min(stop, mid)
                if c > stop:
                    state = 0

            if state != 1 and c > hi:
                state = 1
                stop = mid
            elif state != -1 and c < lo:
                state = -1
                stop = mid

        active_states.append(state)
        if state != 0:
            active_stops.append(stop)

    score = float(sum(active_states)) / float(len(lookbacks))
    avg_stop = float(np.mean(active_stops)) if active_stops else None
    return score, avg_stop


def ols_cointegration_spread(
    closes_a: NDArray[np.float64],
    closes_b: NDArray[np.float64],
    ols_window: int = 60,
    z_window: int = 20,
) -> tuple[float, float, float] | None:
    """Compute log-price OLS hedge ratio beta, spread z-score, and OU mean-reversion half-life.

    Returns (beta, z_score, half_life_bars) or None if insufficient data or non-stationary.
    """
    min_len = max(ols_window, z_window) + 2
    if len(closes_a) < min_len or len(closes_b) < min_len:
        return None

    ya = closes_a[-ols_window:]
    xb = closes_b[-ols_window:]
    if np.any(ya <= 0) or np.any(xb <= 0):
        return None

    log_a = np.log(ya)
    log_b = np.log(xb)

    var_b = float(np.var(log_b, ddof=1))
    if var_b <= 1e-12:
        return None
    cov_ab = float(np.cov(log_a, log_b, ddof=1)[0, 1])
    beta = cov_ab / var_b
    alpha = float(np.mean(log_a) - beta * np.mean(log_b))

    spread = log_a - (alpha + beta * log_b)
    recent_spread = spread[-z_window:]
    spread_std = float(np.std(recent_spread, ddof=1))
    if spread_std <= 1e-10:
        return None
    z_score = float((spread[-1] - np.mean(recent_spread)) / spread_std)

    # Estimate Ornstein-Uhlenbeck half-life: delta_s_t = c + phi * s_{t-1} + eps_t
    s_lag = spread[:-1]
    ds = np.diff(spread)
    var_lag = float(np.var(s_lag, ddof=1))
    if var_lag <= 1e-12:
        return None
    phi = float(np.cov(ds, s_lag, ddof=1)[0, 1]) / var_lag

    # Mean reversion requires phi < 0 and 1 + phi > 0
    if phi >= -1e-6 or (1.0 + phi) <= 1e-6:
        half_life = float("inf")
    else:
        half_life = float(-np.log(2.0) / np.log(1.0 + phi))

    return beta, z_score, half_life
