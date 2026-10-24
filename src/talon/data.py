"""Data ingestion layer supporting deterministic synthetic regimes, yfinance, and CCXT."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from talon.config import DataConfig
from talon.types import BarSeries, MarketMatrix

DAY_SECONDS = 86_400
BASE_TIMESTAMP = 1_704_067_200  # 2024-01-01T00:00:00Z


def generate_synthetic_market(cfg: DataConfig) -> MarketMatrix:
    """Generate a realistic, deterministic multi-asset market matrix with distinct regimes.

    Models:
    - Common macro trend factor with regime switches (bull trend, correction, range-bound chop, recovery)
    - Cross-sectional momentum dispersion across secondary assets
    - Cointegrated Ornstein-Uhlenbeck residual spread between the first two symbols (e.g., ETH-USD and BTC-USD)
    - Realistic intrabar High/Low ranges and volume spikes
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.num_bars
    symbols = cfg.symbols
    timestamps = np.array([BASE_TIMESTAMP + i * DAY_SECONDS for i in range(n)], dtype=np.int64)

    # 1. Construct common factor daily log returns across 4 regimes
    factor_rets = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        phase = (t / float(n)) * 4.0
        if phase < 1.2:
            # Regime 1: Sustained macro uptrend
            drift = 0.0018
            vol = 0.022
        elif phase < 2.0:
            # Regime 2: Bearish drawdown / deleveraging
            drift = -0.0016
            vol = 0.032
        elif phase < 2.9:
            # Regime 3: Sideways chop / mean-reverting range
            drift = 0.0001 * np.sin(t / 6.0)
            vol = 0.018
        else:
            # Regime 4: Strong breakout recovery
            drift = 0.0021
            vol = 0.024
        factor_rets[t] = drift + rng.normal(0.0, vol)

    # Base prices per asset
    base_prices = [42000.0, 2350.0, 95.0, 36.0, 15.5]
    series_map: dict[str, BarSeries] = {}

    # 2. Primary anchor asset (e.g., BTC-USD)
    anchor_sym = symbols[1] if len(symbols) > 1 and symbols[1].startswith("BTC") else symbols[0]
    anchor_log_p = np.zeros(n, dtype=np.float64)
    anchor_log_p[0] = np.log(base_prices[0])
    for t in range(1, n):
        anchor_log_p[t] = anchor_log_p[t - 1] + factor_rets[t] + rng.normal(0.0, 0.008)

    # 3. Build each symbol
    ou_spread = 0.0
    ou_kappa = 0.14  # half-life ~ 4.6 bars

    for idx, sym in enumerate(symbols):
        log_p = np.zeros(n, dtype=np.float64)
        p0 = base_prices[idx % len(base_prices)]
        log_p[0] = np.log(p0)

        if sym == anchor_sym:
            log_p = anchor_log_p.copy()
        elif idx == 1 or (sym.startswith("ETH") and anchor_sym.startswith("BTC")):
            # Cointegrated pair leg with Ornstein-Uhlenbeck spread around beta=1.05 of anchor
            beta = 1.05
            alpha = np.log(p0) - beta * anchor_log_p[0]
            for t in range(1, n):
                ou_spread = (1.0 - ou_kappa) * ou_spread + rng.normal(0.0, 0.014)
                log_p[t] = alpha + beta * anchor_log_p[t] + ou_spread
        else:
            # Cross-sectional momentum asset with persistent medium-term idiosyncratic drift waves
            idio_wave = 0.0
            for t in range(1, n):
                idio_wave = 0.925 * idio_wave + rng.normal(0.0, 0.0028)
                beta_i = 0.85 + 0.15 * (idx % 3)
                log_p[t] = log_p[t - 1] + beta_i * factor_rets[t] + idio_wave + rng.normal(0.0, 0.016)

        closes = np.exp(log_p)
        opens = np.empty(n, dtype=np.float64)
        opens[0] = closes[0] * (1.0 - 0.002)
        opens[1:] = closes[:-1] * np.exp(rng.normal(0.0, 0.0015, size=n - 1))

        intrabar_span = np.abs(rng.normal(0.015, 0.006, size=n)) * closes
        highs = np.maximum(opens, closes) + 0.6 * intrabar_span
        lows = np.maximum(1e-4, np.minimum(opens, closes) - 0.6 * intrabar_span)
        volumes = 50_000_000.0 * (1.0 + np.abs(np.diff(log_p, prepend=log_p[0])) * 25.0) * rng.uniform(0.8, 1.2, size=n)

        series_map[sym] = BarSeries(
            symbol=sym,
            timestamps=timestamps.copy(),
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            volume=volumes,
        )

    return MarketMatrix(
        timestamps=timestamps,
        symbols=tuple(symbols),
        series=series_map,
    )


def fetch_yfinance_market(cfg: DataConfig) -> MarketMatrix:
    """Fetch historical OHLCV bars from Yahoo Finance (requires optional yfinance package)."""
    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Install with `pip install .[live-data]` or use source='synthetic'."
        ) from exc

    raw_series: dict[str, tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = {}
    common_ts: set[int] | None = None

    for sym in cfg.symbols:
        ticker = yf.Ticker(sym)
        df = ticker.history(period=f"{max(30, cfg.num_bars)}d", interval=cfg.timeframe)
        if df.empty:
            raise RuntimeError(f"Empty data returned from yfinance for {sym}")
        ts = (df.index.astype("int64") // 10**9).to_numpy(dtype=np.int64)
        o = df["Open"].to_numpy(dtype=np.float64)
        h = df["High"].to_numpy(dtype=np.float64)
        low_arr = df["Low"].to_numpy(dtype=np.float64)
        c = df["Close"].to_numpy(dtype=np.float64)
        v = df["Volume"].to_numpy(dtype=np.float64)
        raw_series[sym] = (ts, o, h, low_arr, c, v)
        ts_set = set(ts.tolist())
        common_ts = ts_set if common_ts is None else (common_ts & ts_set)

    if not common_ts:
        raise RuntimeError("No overlapping timestamps found across symbols from yfinance")

    sorted_ts = np.array(sorted(common_ts)[-cfg.num_bars :], dtype=np.int64)
    series_map: dict[str, BarSeries] = {}
    for sym, (ts, o, h, low_arr, c, v) in raw_series.items():
        mask = np.isin(ts, sorted_ts)
        series_map[sym] = BarSeries(
            symbol=sym,
            timestamps=ts[mask],
            open=o[mask],
            high=h[mask],
            low=low_arr[mask],
            close=c[mask],
            volume=v[mask],
        )

    return MarketMatrix(timestamps=sorted_ts, symbols=cfg.symbols, series=series_map)


def load_market_data(cfg: DataConfig) -> MarketMatrix:
    """Load market data from the configured source."""
    source = cfg.source.lower().strip()
    if source == "synthetic":
        return generate_synthetic_market(cfg)
    if source == "yfinance":
        return fetch_yfinance_market(cfg)
    raise ValueError(f"Unsupported data source '{cfg.source}'. Supported: 'synthetic', 'yfinance'")
