"""Configuration loading, validation, and live-mode safety locks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from talon.types import ExecutionMode

LIVE_ACKNOWLEDGEMENT_PHRASE = "I-ACCEPT-REAL-MONEY-RISK"


class ConfigError(ValueError):
    """Raised when configuration is invalid or violates safety constraints."""


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"
    symbols: tuple[str, ...] = ("BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD")
    timeframe: str = "1d"
    num_bars: int = 730
    seed: int = 42


@dataclass(frozen=True)
class TrendStrategyConfig:
    enabled: bool = True
    weight_allocation: float = 0.45
    vol_target_ann: float = 0.20
    max_asset_weight: float = 0.40
    activation_threshold: float = 0.40
    atr_stop_mult: float = 2.5


@dataclass(frozen=True)
class CrossSectionalConfig:
    enabled: bool = True
    weight_allocation: float = 0.30
    fast_lookback: int = 7
    slow_lookback: int = 21
    z_entry_threshold: float = 0.75
    z_exit_threshold: float = 0.25
    max_gross_weight: float = 0.50


@dataclass(frozen=True)
class MeanReversionConfig:
    enabled: bool = True
    weight_allocation: float = 0.25
    pair_leg_a: str = "ETH-USD"
    pair_leg_b: str = "BTC-USD"
    ols_window: int = 60
    z_window: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.35
    stop_z: float = 3.5
    min_half_life: float = 2.0
    max_half_life: float = 25.0
    max_leg_weight: float = 0.30


@dataclass(frozen=True)
class StrategiesConfig:
    vol_targeted_trend: TrendStrategyConfig = field(default_factory=TrendStrategyConfig)
    cross_sectional_momentum: CrossSectionalConfig = field(default_factory=CrossSectionalConfig)
    cointegrated_mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)


@dataclass(frozen=True)
class RiskConfig:
    max_drawdown_pct: float = 0.15
    daily_loss_limit_pct: float = 0.04
    max_asset_weight: float = 0.35
    max_gross_leverage: float = 1.50
    portfolio_vol_target: float = 0.25
    max_pairwise_correlation: float = 0.80
    correlated_cluster_cap: float = 0.45
    rebalance_deadband: float = 0.02
    vol_lookback: int = 30


@dataclass(frozen=True)
class ExecutionConfig:
    taker_fee_bps: float = 5.0
    half_spread_bps: float = 2.5
    impact_slippage_coeff: float = 0.10
    annual_funding_drag_pct: float = 0.03


@dataclass(frozen=True)
class WalkForwardConfig:
    train_bars: int = 240
    test_bars: int = 90
    step_bars: int = 90


@dataclass(frozen=True)
class TalonConfig:
    mode: ExecutionMode = ExecutionMode.PAPER
    initial_equity_usd: float = 100_000.0
    bars_per_year: int = 365
    data: DataConfig = field(default_factory=DataConfig)
    strategies: StrategiesConfig = field(default_factory=StrategiesConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)


def _validate_config(cfg: TalonConfig) -> None:
    if cfg.initial_equity_usd <= 0:
        raise ConfigError("initial_equity_usd must be positive")
    if cfg.bars_per_year <= 0:
        raise ConfigError("bars_per_year must be positive")
    if not (0.0 < cfg.risk.max_drawdown_pct < 1.0):
        raise ConfigError("risk.max_drawdown_pct must be in (0, 1)")
    if not (0.0 < cfg.risk.daily_loss_limit_pct < 1.0):
        raise ConfigError("risk.daily_loss_limit_pct must be in (0, 1)")
    if not (0.0 < cfg.risk.max_asset_weight <= 1.0):
        raise ConfigError("risk.max_asset_weight must be in (0, 1]")
    if not (0.0 < cfg.risk.max_gross_leverage <= 3.0):
        raise ConfigError("risk.max_gross_leverage must be in (0, 3.0]")
    if cfg.risk.portfolio_vol_target <= 0:
        raise ConfigError("risk.portfolio_vol_target must be positive")
    if cfg.execution.taker_fee_bps < 0 or cfg.execution.half_spread_bps < 0:
        raise ConfigError("execution fees and spreads must be non-negative")
    if len(cfg.data.symbols) < 2:
        raise ConfigError("data.symbols must contain at least 2 symbols")


def load_config(
    config_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> TalonConfig:
    """Load and validate configuration from YAML and environment variables.

    Enforces hard safety constraints:
    - Defaults strictly to PAPER mode.
    - LIVE mode requires BOTH:
      1. Environment variable TALON_LIVE_TRADING == 'I-ACCEPT-REAL-MONEY-RISK'
      2. A dedicated live config file existing on disk with `live_confirmed: true`
    """
    environ = env if env is not None else os.environ
    resolved_path = Path(config_path or environ.get("TALON_CONFIG_PATH", "configs/default.yaml"))

    raw: dict[str, Any] = {}
    if resolved_path.exists():
        with resolved_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                raw = loaded
    elif config_path is not None:
        raise ConfigError(f"Configuration file not found: {resolved_path}")

    mode_str = environ.get("TALON_MODE", str(raw.get("mode", "paper"))).strip().lower()
    if mode_str not in ("paper", "live"):
        raise ConfigError(f"Invalid execution mode '{mode_str}'. Must be 'paper' or 'live'.")

    mode = ExecutionMode(mode_str)
    if mode == ExecutionMode.LIVE:
        ack = environ.get("TALON_LIVE_TRADING", "").strip()
        if ack != LIVE_ACKNOWLEDGEMENT_PHRASE:
            raise ConfigError(
                "LIVE trading mode is locked by default. To unlock live mode, you must set "
                f"TALON_LIVE_TRADING={LIVE_ACKNOWLEDGEMENT_PHRASE} AND provide a dedicated "
                "live configuration file with 'live_confirmed: true'."
            )
        if not resolved_path.exists() or raw.get("live_confirmed") is not True:
            raise ConfigError(
                "LIVE trading mode requires an explicit configuration file on disk with "
                "'live_confirmed: true'. Paper trading is the only permitted default."
            )

    data_raw = raw.get("data", {})
    strat_raw = raw.get("strategies", {})
    trend_raw = strat_raw.get("vol_targeted_trend", {})
    csm_raw = strat_raw.get("cross_sectional_momentum", {})
    mr_raw = strat_raw.get("cointegrated_mean_reversion", {})
    risk_raw = raw.get("risk", {})
    exec_raw = raw.get("execution", {})
    wf_raw = raw.get("walk_forward", {})

    cfg = TalonConfig(
        mode=mode,
        initial_equity_usd=float(raw.get("initial_equity_usd", 100_000.0)),
        bars_per_year=int(raw.get("bars_per_year", 365)),
        data=DataConfig(
            source=str(data_raw.get("source", "synthetic")),
            symbols=tuple(data_raw.get("symbols", ("BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD"))),
            timeframe=str(data_raw.get("timeframe", "1d")),
            num_bars=int(data_raw.get("num_bars", 730)),
            seed=int(data_raw.get("seed", 42)),
        ),
        strategies=StrategiesConfig(
            vol_targeted_trend=TrendStrategyConfig(
                enabled=bool(trend_raw.get("enabled", True)),
                weight_allocation=float(trend_raw.get("weight_allocation", 0.45)),
                vol_target_ann=float(trend_raw.get("vol_target_ann", 0.20)),
                max_asset_weight=float(trend_raw.get("max_asset_weight", 0.40)),
                activation_threshold=float(trend_raw.get("activation_threshold", 0.40)),
                atr_stop_mult=float(trend_raw.get("atr_stop_mult", 2.5)),
            ),
            cross_sectional_momentum=CrossSectionalConfig(
                enabled=bool(csm_raw.get("enabled", True)),
                weight_allocation=float(csm_raw.get("weight_allocation", 0.30)),
                fast_lookback=int(csm_raw.get("fast_lookback", 7)),
                slow_lookback=int(csm_raw.get("slow_lookback", 21)),
                z_entry_threshold=float(csm_raw.get("z_entry_threshold", 0.75)),
                z_exit_threshold=float(csm_raw.get("z_exit_threshold", 0.25)),
                max_gross_weight=float(csm_raw.get("max_gross_weight", 0.50)),
            ),
            cointegrated_mean_reversion=MeanReversionConfig(
                enabled=bool(mr_raw.get("enabled", True)),
                weight_allocation=float(mr_raw.get("weight_allocation", 0.25)),
                pair_leg_a=str(mr_raw.get("pair_leg_a", "ETH-USD")),
                pair_leg_b=str(mr_raw.get("pair_leg_b", "BTC-USD")),
                ols_window=int(mr_raw.get("ols_window", 60)),
                z_window=int(mr_raw.get("z_window", 20)),
                entry_z=float(mr_raw.get("entry_z", 2.0)),
                exit_z=float(mr_raw.get("exit_z", 0.35)),
                stop_z=float(mr_raw.get("stop_z", 3.5)),
                min_half_life=float(mr_raw.get("min_half_life", 2.0)),
                max_half_life=float(mr_raw.get("max_half_life", 25.0)),
                max_leg_weight=float(mr_raw.get("max_leg_weight", 0.30)),
            ),
        ),
        risk=RiskConfig(
            max_drawdown_pct=float(risk_raw.get("max_drawdown_pct", 0.15)),
            daily_loss_limit_pct=float(risk_raw.get("daily_loss_limit_pct", 0.04)),
            max_asset_weight=float(risk_raw.get("max_asset_weight", 0.35)),
            max_gross_leverage=float(risk_raw.get("max_gross_leverage", 1.50)),
            portfolio_vol_target=float(risk_raw.get("portfolio_vol_target", 0.25)),
            max_pairwise_correlation=float(risk_raw.get("max_pairwise_correlation", 0.80)),
            correlated_cluster_cap=float(risk_raw.get("correlated_cluster_cap", 0.45)),
            rebalance_deadband=float(risk_raw.get("rebalance_deadband", 0.02)),
            vol_lookback=int(risk_raw.get("vol_lookback", 30)),
        ),
        execution=ExecutionConfig(
            taker_fee_bps=float(exec_raw.get("taker_fee_bps", 5.0)),
            half_spread_bps=float(exec_raw.get("half_spread_bps", 2.5)),
            impact_slippage_coeff=float(exec_raw.get("impact_slippage_coeff", 0.10)),
            annual_funding_drag_pct=float(exec_raw.get("annual_funding_drag_pct", 0.03)),
        ),
        walk_forward=WalkForwardConfig(
            train_bars=int(wf_raw.get("train_bars", 240)),
            test_bars=int(wf_raw.get("test_bars", 90)),
            step_bars=int(wf_raw.get("step_bars", 90)),
        ),
    )
    _validate_config(cfg)
    return cfg
