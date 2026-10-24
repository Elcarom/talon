"""Command-line interface for the TALON quantitative trading worker."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from talon.backtest import run_backtest, run_walk_forward
from talon.config import ConfigError, load_config
from talon.data import load_market_data
from talon.logging_utils import get_logger
from talon.reporting import generate_reports
from talon.risk import RiskEngine
from talon.strategies import build_default_strategies, combine_strategy_intents
from talon.types import PortfolioState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="talon",
        description="TALON: Tactical Allocation & Liquidity Order Engine (Paper/Simulation by Default)",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bt_parser = sub.add_parser(
        "backtest",
        help="Run causal backtest and walk-forward validation, generating reports/ artifacts",
    )
    bt_parser.add_argument(
        "-o",
        "--output-dir",
        default="reports",
        help="Output directory for generated reports and charts (default: reports)",
    )

    wf_parser = sub.add_parser(
        "walk-forward",
        help="Run rolling walk-forward evaluation and print JSON summary",
    )
    wf_parser.add_argument(
        "--train-bars",
        type=int,
        default=None,
        help="Override training window length in bars",
    )
    wf_parser.add_argument(
        "--test-bars",
        type=int,
        default=None,
        help="Override out-of-sample test window length in bars",
    )

    sub.add_parser(
        "paper-tick",
        help="Execute a single causal paper-trading evaluation cycle on the latest bar",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = get_logger("talon")

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        logger.error(f"Configuration validation error: {exc}")
        return 2

    market = load_market_data(cfg.data)

    if args.command == "backtest":
        logger.info(
            "Starting full-sample causal backtest and walk-forward evaluation",
            extra={
                "structured_data": {
                    "mode": str(cfg.mode),
                    "symbols": list(cfg.data.symbols),
                    "num_bars": market.num_bars,
                }
            },
        )
        bt_res = run_backtest(cfg, market)
        wf_res = run_walk_forward(cfg, market)
        paths = generate_reports(cfg, bt_res, wf_res, output_dir=args.output_dir)
        logger.info(
            "Backtest and walk-forward report generated",
            extra={
                "structured_data": {
                    "report": str(paths["report_md"]),
                    "sharpe": bt_res.metrics.sharpe_ratio,
                    "max_dd_pct": bt_res.metrics.max_drawdown_pct,
                    "oos_sharpe": wf_res.oos_combined_metrics.sharpe_ratio,
                }
            },
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "report_path": str(paths["report_md"]),
                    "full_sample": bt_res.metrics.to_dict(),
                    "walk_forward_oos": wf_res.oos_combined_metrics.to_dict(),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "walk-forward":
        wf_res = run_walk_forward(cfg, market)
        print(json.dumps(wf_res.to_dict(), indent=2))
        return 0

    if args.command == "paper-tick":
        strategies = build_default_strategies(cfg.strategies)
        intents_by_strat = {
            s.strategy_id: s.generate_signals(market, bars_per_year=cfg.bars_per_year)
            for s in strategies
        }
        proposed_weights, _ = combine_strategy_intents(
            market.symbols,
            strategies,
            intents_by_strat,
        )
        state = PortfolioState(
            cash_usd=cfg.initial_equity_usd,
            equity_usd=cfg.initial_equity_usd,
            peak_equity_usd=cfg.initial_equity_usd,
            day_start_equity_usd=cfg.initial_equity_usd,
            positions={sym: 0.0 for sym in market.symbols},
            weights={sym: 0.0 for sym in market.symbols},
        )
        risk_engine = RiskEngine(cfg.risk, bars_per_year=cfg.bars_per_year)
        report = risk_engine.evaluate(
            timestamp=int(market.timestamps[-1]),
            state=state,
            proposed_weights=proposed_weights,
            market=market,
        )
        summary = {
            "timestamp": report.timestamp,
            "mode": str(cfg.mode),
            "halted": report.halted,
            "ex_ante_vol_ann": round(report.ex_ante_vol_ann, 4),
            "gross_leverage_approved": round(report.gross_leverage_approved, 4),
            "decisions": {
                sym: {
                    "proposed_weight": round(d.proposed_weight, 4),
                    "approved_weight": round(d.approved_weight, 4),
                    "action": str(d.action),
                    "reasons": list(d.reasons),
                }
                for sym, d in report.decisions.items()
            },
        }
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
