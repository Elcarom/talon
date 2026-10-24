"""Report and SVG chart generation for reproducible backtest and walk-forward artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from talon.backtest import BacktestResult, WalkForwardSummary
from talon.config import TalonConfig


def _render_svg_line_chart(
    timestamps: NDArray[np.int64],
    values: NDArray[np.float64],
    title: str,
    y_label: str,
    stroke_color: str = "#10b981",
    fill_color: str = "rgba(16, 185, 129, 0.12)",
    is_percentage: bool = False,
) -> str:
    """Render a clean, standalone SVG chart with gridlines and axis labels."""
    width = 900
    height = 360
    pad_left = 80
    pad_right = 30
    pad_top = 45
    pad_bottom = 45
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    n = len(values)
    if n < 2:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'

    y_min = float(np.min(values))
    y_max = float(np.max(values))
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0

    points: list[str] = []
    for i in range(n):
        x = pad_left + (i / float(n - 1)) * plot_w
        y = pad_top + plot_h - ((float(values[i]) - y_min) / (y_max - y_min)) * plot_h
        points.append(f"{x:.1f},{y:.1f}")

    polyline_pts = " ".join(points)
    area_pts = f"{pad_left:.1f},{pad_top + plot_h:.1f} {polyline_pts} {pad_left + plot_w:.1f},{pad_top + plot_h:.1f}"

    # Horizontal gridlines
    grid_lines: list[str] = []
    for g in range(5):
        frac = g / 4.0
        gy = pad_top + plot_h * (1.0 - frac)
        gval = y_min + frac * (y_max - y_min)
        label_str = f"{gval:.1f}%" if is_percentage else f"${gval:,.0f}"
        grid_lines.append(
            f'<line x1="{pad_left}" y1="{gy:.1f}" x2="{width - pad_right}" y2="{gy:.1f}" '
            f'stroke="#334155" stroke-dasharray="4,4" stroke-width="1" />'
            f'<text x="{pad_left - 10}" y="{gy + 4:.1f}" fill="#94a3b8" font-family="monospace" '
            f'font-size="11" text-anchor="end">{label_str}</text>'
        )

    start_dt = datetime.fromtimestamp(int(timestamps[0]), tz=UTC).strftime("%Y-%m-%d")
    mid_dt = datetime.fromtimestamp(int(timestamps[n // 2]), tz=UTC).strftime("%Y-%m-%d")
    end_dt = datetime.fromtimestamp(int(timestamps[-1]), tz=UTC).strftime("%Y-%m-%d")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
  <rect width="{width}" height="{height}" fill="#0f172a" rx="8" />
  <text x="{pad_left}" y="28" fill="#f8fafc" font-family="sans-serif" font-size="15" font-weight="bold">{title}</text>
  <text x="{width - pad_right}" y="28" fill="#64748b" font-family="monospace" font-size="12" text-anchor="end">{y_label}</text>
  {"".join(grid_lines)}
  <polygon points="{area_pts}" fill="{fill_color}" />
  <polyline fill="none" stroke="{stroke_color}" stroke-width="2.2" points="{polyline_pts}" />
  <text x="{pad_left}" y="{height - 14}" fill="#94a3b8" font-family="monospace" font-size="11" text-anchor="start">{start_dt}</text>
  <text x="{pad_left + plot_w / 2:.1f}" y="{height - 14}" fill="#94a3b8" font-family="monospace" font-size="11" text-anchor="middle">{mid_dt}</text>
  <text x="{width - pad_right}" y="{height - 14}" fill="#94a3b8" font-family="monospace" font-size="11" text-anchor="end">{end_dt}</text>
</svg>"""
    return svg


def generate_reports(
    cfg: TalonConfig,
    bt_result: BacktestResult,
    wf_summary: WalkForwardSummary,
    output_dir: str | Path = "reports",
) -> dict[str, Path]:
    """Write reproducible backtest report, JSON metrics, and SVG equity/drawdown charts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    equity_svg_path = out_path / "equity_curve.svg"
    drawdown_svg_path = out_path / "drawdown.svg"
    metrics_json_path = out_path / "metrics.json"
    report_md_path = out_path / "backtest_report.md"

    equity_svg = _render_svg_line_chart(
        timestamps=bt_result.timestamps,
        values=bt_result.equity_curve,
        title="TALON Portfolio Equity Curve (Net of Fees, Spread & Slippage)",
        y_label="USD Equity",
        stroke_color="#10b981",
        fill_color="rgba(16, 185, 129, 0.14)",
        is_percentage=False,
    )
    equity_svg_path.write_text(equity_svg, encoding="utf-8")

    dd_pct_series = -bt_result.drawdown_curve * 100.0
    drawdown_svg = _render_svg_line_chart(
        timestamps=bt_result.timestamps,
        values=dd_pct_series,
        title="TALON Portfolio Underwater Drawdown (%)",
        y_label="Drawdown %",
        stroke_color="#ef4444",
        fill_color="rgba(239, 68, 68, 0.18)",
        is_percentage=True,
    )
    drawdown_svg_path.write_text(drawdown_svg, encoding="utf-8")

    payload: dict[str, Any] = {
        "project": "TALON Quantitative Worker",
        "mode": str(cfg.mode),
        "data_source": cfg.data.source,
        "symbols": list(cfg.data.symbols),
        "num_bars": int(len(bt_result.timestamps)),
        "full_sample_metrics": bt_result.metrics.to_dict(),
        "strategy_sleeve_metrics": {
            k: v.to_dict() for k, v in bt_result.strategy_metrics.items()
        },
        "walk_forward": wf_summary.to_dict(),
    }
    metrics_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    m = bt_result.metrics
    wf_m = wf_summary.oos_combined_metrics

    strat_rows = []
    for s_id, sm in bt_result.strategy_metrics.items():
        strat_rows.append(
            f"| `{s_id}` | {sm.total_return_pct:+.2f}% | {sm.cagr_pct:+.2f}% | "
            f"{sm.sharpe_ratio:.2f} | {sm.sortino_ratio:.2f} | {sm.max_drawdown_pct:.2f}% | "
            f"{sm.calmar_ratio:.2f} | {sm.hit_rate_pct:.1f}% | {sm.total_trades} |"
        )
    strat_table = "\n".join(strat_rows)

    fold_rows = []
    for f in wf_summary.folds:
        t_start = datetime.fromtimestamp(f.test_start_ts, tz=UTC).strftime("%Y-%m-%d")
        t_end = datetime.fromtimestamp(f.test_end_ts, tz=UTC).strftime("%Y-%m-%d")
        fold_rows.append(
            f"| Fold {f.fold_index + 1} | {t_start} to {t_end} | "
            f"{f.in_sample_metrics.sharpe_ratio:.2f} | {f.out_of_sample_metrics.sharpe_ratio:.2f} | "
            f"{f.out_of_sample_metrics.total_return_pct:+.2f}% | "
            f"{f.out_of_sample_metrics.max_drawdown_pct:.2f}% |"
        )
    fold_table = "\n".join(fold_rows)

    md_content = f"""# TALON Reproducible Backtest & Walk-Forward Report

> **Disclaimer:** *Educational & research simulation only. Not financial advice. Past or simulated performance is not indicative of future results.*

---

## 1. Run Configuration Summary

- **Execution Mode:** `{cfg.mode}` (Paper / Simulation by default)
- **Data Source:** `{cfg.data.source}` (Seed: `{cfg.data.seed}`, Bars: `{len(bt_result.timestamps)}`, Timeframe: `{cfg.data.timeframe}`)
- **Universe:** `{", ".join(cfg.data.symbols)}`
- **Transaction Cost Model:** `{cfg.execution.taker_fee_bps} bps` taker fee + `{cfg.execution.half_spread_bps} bps` half-spread + `{cfg.execution.impact_slippage_coeff}` vol impact coeff + `{cfg.execution.annual_funding_drag_pct * 100:.1f}%` annualized funding drag
- **Risk Guardrails:** Max DD Halt `{cfg.risk.max_drawdown_pct * 100:.1f}%`, Daily Loss Limit `{cfg.risk.daily_loss_limit_pct * 100:.1f}%`, Asset Cap `{cfg.risk.max_asset_weight * 100:.1f}%`, Max Gross Leverage `{cfg.risk.max_gross_leverage:.2f}x`

---

## 2. Portfolio Full-Sample Performance (Net of All Costs)

| Metric | Combined Portfolio | Out-of-Sample (Walk-Forward Stitched) |
| :--- | :---: | :---: |
| **Initial Equity** | `${m.initial_equity_usd:,.2f}` | `${wf_m.initial_equity_usd:,.2f}` |
| **Final Equity** | `${m.final_equity_usd:,.2f}` | `${wf_m.final_equity_usd:,.2f}` |
| **Total Return** | `{m.total_return_pct:+.2f}%` | `{wf_m.total_return_pct:+.2f}%` |
| **Annualized Return (CAGR)** | `{m.cagr_pct:+.2f}%` | `{wf_m.cagr_pct:+.2f}%` |
| **Annualized Volatility** | `{m.annualized_vol_pct:.2f}%` | `{wf_m.annualized_vol_pct:.2f}%` |
| **Sharpe Ratio** | `{m.sharpe_ratio:.3f}` | `{wf_m.sharpe_ratio:.3f}` |
| **Sortino Ratio** | `{m.sortino_ratio:.3f}` | `{wf_m.sortino_ratio:.3f}` |
| **Max Drawdown** | `{m.max_drawdown_pct:.2f}%` | `{wf_m.max_drawdown_pct:.2f}%` |
| **Calmar Ratio** | `{m.calmar_ratio:.3f}` | `{wf_m.calmar_ratio:.3f}` |
| **Annualized Turnover** | `{m.annualized_turnover:.2f}x` | `{wf_m.annualized_turnover:.2f}x` |
| **Hit Rate (Active Bars)** | `{m.hit_rate_pct:.2f}%` | `{wf_m.hit_rate_pct:.2f}%` |
| **Profit Factor** | `{m.profit_factor:.3f}` | `{wf_m.profit_factor:.3f}` |
| **Total Executed Fills** | `{m.total_trades}` | `{wf_m.total_trades}` |
| **Total Exchange Fees** | `${m.total_fees_usd:,.2f}` | `${wf_m.total_fees_usd:,.2f}` |
| **Total Slippage & Impact** | `${m.total_slippage_usd:,.2f}` | `${wf_m.total_slippage_usd:,.2f}` |
| **Total Funding Drag** | `${m.total_funding_usd:,.2f}` | `${wf_m.total_funding_usd:,.2f}` |
| **Risk Engine Vetoes** | `{m.risk_veto_count}` | `{wf_m.risk_veto_count}` |
| **Risk Engine Scalings** | `{m.risk_scale_count}` | `{wf_m.risk_scale_count}` |

---

## 3. Standalone Strategy Sleeve Attribution

| Strategy Sleeve | Total Return | CAGR | Sharpe | Sortino | Max DD | Calmar | Hit Rate | Trades |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{strat_table}

---

## 4. Walk-Forward Evaluation (`{cfg.walk_forward.train_bars}`d Train / `{cfg.walk_forward.test_bars}`d OOS Test)

- **Mean In-Sample Sharpe:** `{wf_summary.mean_is_sharpe:.3f}`
- **Mean Out-of-Sample Sharpe:** `{wf_summary.mean_oos_sharpe:.3f}`
- **OOS / IS Sharpe Retention Ratio:** `{wf_summary.sharpe_decay_ratio:.3f}`

| Fold | Out-of-Sample Window | IS Sharpe | OOS Sharpe | OOS Return | OOS Max DD |
| :--- | :---: | :---: | :---: | :---: | :---: |
{fold_table}

---

## 5. Visual Artifacts

- Equity Curve: [`equity_curve.svg`](equity_curve.svg)
- Underwater Drawdown: [`drawdown.svg`](drawdown.svg)
- Machine-Readable Metrics: [`metrics.json`](metrics.json)

![Equity Curve](equity_curve.svg)

![Underwater Drawdown](drawdown.svg)

---

## 6. Honest Caveats & Limitations

1. **Synthetic / Historical Regime Dependency:** The default reproducible benchmark runs on a seeded multi-regime statistical process (`seed=42`) to ensure deterministic CI verification without external API flakiness. Real crypto and equity markets experience fat-tailed liquidity black holes, exchange outages, and structural correlation breaks that can exceed historical or parametric bounds.
2. **Funding & Borrow Variability:** While this backtest deducts a continuous `{cfg.execution.annual_funding_drag_pct * 100:.1f}%` annualized funding drag on open notional, real perpetual futures funding rates can spike to `>50%` annualized during crowded bull runs, eroding trend-following carry.
3. **Cointegration Breakdown Risk:** Statistical arbitrage pairs (`ETH-USD` vs `BTC-USD`) can undergo permanent structural divergence where the Ornstein-Uhlenbeck half-life filter triggers a stop-loss exit at a realized loss.
"""
    report_md_path.write_text(md_content, encoding="utf-8")

    return {
        "report_md": report_md_path,
        "metrics_json": metrics_json_path,
        "equity_svg": equity_svg_path,
        "drawdown_svg": drawdown_svg_path,
    }
