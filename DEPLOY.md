# TALON Deployment & Operations Guide

> **Safety Default:** TALON defaults strictly to **paper / simulation mode** (`execution.mode: paper`). No live exchange orders can be transmitted unless both the environment interlock and explicit configuration unlock are satisfied.

---

## 1. Local Setup & Verification

```bash
# 1. Create and activate a Python 3.11+ virtual environment
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# 2. Install TALON with development and verification tools
pip install -e ".[dev]"

# 3. Run static analysis, strict type checking, and unit test suite
ruff check src tests
mypy src tests
pytest

# 4. Generate the full reproducible backtest & walk-forward report
python -m talon.cli backtest --config configs/default.yaml --output-dir reports

# 5. Execute a single-step paper trading evaluation tick
python -m talon.cli paper-tick --config configs/default.yaml
```

---

## 2. Publishing to GitHub & Enabling CI

If GitHub authentication (`gh auth status`) was not active in the local environment when the repository was initialized, run the following exact commands from the project root (`C:\Users\matt\.gemini\antigravity\scratch\talon`) to publish the repository and trigger the GitHub Actions CI pipeline:

```bash
# 1. Authenticate GitHub CLI (if not already authenticated)
gh auth login

# 2. Create the remote repository on GitHub and push all commits on main
gh repo create talon --public --source=. --remote=origin --push

# 3. Verify the GitHub Actions workflow run
gh run list --limit 5
gh run watch
```

Alternatively, if using a manually created GitHub repository URL:

```bash
git remote add origin https://github.com/<your-username>/talon.git
git branch -M main
git push -u origin main
```

---

## 3. Continuous Paper-Trading Schedule

TALON includes two mechanisms for scheduled paper evaluation:

### Option A: GitHub Actions Scheduled Workflow (Zero Infrastructure)
`.github/workflows/ci.yml` includes a `schedule` trigger (`cron: "0 6 * * *"`) that runs `python -m talon.cli backtest` and `python -m talon.cli paper-tick` daily at `06:00 UTC` and uploads the resulting markdown report, JSON metrics, and SVG equity/drawdown charts as workflow artifacts (`talon-backtest-reports`).

### Option B: Systemd Timer / Cron Job on a Dedicated Host
1. Copy `.env.example` to `.env` and ensure `TALON_MODE=paper`.
2. Add a daily cron entry (or Windows Task Scheduler task):
   ```cron
   5 0 * * * cd /opt/talon && /opt/talon/.venv/bin/python -m talon.cli paper-tick --config configs/default.yaml >> /var/log/talon/paper.jsonl 2>&1
   ```

---

## 4. Dual-Key Live Execution Interlock (Production Safety)

By design, TALON refuses to initialize live order routing unless **two independent interlocks** are explicitly enabled by the operator:

1. **Environment Flag:**
   ```bash
   export TALON_MODE=live
   export TALON_LIVE_TRADING="I-ACCEPT-REAL-MONEY-RISK"
   ```
2. **User-Created Live Configuration File (`configs/live.yaml`):**
   Create a separate `configs/live.yaml` (never committed to git) containing:
   ```yaml
   execution:
     mode: live
     live_confirmed: true
   ```
3. **Subclassing `ExecutionAdapter`:**
   Even when both interlocks are unlocked, the default binary ships only with `PaperExecutionAdapter` and raises a `RuntimeError` if `mode: live` is invoked without an explicit exchange execution adapter wired into `cli.py`. This guarantees zero accidental real-money orders during research or CI runs.
