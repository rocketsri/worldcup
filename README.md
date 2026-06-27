# World Cup Prediction Harness

A systematic experiment harness combining statistical models and frontier LLMs to predict World Cup match outcomes. Backtested on 2018 and 2022 WC; the best configuration (**Opus 4.8 with 5-shot prompting**) achieves **65.6% winner accuracy on 2022 WC**, outperforming Pinnacle sportsbook (61.7%), FiveThirtyEight (54.7%), Gracenote (53.1%), and Goldman Sachs (51.6%).

An independent live benchmark (The Arena, 2026 WC) validates the result: live Opus on actual 2026 games hits 65.2%, matching our backtested estimate.

## Key Results

| System | 2022 WC Accuracy | RPS |
|---|---|---|
| **Opus 5-shot (E19)** | **65.6%** | **0.177** |
| Pinnacle sportsbook | 61.7% | — |
| FiveThirtyEight | 54.7% | — |
| Stat model (E22) | 54.7% | 0.238 |
| Gracenote | 53.1% | — |
| Goldman Sachs | 51.6% | — |
| Random baseline | 33.3% | — |

Full methodology, experiment log, and analysis: see [`PRESENTATION.md`](PRESENTATION.md) and [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd project-interview
pip install -r requirements.txt
```

### 2. Add API keys

Create a `.env` file in the project root (never commit this file):

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Both keys are required. Anthropic key runs Claude (Opus, Sonnet). OpenAI key runs GPT-4o and o3-mini.

---

## Running

### Full pipeline (all experiments + 2026 predictions)

```bash
python main.py
```

This runs in order:
1. **Phase 1** — stat baseline experiments (E01–E04): Poisson, XGBoost on ELO + form features
2. **Phase 2** — LLM experiments (E10–E21): LLM direct prediction across models and prompting strategies
3. **Phase 3** — Coordinator loop: Opus reads the leaderboard and proposes new experiments
4. **Phase 4** — 2026 predictions: best model predicts remaining 2026 WC games

Runtime: ~45–90 minutes depending on API response time (LLM calls are parallelized per tournament).

### Stat experiments only (no API keys needed)

```bash
python -c "
from experiments.runner import run_stat_experiment
from experiments.config import INITIAL_GRID
from experiments.leaderboard import Leaderboard
lb = Leaderboard()
for cfg in INITIAL_GRID[:4]:
    run_stat_experiment(cfg, lb)
lb.print_table()
"
```

---

## Project Structure

```
project-interview/
├── .env                          # API keys (create this, never commit)
├── requirements.txt
├── main.py                       # Master orchestrator
├── config.py                     # API key loading, global constants
│
├── data/
│   ├── fetch.py                  # ESPN API loader (WC 2006–2022, 320 games)
│   ├── elo.py                    # Rolling ELO ratings (WC-only + all-international)
│   ├── features.py               # Feature engineering: form, H2H, ELO diff
│   └── raw/                      # Cached data files (auto-fetched on first run)
│       ├── all_wc_matches.csv    # 320 WC games from ESPN (2006–2022)
│       ├── all_wc_matches_1994.csv  # 758 games incl. Confederations Cup (1994–2022)
│       ├── international_results.csv  # 32k international games from martj42
│       └── espn_wc_*.json        # ESPN API cache per year
│
├── models/
│   ├── base.py                   # Abstract PredictionModel interface
│   ├── poisson_model.py          # Poisson regression via MLE
│   ├── dixon_coles.py            # Dixon-Coles bivariate Poisson
│   └── xgboost_model.py          # XGBoost W/D/L classifier
│
├── agents/
│   ├── tools.py                  # Tool implementations for LLM context agents
│   ├── news_search.py            # Web search for team news / injuries
│   └── coordinator.py            # Autoresearch coordinator (Opus proposes experiments)
│
├── experiments/
│   ├── config.py                 # ExperimentConfig dataclass + full experiment grid
│   ├── runner.py                 # Async parallel experiment runner
│   └── leaderboard.py            # Results ledger + Rich table display
│
├── evaluation/
│   ├── metrics.py                # winner_accuracy, RPS, MAE
│   └── backtest.py               # Temporal backtest engine (no leakage)
│
├── predict/
│   └── predict_2026.py           # 2026 bracket predictions
│
├── results/
│   ├── leaderboard.json          # All experiment results (auto-updated)
│   ├── predictions_2026.json     # 2026 WC predictions
│   ├── kelly_simulation.json     # Kelly criterion bankroll simulation
│   └── coordinator_reasoning.json  # Coordinator experiment proposals
│
├── PRESENTATION.md               # Full writeup: methodology, results, analysis
└── RESEARCH_LOG.md               # Chronological experiment log with decisions
```

---

## Data Sources

| Dataset | Source | Games | Used for |
|---|---|---|---|
| WC 2006–2022 | ESPN API (auto-fetched) | 320 | Training + backtest |
| WC + Confed Cup 1994–2022 | [jfjelstul/world-cup](https://github.com/jfjelstul/world-cup) | 758 | Extended ELO signal |
| All international 1990–2024 | [martj42/international_results](https://github.com/martj42/international_results) | 32k | International ELO (experimental) |

The ESPN data is fetched automatically on first run and cached in `data/raw/`. The jfjelstul and martj42 CSVs are included in the repo for immediate reproducibility.

---

## Experiments

26 experiments total, organized in waves:

- **E01–E04**: Stat baselines (Poisson, XGBoost, ELO-only, full feature set)
- **E05–E09**: Stat + LLM hybrid (LLM extracts qualitative features added to stat model)
- **E10–E15**: LLM direct prediction (zero-shot and 5-shot, across Sonnet/Opus/GPT-4o/o3-mini)
- **E16–E21**: Coordinator-proposed (chain-of-thought, improved few-shot, ablations)
- **E22–E26**: Feature ablations and alternative approaches (minimal features, international ELO, FIFA rank, market prior)

Key finding: **knowledge leakage** is the critical variable for LLM performance. GPT-4o drops 14.1 percentage points from 2018 → 2022 (it memorized 2018 results). Opus drops only 3.1 pp — a much more honest predictor.

---

## Reproducing Key Results

### Best model (E19 — Opus 5-shot)
Requires `ANTHROPIC_API_KEY`. Runs Opus 4.8 with 5 few-shot examples on all 2022 WC games, predicting winner from match context and ELO ratings.

### Stat baseline (E22 — 5-feature Poisson)
No API keys needed. Uses `elo_diff`, `confederation_diff`, `win_streak_a`, `win_streak_b`, `stage_weight` as features. Achieves 54.7% on 2022 WC.

### Kelly criterion simulation
```bash
python -c "import json; print(json.load(open('results/kelly_simulation.json')))"
```
Shows -37.9% ROI using Kelly criterion on the stat model — confirms you need well-calibrated probabilities (low RPS) before sizing bets.
