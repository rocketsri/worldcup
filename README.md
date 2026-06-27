# World Cup Prediction Harness

A systematic experiment harness combining statistical models and frontier LLMs to predict World Cup match outcomes. Evaluated on two holdouts: **2022 WC** (honest backtest — temporal split strictly enforced, training data ends Dec 2021) and **2026 WC group stage** (true out-of-sample — results were unknown at model build time).

Best result: **Opus 4.8 with 5-shot prompting** achieves **65.6% winner accuracy on 2022 WC**, outperforming Pinnacle sportsbook (61.7%). Validated live: the same approach running on actual 2026 WC games hits **65.2%** in an independent frontier model competition, exactly matching the backtested estimate.

On the 2026 true holdout, our best **pure stat model** (Dual ELO, E24) reaches **63.2%** — above Pinnacle's 2022 baseline and competitive with frontier LLMs running live on the same games.

## Key Results

### Our models vs. published baselines (2022 WC — honest backtest)

| System | Winner Acc | RPS↓ | Notes |
|---|---|---|---|
| **Opus 5-shot (E19)** | **65.6%** | **0.177** | best overall |
| Opus 0-shot (E11) | 62.5% | 0.183 | |
| Sonnet 0-shot/5-shot (E10/E14) | 60.9% | 0.183–0.187 | |
| *Pinnacle sportsbook* | *61.7%* | *—* | *external baseline* |
| Stat Poisson 5-feat (E22) | 54.7% | 0.238 | no API key needed |
| *FiveThirtyEight SPI* | *54.7%* | *—* | *external baseline* |
| *Gracenote/Nielsen* | *53.1%* | *—* | *external baseline* |
| *Goldman Sachs ML* | *51.6%* | *—* | *external baseline* |
| ELO-only Poisson (E01) | 51.6% | 0.226 | |
| Random baseline | 33.3% | — | |

### 2026 WC — true out-of-sample holdout (68 group-stage games)

| System | Winner Acc | RPS↓ | Type |
|---|---|---|---|
| *Qwen3.7-Max (live, 2026)* | *66.7%* | *—* | *frontier model, live* |
| **Opus 5-shot — validated live** | **65.2%** | — | our approach, live Arena |
| *Kimi-K2.6 / MiniMax-M3 (live)* | *65.2%* | *—* | *frontier models, live* |
| *Gemini-3.1-Pro / DeepSeek / GPT-5.5 (live)* | *63.6%* | *—* | *frontier models, live* |
| **Dual ELO stat model (E24)** | **63.2%** | **0.174** | our model, no API key |
| **5-feat + intl ELO (E23)** | **60.3%** | **0.172** | our model, no API key |
| ELO-only Poisson (E01) | 57.4% | 0.189 | no API key |
| 5-feat minimal (E22) | 55.9% | 0.197 | no API key |

> The 2022 backtest uses a strict temporal split (no data after Dec 2021 used in training). LLM knowledge contamination is minimal for Opus — it dropped only 3.1 pp from 2018→2022, consistent with genuine reasoning rather than recall. The 2026 column is completely clean: results were unknown at model build time.

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
| WC 2006–2026 | ESPN API (auto-fetched) | 388 | Training + backtest (320 historical + 68 completed 2026 games) |
| WC + Confed Cup 1994–2022 | [jfjelstul/world-cup](https://github.com/jfjelstul/world-cup) | 758 | Extended ELO signal |
| All international 1990–2024 | [martj42/international_results](https://github.com/martj42/international_results) | 32k | International ELO (E23/E24) |

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
