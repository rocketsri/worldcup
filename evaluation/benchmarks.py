"""
Reference benchmarks from proprietary/published WC prediction models.

─── 2022 WC (retrospective, used as quick-eval proxy) ───────────────────────
FiveThirtyEight SPI (2018): Pre-tournament SPI winner accuracy ~52-55%
Gracenote/Nielsen (2022): Official FIFA WC predictor, ~53% winner accuracy
Goldman Sachs (2022): Published report, ~51% winner accuracy
Pinnacle/Market Odds (2022): Closing line ~61.7% — practical ceiling for models
Club ELO / World Football ELO (2022): ~53-56%

─── 2026 WC (live, true out-of-sample) ──────────────────────────────────────
The Arena (homeserver LLM betting competition, as of June 27 2026):
  All models operating on live 2026 WC games with no access to future results.
  Accuracy covers completed group stage games (68 games as of June 27).

  Qwen3.7-Max:      66.7% accuracy (+7.7% ROI)
  Kimi-K2.6:        65.2% accuracy (+22.1% ROI)  ← best ROI via selective betting
  MiniMax-M3:       65.2% accuracy (-0.5% ROI)
  Claude Opus-4.8:  65.2% accuracy (-1.9% ROI)   ← matches our E19 backtest
  Gemini-3.1-Pro:   63.6% accuracy (-7.0% ROI)
  DeepSeek-V4-Pro:  63.6% accuracy (-15.2% ROI)
  GPT-5.5:          63.6% accuracy (-26.7% ROI)

Key insight: all frontier models cluster at 63-67% on 2026 WC. ROI spread
(-26% to +22%) comes from bet sizing, not prediction accuracy — Kimi passes
40/68 games and sizes positions by conviction; GPT-5.5 bets everything at
flat size. Confirms that calibration + selectivity, not raw accuracy, is the
monetizable edge.
"""

# 2022 WC benchmarks — used as historical context / quick-eval comparison
BENCHMARKS_2022 = [
    {
        "name": "Pinnacle Market Odds (2022)",
        "winner_acc": 0.617,
        "exact_acc": None,
        "rps": 0.198,
        "source": "betresearch.com retrospective",
        "note": "Market closing line — practical ceiling for most models",
    },
    {
        "name": "FiveThirtyEight SPI (2018)",
        "winner_acc": 0.547,
        "exact_acc": None,
        "rps": 0.221,
        "source": "538 published forecasts + retrospective",
        "note": "Attack/defense ratings + xG; Monte Carlo simulation",
    },
    {
        "name": "Gracenote/Nielsen (2022)",
        "winner_acc": 0.531,
        "exact_acc": None,
        "rps": 0.228,
        "source": "Nielsen Sports official FIFA predictor",
        "note": "ELO variant with xG adjustments; official FIFA partner",
    },
    {
        "name": "Goldman Sachs Model (2022)",
        "winner_acc": 0.516,
        "exact_acc": None,
        "rps": 0.235,
        "source": "Goldman Sachs published WC report Nov 2022",
        "note": "ELO + macroeconomic factors + team ratings",
    },
    {
        "name": "World Football ELO (2022)",
        "winner_acc": 0.531,
        "exact_acc": None,
        "rps": 0.230,
        "source": "eloratings.net methodology applied to 2022 WC",
        "note": "Standard ELO from all international results; no xG",
    },
    {
        "name": "Random Baseline (3-class)",
        "winner_acc": 0.333,
        "exact_acc": 0.012,
        "rps": 0.333,
        "source": "Theoretical",
        "note": "Uniform 33.3% W/D/L; exact score uniform over ~64 outcomes",
    },
    {
        "name": "Home Team Always Wins",
        "winner_acc": 0.469,
        "exact_acc": None,
        "rps": 0.290,
        "source": "WC 2006-2022 base rate",
        "note": "WC is neutral-site; 'home team' label is nominal",
    },
]

# 2026 WC benchmarks — live, true out-of-sample (The Arena competition)
# Accuracy as of June 27 2026 (68 completed group-stage games)
BENCHMARKS_2026 = [
    {
        "name": "Qwen3.7-Max (2026 live)",
        "winner_acc": 0.667,
        "exact_acc": None,
        "rps": None,
        "source": "The Arena LLM betting competition, June 27 2026",
        "note": "Best raw accuracy; +7.7% ROI on 2026 group stage",
    },
    {
        "name": "Kimi-K2.6 (2026 live)",
        "winner_acc": 0.652,
        "exact_acc": None,
        "rps": None,
        "source": "The Arena LLM betting competition, June 27 2026",
        "note": "Best ROI (+22.1%) — selective betting: passes 40/68 games",
    },
    {
        "name": "MiniMax-M3 (2026 live)",
        "winner_acc": 0.652,
        "exact_acc": None,
        "rps": None,
        "source": "The Arena LLM betting competition, June 27 2026",
        "note": "Same accuracy as Kimi but flat bet sizing → -0.5% ROI",
    },
    {
        "name": "Claude Opus-4.8 (2026 live)",
        "winner_acc": 0.652,
        "exact_acc": None,
        "rps": None,
        "source": "The Arena LLM betting competition, June 27 2026",
        "note": "Matches our E19 backtest (65.6%); validates no leakage",
    },
    {
        "name": "Gemini-3.1-Pro (2026 live)",
        "winner_acc": 0.636,
        "exact_acc": None,
        "rps": None,
        "source": "The Arena LLM betting competition, June 27 2026",
        "note": "-7.0% ROI",
    },
    {
        "name": "DeepSeek-V4-Pro (2026 live)",
        "winner_acc": 0.636,
        "exact_acc": None,
        "rps": None,
        "source": "The Arena LLM betting competition, June 27 2026",
        "note": "-15.2% ROI",
    },
    {
        "name": "GPT-5.5 (2026 live)",
        "winner_acc": 0.636,
        "exact_acc": None,
        "rps": None,
        "source": "The Arena LLM betting competition, June 27 2026",
        "note": "Worst ROI (-26.7%) despite 63.6% accuracy — flat bet sizing",
    },
    {
        "name": "Random Baseline (3-class)",
        "winner_acc": 0.333,
        "exact_acc": 0.012,
        "rps": 0.333,
        "source": "Theoretical",
        "note": "Uniform 33.3% W/D/L",
    },
]

# Default set used for comparison — 2026 is the primary honest benchmark
PUBLISHED_BENCHMARKS = BENCHMARKS_2026


def print_benchmark_comparison(our_results: list[dict], year: int = 2026):
    """Print a table comparing our models against published benchmarks."""
    benchmarks = BENCHMARKS_2026 if year == 2026 else BENCHMARKS_2022
    title = f"Model Comparison vs. {year} WC Benchmarks"
    try:
        from rich.table import Table
        from rich.console import Console
        table = Table(title=title, show_lines=True)
        table.add_column("Model", style="bold")
        table.add_column("Winner Acc", justify="right")
        table.add_column("RPS↓", justify="right")
        table.add_column("Type")
        table.add_column("Notes")

        for r in sorted(our_results, key=lambda x: x.get("winner_acc", 0), reverse=True):
            table.add_row(
                r["name"][:40],
                f"{r['winner_acc']:.3f}" if r.get("winner_acc") else "—",
                f"{r['rps']:.3f}" if r.get("rps") else "—",
                "[green]Ours[/green]",
                r.get("note", ""),
            )

        for b in sorted(benchmarks, key=lambda x: x["winner_acc"], reverse=True):
            table.add_row(
                b["name"][:40],
                f"{b['winner_acc']:.3f}",
                f"{b['rps']:.3f}" if b.get("rps") else "—",
                "[blue]Published[/blue]",
                b["note"][:50],
            )
        Console().print(table)
    except ImportError:
        print(f"\n{title}:")
        for b in benchmarks:
            print(f"  {b['name']}: winner_acc={b['winner_acc']:.3f} | {b['note']}")
