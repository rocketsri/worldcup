"""Results ledger with Rich table display and JSON persistence."""
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from evaluation.metrics import EvalResult

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LEDGER_PATH = RESULTS_DIR / "leaderboard.json"


@dataclass
class ExperimentResult:
    exp_id: str
    name: str
    feature_set: str
    model_type: str
    llm_model: str | None
    quick_winner_acc: float | None = None
    full_winner_acc: float | None = None
    full_exact_acc: float | None = None
    full_mae: float | None = None
    full_rps: float | None = None
    n_games: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    hypothesis: str = ""


class Leaderboard:
    def __init__(self):
        self.results: dict[str, ExperimentResult] = {}
        self._load()

    def _load(self):
        if LEDGER_PATH.exists():
            data = json.loads(LEDGER_PATH.read_text())
            for d in data:
                r = ExperimentResult(**d)
                self.results[r.exp_id] = r

    def save(self, result: ExperimentResult):
        self.results[result.exp_id] = result
        LEDGER_PATH.write_text(
            json.dumps([asdict(r) for r in self.results.values()], indent=2)
        )

    def display(self):
        try:
            from rich.table import Table
            from rich.console import Console
            table = Table(title="Experiment Leaderboard", show_lines=True)
            table.add_column("Rank", style="bold")
            table.add_column("ID")
            table.add_column("Name")
            table.add_column("Quick Acc", justify="right")
            table.add_column("Full Acc", justify="right")
            table.add_column("Exact%", justify="right")
            table.add_column("MAE", justify="right")
            table.add_column("RPS↓", justify="right")
            table.add_column("Status")

            sorted_results = sorted(
                self.results.values(),
                key=lambda r: (r.full_winner_acc or r.quick_winner_acc or -1),
                reverse=True,
            )
            for rank, r in enumerate(sorted_results, 1):
                status = "✓" if r.full_winner_acc is not None else (
                    "~" if r.quick_winner_acc is not None else "✗" if r.error else "…"
                )
                table.add_row(
                    str(rank),
                    r.exp_id,
                    r.name[:35],
                    f"{r.quick_winner_acc:.3f}" if r.quick_winner_acc is not None else "—",
                    f"{r.full_winner_acc:.3f}" if r.full_winner_acc is not None else "—",
                    f"{r.full_exact_acc:.3f}" if r.full_exact_acc is not None else "—",
                    f"{r.full_mae:.3f}" if r.full_mae is not None else "—",
                    f"{r.full_rps:.3f}" if r.full_rps is not None else "—",
                    status,
                )
            Console().print(table)
        except ImportError:
            for r in self.results.values():
                print(f"{r.exp_id:4s} | {r.name:35s} | quick={r.quick_winner_acc} full={r.full_winner_acc}")

    def to_summary_str(self) -> str:
        lines = ["Experiment Leaderboard:"]
        for r in sorted(self.results.values(), key=lambda x: x.exp_id):
            lines.append(
                f"  {r.exp_id}: {r.name} | quick={r.quick_winner_acc} "
                f"full={r.full_winner_acc} rps={r.full_rps} error={r.error}"
            )
        return "\n".join(lines)
