"""Global config — paths, model IDs, API clients."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Model IDs
MODELS = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
    "gpt4o": "gpt-4o",
    "o3mini": "o3-mini",
}

# Concurrency
LLM_SEMAPHORE_LIMIT = 6

# Quick eval: 2018 WC, Full eval: 2022 WC
QUICK_EVAL_YEAR = 2018
FULL_EVAL_YEAR = 2022


def make_clients():
    import anthropic
    from openai import OpenAI
    ac = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    oc = OpenAI(api_key=OPENAI_API_KEY)
    return ac, oc
