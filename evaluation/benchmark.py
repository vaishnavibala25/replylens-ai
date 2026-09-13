"""
benchmark.py
------------
Two independent evaluation runs:

1. run_benchmark(): pulls every email in the "evaluation" split, retrieves
   few-shot examples ONLY from the "reference" split (no leakage), generates
   a response, evaluates it, and aggregates system-wide + per-category
   scores. This answers "how good is the system overall?"

2. run_calibration(): loads evaluation/calibration_examples.json (pre-written
   incoming_email + generated_response + human_rating triples) and runs each
   through the ReplyLens evaluator (skipping generation, since the response
   is already fixed), then computes Spearman correlation between human
   ratings and the automated overall_score. This answers "does the metric
   track human judgment?" — see README for the honest caveats about sample
   size.

Both are also runnable standalone: `python -m evaluation.benchmark`
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from services.llm_service import LLMService
from services.retrieval_service import RetrievalService
from services.generation_service import generate_response
from services.evaluation_service import evaluate_response
from evaluation.metrics import aggregate_scores, spearman_correlation

DATA_PATH = Path(__file__).parent.parent / "data" / "emails.csv"
CALIBRATION_PATH = Path(__file__).parent / "calibration_examples.json"


def run_benchmark(llm_service: LLMService, limit: int = None, top_k: int = 3,
                   progress_callback=None) -> dict:
    df = pd.read_csv(DATA_PATH)
    retrieval = RetrievalService(df)
    eval_df = df[df["split"] == "evaluation"].reset_index(drop=True)
    if limit:
        eval_df = eval_df.head(limit)

    results = []
    for i, row in eval_df.iterrows():
        incoming = row["incoming_email"]
        retrieved = retrieval.retrieve(incoming, top_k=top_k)
        gen = generate_response(incoming, retrieved, llm_service)
        evaluation = evaluate_response(incoming, retrieved, gen["response_text"], llm_service)
        evaluation["category"] = row["category"]
        evaluation["email_id"] = row["id"]
        evaluation["is_demo"] = gen["is_demo"]
        results.append(evaluation)
        if progress_callback:
            progress_callback(i + 1, len(eval_df))

    summary = aggregate_scores(results)
    summary["is_demo"] = not llm_service.is_configured
    return {"summary": summary, "results": results}


def run_calibration(llm_service: LLMService) -> dict:
    with open(CALIBRATION_PATH, encoding="utf-8") as f:
        calibration = json.load(f)

    cases = calibration["cases"]
    human_ratings = []
    automated_scores = []
    per_case = []

    for case in cases:
        # Scale human 1-5 rating to a 0-100 axis for correlation comparability.
        evaluation = evaluate_response(
            case["incoming_email"], [], case["generated_response"], llm_service
        )
        automated = evaluation["overall_score"]
        human_ratings.append(case["human_rating"])
        automated_scores.append(automated)
        per_case.append({
            "id": case["id"],
            "human_rating": case["human_rating"],
            "automated_score": automated,
            "notes": case["notes"],
        })

    rho = spearman_correlation(human_ratings, automated_scores)
    return {
        "n_cases": len(cases),
        "spearman_rho": rho,
        "per_case": per_case,
        "is_demo": not llm_service.is_configured,
    }


if __name__ == "__main__":
    llm = LLMService()
    print("Running system benchmark on evaluation split...")
    bench = run_benchmark(llm)
    print(json.dumps(bench["summary"], indent=2))

    print("\nRunning human-calibration correlation...")
    cal = run_calibration(llm)
    print(f"n_cases={cal['n_cases']}  spearman_rho={cal['spearman_rho']}  demo_mode={cal['is_demo']}")
