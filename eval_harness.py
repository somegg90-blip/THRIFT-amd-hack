# eval_harness.py
"""
Threshold sweeper — proves the token-cost vs accuracy tradeoff instead
of guessing a threshold. This is the evidence behind the pitch: "we
measured the optimal point, we didn't guess it."

Accuracy proxy: since we don't have ground-truth grading at hackathon
time, each sample query carries `expected_keywords` -- a loose,
case-insensitive substring check used as a cheap correctness signal.
This is NOT a substitute for the real LLM-judge / leaderboard scoring
that will run at kickoff -- it's a local sanity proxy so you can sweep
thresholds *before* the real tasks are revealed, then re-run this same
harness against the real task set once it drops.

Usage:
    python3 eval_harness.py                  # full sweep, sample data
    python3 eval_harness.py --queries data/sample_queries.json
    python3 eval_harness.py --start 0.5 --end 0.9 --step 0.1
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from config import THRESHOLD_SWEEP_START, THRESHOLD_SWEEP_END, THRESHOLD_SWEEP_STEP
from thrift import Thrift

logging.basicConfig(level=logging.WARNING)  # keep sweep output clean
logger = logging.getLogger("thrift.eval")


@dataclass
class ThresholdResult:
    threshold: float
    total_queries: int
    total_tokens: int
    avg_tokens_per_query: float
    accuracy: float                  # fraction of queries matching expected keywords
    degraded_rate: float
    avg_latency_sec: float

    def to_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 2),
            "total_tokens": self.total_tokens,
            "avg_tokens_per_query": round(self.avg_tokens_per_query, 2),
            "accuracy": round(self.accuracy, 3),
            "degraded_rate": round(self.degraded_rate, 3),
            "avg_latency_sec": round(self.avg_latency_sec, 3),
        }


def load_queries(path: str) -> List[dict]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        queries = data.get("queries", [])
        if not queries:
            raise ValueError("No queries found in file")
        return queries
    except FileNotFoundError:
        logger.error(f"Query file not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Malformed JSON in {path}: {e}")
        sys.exit(1)


def score_answer(answer_text: str, expected_keywords: List[str]) -> bool:
    """
    Loose correctness proxy: does the answer contain at least one expected
    keyword (case-insensitive)? If a query has no expected_keywords listed
    (e.g. open-ended creative tasks), it's counted as correct by default --
    there's no ground truth to check against, so it shouldn't penalize the score.
    """
    if not expected_keywords:
        return True
    lower = answer_text.lower()
    return any(kw.lower() in lower for kw in expected_keywords)


def run_single_threshold(threshold: float, queries: List[dict]) -> ThresholdResult:
    agent = Thrift(confidence_threshold=threshold)
    correct = 0

    for q in queries:
        run = agent.answer(q["text"])
        if score_answer(run.final_answer.text, q.get("expected_keywords", [])):
            correct += 1

    stats = agent.get_stats()
    total = max(stats["total_queries"], 1)

    return ThresholdResult(
        threshold=threshold,
        total_queries=stats["total_queries"],
        total_tokens=stats["total_tokens_used"],
        avg_tokens_per_query=stats["avg_tokens_per_query"],
        accuracy=correct / total,
        degraded_rate=stats["degraded_rate"],
        avg_latency_sec=stats["avg_latency_sec"],
    )


def sweep(
    queries: List[dict],
    start: float = THRESHOLD_SWEEP_START,
    end: float = THRESHOLD_SWEEP_END,
    step: float = THRESHOLD_SWEEP_STEP,
) -> List[ThresholdResult]:
    results = []
    threshold = start
    while threshold <= end + 1e-9:  # float-safe upper bound
        print(f"Running threshold {threshold:.2f} ...", file=sys.stderr)
        results.append(run_single_threshold(round(threshold, 2), queries))
        threshold += step
    return results


def find_optimal(results: List[ThresholdResult], min_accuracy: float = 0.8) -> Optional[ThresholdResult]:
    """
    Pick the threshold that minimizes tokens while staying at or above
    the accuracy floor -- this is the actual number you submit, backed
    by measurement rather than a guess.
    """
    candidates = [r for r in results if r.accuracy >= min_accuracy]
    if not candidates:
        logger.warning(
            f"No threshold reached {min_accuracy:.0%} accuracy on this sample set -- "
            f"falling back to the highest-accuracy threshold available."
        )
        return max(results, key=lambda r: r.accuracy) if results else None
    return min(candidates, key=lambda r: r.avg_tokens_per_query)


def print_table(results: List[ThresholdResult]):
    print("\n" + "=" * 78)
    print(f"{'Threshold':>10} | {'Avg Tokens':>11} | {'Accuracy':>9} | {'Degraded%':>10} | {'Latency(s)':>10}")
    print("-" * 78)
    for r in results:
        print(f"{r.threshold:>10.2f} | {r.avg_tokens_per_query:>11.2f} | "
              f"{r.accuracy:>9.1%} | {r.degraded_rate:>10.1%} | {r.avg_latency_sec:>10.3f}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="THRIFT threshold sweep")
    parser.add_argument("--queries", default="data/sample_queries.json",
                         help="Path to JSON file with a 'queries' list")
    parser.add_argument("--start", type=float, default=THRESHOLD_SWEEP_START)
    parser.add_argument("--end", type=float, default=THRESHOLD_SWEEP_END)
    parser.add_argument("--step", type=float, default=THRESHOLD_SWEEP_STEP)
    parser.add_argument("--min-accuracy", type=float, default=0.8,
                         help="Minimum acceptable accuracy when picking the optimal threshold")
    parser.add_argument("--output", default="eval_results.json",
                         help="Where to save the full sweep results as JSON")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries from {args.queries}")

    results = sweep(queries, args.start, args.end, args.step)
    print_table(results)

    optimal = find_optimal(results, min_accuracy=args.min_accuracy)
    if optimal:
        print(f"RECOMMENDED THRESHOLD: {optimal.threshold:.2f}")
        print(f"  -> {optimal.avg_tokens_per_query:.2f} avg tokens/query "
              f"at {optimal.accuracy:.1%} accuracy")
        print(f"  -> Set CONFIDENCE_THRESHOLD = {optimal.threshold:.2f} in config.py\n")

    with open(args.output, "w") as f:
        json.dump({
            "sweep_results": [r.to_dict() for r in results],
            "recommended_threshold": optimal.threshold if optimal else None,
            "min_accuracy_floor": args.min_accuracy,
        }, f, indent=2)
    print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()