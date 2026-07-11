# thrift.py
"""
THRIFT — Main Agent.

Ties the three stages together into one callable pipeline:

    Query -> Decomposer -> Cascade (per subtask) -> Reassembler -> FinalAnswer

This is the single entry point used by app.py (demo server) and
eval_harness.py (threshold sweeping / scoring).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from config import CONFIDENCE_THRESHOLD
from decomposer import Decomposer, SubTask
from cascade import Cascade, CascadeResult
from reassembler import Reassembler, FinalAnswer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("thrift.agent")


@dataclass
class QueryRun:
    """Complete record of one query's trip through the pipeline.
    This is the object eval_harness.py scores against."""
    query: str
    final_answer: FinalAnswer
    subtasks: List[SubTask]
    cascade_results: List[CascadeResult]
    latency_sec: float
    tokens_used: int
    any_degraded: bool

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.final_answer.text,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "cascade_results": [r.to_dict() for r in self.cascade_results],
            "tokens_used": self.tokens_used,
            "latency_sec": round(self.latency_sec, 3),
            "any_degraded": self.any_degraded,
        }


class Thrift:
    """
    The THRIFT agent. Construct once, call .answer(query) repeatedly.
    Tiers' underlying models are loaded lazily on first use and reused
    across every subsequent call -- don't construct a new Thrift()
    per query, or you'll reload the local model every time.
    """

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.decomposer = Decomposer()
        self.cascade = Cascade(confidence_threshold=confidence_threshold)
        self.reassembler = Reassembler()

        self.total_queries = 0
        self.total_latency_sec = 0.0
        self.run_history: List[QueryRun] = []

    # ── Public API ────────────────────────────────────────────────────

    def answer(self, query: str, context: str = "", record_history: bool = True) -> QueryRun:
        """
        Run a single query through the full pipeline.
        Always returns a QueryRun -- never raises. A completely malformed
        or empty query still produces a well-formed (if degraded) result,
        so callers (app.py, eval_harness.py) never need to special-case
        exceptions from this method.
        """
        start = time.time()

        if not query or not query.strip():
            return self._empty_query_run(query, start)

        try:
            subtasks = self.decomposer.decompose(query)
        except Exception as e:
            logger.error(f"[Thrift] Decomposition failed, treating as single task: {e}")
            subtasks = [SubTask(query)]

        try:
            cascade_results = self.cascade.run_all(subtasks, context=context)
        except Exception as e:
            # Should not happen -- cascade.run() already guards every subtask --
            # but guard the whole-list call too, since a future change to
            # run_all() could introduce a list-level failure mode.
            logger.error(f"[Thrift] Cascade run_all failed unexpectedly: {e}")
            cascade_results = [
                CascadeResult(subtask_text=st.text, answer="", confidence=0.0,
                               tier_used=-1, tokens_used=0, tiers_attempted=[],
                               degraded=True, errors=[str(e)])
                for st in subtasks
            ]

        try:
            final_answer = self.reassembler.assemble(cascade_results)
        except Exception as e:
            logger.error(f"[Thrift] Reassembly failed: {e}")
            final_answer = FinalAnswer(
                text="An error occurred while assembling the response.",
                total_tokens_used=sum(r.tokens_used for r in cascade_results),
                tiers_used=[r.tier_used for r in cascade_results],
                subtask_count=len(cascade_results),
                any_degraded=True,
                degraded_subtasks=[r.subtask_text for r in cascade_results],
            )

        latency = time.time() - start
        self.total_queries += 1
        self.total_latency_sec += latency

        run = QueryRun(
            query=query,
            final_answer=final_answer,
            subtasks=subtasks,
            cascade_results=cascade_results,
            latency_sec=latency,
            tokens_used=final_answer.total_tokens_used,
            any_degraded=final_answer.any_degraded,
        )

        if record_history:
            self.run_history.append(run)

        return run

    def answer_text(self, query: str, context: str = "") -> str:
        """Convenience wrapper -- just the final text, for quick use in app.py."""
        return self.answer(query, context).final_answer.text

    # ── Internals ─────────────────────────────────────────────────────

    def _empty_query_run(self, query: str, start_time: float) -> QueryRun:
        empty_answer = FinalAnswer(
            text="Please provide a question or request.",
            total_tokens_used=0,
            tiers_used=[],
            subtask_count=0,
            any_degraded=False,
        )
        return QueryRun(
            query=query or "",
            final_answer=empty_answer,
            subtasks=[],
            cascade_results=[],
            latency_sec=time.time() - start_time,
            tokens_used=0,
            any_degraded=False,
        )

    # ── Reporting ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Aggregate run-level stats -- consumed by eval_harness.py and
        the demo dashboard for the 'token savings over time' visualization."""
        total = max(self.total_queries, 1)
        degraded = sum(1 for r in self.run_history if r.any_degraded)
        total_tokens = sum(r.tokens_used for r in self.run_history)
        naive_remote_estimate = self._estimate_naive_remote_cost()

        return {
            "total_queries": self.total_queries,
            "avg_latency_sec": round(self.total_latency_sec / total, 3),
            "total_tokens_used": total_tokens,
            "avg_tokens_per_query": round(total_tokens / total, 2),
            "degraded_queries": degraded,
            "degraded_rate": round(degraded / total, 3),
            "decomposer_stats": self.decomposer.get_stats(),
            "cascade_stats": self.cascade.get_stats(),
            "estimated_naive_remote_tokens": naive_remote_estimate,
            "estimated_token_savings_pct": round(
                100 * (1 - total_tokens / max(naive_remote_estimate, 1)), 1
            ) if naive_remote_estimate else 0.0,
        }

    def _estimate_naive_remote_cost(self) -> int:
        """
        Rough baseline for the demo's 'savings vs naive all-remote' comparison:
        assumes every subtask THRIFT actually ran would have cost roughly the
        same tokens as the average Tier-2 call observed so far. If Tier 2 was
        never called, falls back to a fixed estimate (~300 tokens/subtask)
        so the comparison still renders something meaningful pre-kickoff.
        """
        tier2_calls = self.cascade.tier2.call_count
        tier2_tokens = self.cascade.tier2.total_tokens_used
        avg_remote_tokens = (tier2_tokens / tier2_calls) if tier2_calls else 300

        total_subtasks = sum(len(r.subtasks) or 1 for r in self.run_history)
        return int(total_subtasks * avg_remote_tokens)

    def reset_stats(self):
        """Clear run history -- useful between eval_harness sweeps so one
        threshold's stats don't bleed into the next."""
        self.total_queries = 0
        self.total_latency_sec = 0.0
        self.run_history = []