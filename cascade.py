"""
Stage 2 — Cascade Orchestrator.

Runs a single subtask through Tier 0 -> Tier 1 -> Tier 2, escalating
only when confidence falls below threshold. Tracks token usage and
per-tier statistics for the eval harness and demo dashboard.

Design principle: never crash on a single subtask failure. If every
tier fails, return the best (highest-confidence) answer seen so far
rather than an empty result — graceful degradation beats a hard error.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from config import CONFIDENCE_THRESHOLD, SKIP_TIER_1
from tiers import TierZero, TierOne, TierTwo, TierResult
from decomposer import SubTask

logger = logging.getLogger("thrift.cascade")

# ── Smart Routing ─────────────────────────────────────────────────────
# Intents that are too complex or slow for the tiny local model on 2 vCPU.
# We skip Tier 1 for these and go straight to the powerful Fireworks API (Tier 2).
HARD_INTENTS = {"generation", "computation", "analysis"}


@dataclass
class CascadeResult:
    """Final result for a single subtask after running through the cascade."""
    subtask_text: str
    answer: str
    confidence: float
    tier_used: int                 # which tier's answer was ultimately accepted
    tokens_used: int               # total tokens spent (only Tier 2 costs anything)
    tiers_attempted: List[int] = field(default_factory=list)
    degraded: bool = False         # True if no tier cleared the threshold
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "subtask": self.subtask_text,
            "answer": self.answer,
            "confidence": self.confidence,
            "tier_used": self.tier_used,
            "tokens_used": self.tokens_used,
            "tiers_attempted": self.tiers_attempted,
            "degraded": self.degraded,
            "errors": self.errors,
        }


class Cascade:
    """
    Orchestrates the 3-tier escalation for individual subtasks.
    Owns single instances of each tier so models are loaded once
    and reused across every query in a run.
    """

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = confidence_threshold
        
        # Smart Tier 1 initialization: skip entirely if configured or if we want to save time
        self.skip_tier_1 = SKIP_TIER_1
        self.tier0 = TierZero()
        self.tier1 = TierOne() if not self.skip_tier_1 else None
        self.tier2 = TierTwo()

        # Run-level stats, used by eval_harness / dashboard
        self.total_subtasks = 0
        self.tier_usage_counts = {0: 0, 1: 0, 2: 0}
        self.total_tokens_used = 0
        self.degraded_count = 0

    # ── Public API ────────────────────────────────────────────────────

    def run(self, subtask: SubTask, context: str = "") -> CascadeResult:
        """
        Run a single subtask through the cascade.
        Always returns a CascadeResult — never raises.
        """
        self.total_subtasks += 1
        query = subtask.text
        intent = subtask.intent
        tiers_attempted: List[int] = []
        errors: List[str] = []

        best_so_far: Optional[TierResult] = None

        def consider(result: Optional[TierResult]) -> bool:
            """Track the best result seen so far. Returns True if this
            result clears the confidence threshold (cascade can stop)."""
            nonlocal best_so_far
            if result is None:
                return False
            if result.error:
                errors.append(f"Tier {result.tier}: {result.error}")
            if best_so_far is None or result.confidence > best_so_far.confidence:
                best_so_far = result
            return result.confidence >= self.threshold

        # ── Tier 0: Instant Math/Heuristics ──
        tiers_attempted.append(0)
        r0 = self._safe_call(self.tier0.try_handle, query)
        if consider(r0):
            return self._finalize(subtask, best_so_far, tiers_attempted, errors)

        # ── Tier 1: Local Small Model ──
        # SMART ROUTING: Skip local model if task is "hard" (code/math/analysis) 
        # because the tiny model will fail or take too long anyway.
        if not self.skip_tier_1 and self.tier1 and intent not in HARD_INTENTS:
            tiers_attempted.append(1)
            r1 = self._safe_call(self.tier1.try_handle, query, context)
            if consider(r1):
                return self._finalize(subtask, best_so_far, tiers_attempted, errors)

        # ── Tier 2: Fireworks Remote API (last resort — costs tokens) ──
        tiers_attempted.append(2)
        # Pass the intent so Tier 2 can select the right model and use specialized prompts
        r2 = self._safe_call(self.tier2.try_handle, query, context, intent=intent)
        if consider(r2):
            return self._finalize(subtask, best_so_far, tiers_attempted, errors)

        # Nothing cleared the threshold — degrade gracefully to the best
        # answer we actually got, rather than returning nothing.
        result = self._finalize(subtask, best_so_far, tiers_attempted, errors)
        result.degraded = True
        self.degraded_count += 1
        logger.warning(
            f"[Cascade] No tier cleared threshold for subtask "
            f"'{query[:60]}...' — using best available (tier {result.tier_used}, "
            f"confidence {result.confidence:.2f})"
        )
        return result

    def run_all(self, subtasks: List[SubTask], context: str = "") -> List[CascadeResult]:
        """Run the cascade over every subtask of a decomposed query."""
        return [self.run(st, context) for st in subtasks]

    # ── Internals ─────────────────────────────────────────────────────

    def _safe_call(self, fn, *args, **kwargs) -> Optional[TierResult]:
        """
        Defense in depth: even though each tier already catches its own
        errors internally, this guards against any unexpected exception
        (e.g. a bug in tier code) from killing the whole cascade run.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"[Cascade] Unexpected error calling {fn}: {e}")
            return None

    def _finalize(
        self,
        subtask: SubTask,
        result: Optional[TierResult],
        tiers_attempted: List[int],
        errors: List[str],
    ) -> CascadeResult:
        if result is None:
            # Every tier returned None/failed outright — total failure.
            # Still return a well-formed result so callers never branch on None.
            self.tier_usage_counts[tiers_attempted[-1] if tiers_attempted else 1] += 1
            return CascadeResult(
                subtask_text=subtask.text,
                answer="",
                confidence=0.0,
                tier_used=tiers_attempted[-1] if tiers_attempted else -1,
                tokens_used=0,
                tiers_attempted=tiers_attempted,
                degraded=True,
                errors=errors or ["All tiers failed with no result"],
            )

        self.tier_usage_counts[result.tier] += 1
        self.total_tokens_used += result.tokens_used

        return CascadeResult(
            subtask_text=subtask.text,
            answer=result.answer,
            confidence=result.confidence,
            tier_used=result.tier,
            tokens_used=result.tokens_used,
            tiers_attempted=tiers_attempted,
            degraded=False,
            errors=errors,
        )

    # ── Reporting ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        total = max(self.total_subtasks, 1)
        return {
            "total_subtasks": self.total_subtasks,
            "tier_usage_counts": dict(self.tier_usage_counts),
            "tier_usage_pct": {
                tier: round(100 * count / total, 1)
                for tier, count in self.tier_usage_counts.items()
            },
            "total_tokens_used": self.total_tokens_used,
            "avg_tokens_per_subtask": round(self.total_tokens_used / total, 2),
            "degraded_count": self.degraded_count,
            "degraded_rate": round(self.degraded_count / total, 3),
            "tier0_stats": self.tier0.get_stats(),
            "tier2_stats": self.tier2.get_stats(),
        }