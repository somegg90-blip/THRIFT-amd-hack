# reassembler.py
"""
Stage 3 — Response Reassembly.

Takes per-subtask CascadeResults and produces one coherent final answer.
Single subtasks: pass through untouched.
Multiple subtasks: clearly numbered so the user always knows which
answer maps to which question.
"""

import logging
from dataclasses import dataclass, field
from typing import List

from cascade import CascadeResult

logger = logging.getLogger("thrift.reassembler")


@dataclass
class FinalAnswer:
    text: str
    total_tokens_used: int
    tiers_used: List[int]
    subtask_count: int
    any_degraded: bool
    degraded_subtasks: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "total_tokens_used": self.total_tokens_used,
            "tiers_used": self.tiers_used,
            "subtask_count": self.subtask_count,
            "any_degraded": self.any_degraded,
            "degraded_subtasks": self.degraded_subtasks,
        }


class Reassembler:

    def assemble(self, results: List[CascadeResult]) -> FinalAnswer:
        if not results:
            return FinalAnswer(
                text="No answer could be generated.",
                total_tokens_used=0, tiers_used=[], subtask_count=0,
                any_degraded=True, degraded_subtasks=["<no subtasks>"],
            )

        if len(results) == 1:
            r = results[0]
            text = r.answer if r.answer.strip() else self._fallback_text(r)
            return FinalAnswer(
                text=text, total_tokens_used=r.tokens_used, tiers_used=[r.tier_used],
                subtask_count=1, any_degraded=r.degraded,
                degraded_subtasks=[r.subtask_text] if r.degraded else [],
            )

        # Multiple subtasks — number each answer so it's crystal clear
        # which answer belongs to which question.
        parts = []
        tiers_used = []
        total_tokens = 0
        degraded_subtasks = []

        for i, r in enumerate(results, 1):
            tiers_used.append(r.tier_used)
            total_tokens += r.tokens_used
            if r.degraded:
                degraded_subtasks.append(r.subtask_text)

            answer = r.answer.strip() if r.answer.strip() else self._fallback_text(r)

            # Trim the original subtask to a short label (max 60 chars)
            label = r.subtask_text.strip()
            if len(label) > 60:
                label = label[:57] + "..."

            parts.append(f"{i}. {label}\n→ {answer}")

        combined = "\n\n".join(parts)

        return FinalAnswer(
            text=combined, total_tokens_used=total_tokens, tiers_used=tiers_used,
            subtask_count=len(results), any_degraded=any(r.degraded for r in results),
            degraded_subtasks=degraded_subtasks,
        )

    def _fallback_text(self, result: CascadeResult) -> str:
        logger.warning(f"[Reassembler] Empty answer for '{result.subtask_text[:60]}'")
        return f'[Unable to generate a confident answer for: "{result.subtask_text}"]'