# decomposer.py
"""
Stage 1 — Conservative Query Decomposition.

Philosophy: A bad split is worse than no split.
Only decompose when we have HIGH confidence the subtasks are independent.
"""

import re
from typing import List, Optional
from config import INTENT_KEYWORDS, MIN_SUBTASK_LENGTH, MAX_SUBTASKS


class SubTask:
    """A single atomic subtask extracted from a query."""
    def __init__(self, text: str, intent: str = "unknown", index: int = 0):
        self.text = text.strip()
        self.intent = intent
        self.index = index

    def to_dict(self) -> dict:
        return {"text": self.text, "intent": self.intent, "index": self.index}

    def __repr__(self):
        return f"SubTask({self.index}, {self.intent}, '{self.text[:50]}...')"


class Decomposer:
    """
    Conservative decomposition: split ONLY when signals are unambiguous.
    Three strategies, tried in order of reliability:
      1. Numbered lists   (1. ... 2. ... 3. ...)
      2. Semicolons       (...; ...; ...)
      3. "and" with       (Explain X and write code for Y)
         different intents
    If none apply, return the query as a single SubTask.
    """

    def __init__(self):
        self.intent_keywords = INTENT_KEYWORDS
        self.split_count = 0
        self.total_count = 0
        self.subtask_counts: List[int] = []

    # ── Public API ────────────────────────────────────────────────────

    def decompose(self, query: str) -> List[SubTask]:
        """
        Main entry point.
        Returns a list of SubTasks — always at least one.
        """
        self.total_count += 1

        subtasks = (
            self._split_numbered_list(query)
            or self._split_semicolons(query)
            or self._split_and_different_intents(query)
        )

        # Validate: every subtask must be non-trivial
        if subtasks and all(len(s.text) >= MIN_SUBTASK_LENGTH for s in subtasks):
            if len(subtasks) <= MAX_SUBTASKS:
                self.split_count += 1
                self.subtask_counts.append(len(subtasks))
                for i, st in enumerate(subtasks):
                    st.index = i
                return subtasks

        # No valid decomposition — return as single task
        self.subtask_counts.append(1)
        return [SubTask(query, intent=self._classify_intent(query), index=0)]

    def get_stats(self) -> dict:
        """Return decomposition statistics for reporting."""
        return {
            "total_queries":       self.total_count,
            "split_queries":       self.split_count,
            "split_rate":          round(self.split_count / max(self.total_count, 1), 3),
            "avg_subtasks":        round(
                sum(self.subtask_counts) / max(len(self.subtask_counts), 1), 2
            ),
            "subtask_distribution": self.subtask_counts,
        }

    # ── Strategy 1: Numbered Lists ────────────────────────────────────

    def _split_numbered_list(self, query: str) -> Optional[List[SubTask]]:
        """
        Split on '1. text 2. text' or '1) text 2) text'.
        Handles both newline-separated lists and inline lists on a single
        line (e.g. "1. What is X? 2. What is Y?"), since hand-typed
        queries rarely include real newlines between items.
        A numbered marker is only treated as a list item boundary when
        followed by whitespace then a capital letter or digit -- this
        avoids false positives like "version 2.5 of the library".
        """
        marker_pattern = r'(?:^|(?<=[\s\.\?\!]))(\d+)[\.\)]\s+(?=[A-Z0-9])'
        markers = list(re.finditer(marker_pattern, query))

        if len(markers) < 2:
            return None

        # Require strictly increasing item numbers (1, 2, 3...) so we don't
        # accidentally split on unrelated numbers that happen to look similar.
        numbers = [int(m.group(1)) for m in markers]
        if numbers != sorted(set(numbers)) or numbers[0] not in (0, 1):
            return None

        matches = []
        for i, m in enumerate(markers):
            start = m.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(query)
            matches.append(query[start:end])

        if len(matches) >= 2:
            subtasks = []
            for text in matches:
                clean = text.strip()
                if clean:
                    subtasks.append(SubTask(text=clean,
                                            intent=self._classify_intent(clean)))
            if len(subtasks) >= 2:
                return subtasks
        return None

    # ── Strategy 2: Semicolons ────────────────────────────────────────

    def _split_semicolons(self, query: str) -> Optional[List[SubTask]]:
        """Split on semicolons when each part looks like a standalone ask."""
        parts = [p.strip() for p in query.split(";")]

        if len(parts) >= 2 and all(len(p) >= MIN_SUBTASK_LENGTH for p in parts):
            if all(self._looks_independent(p) for p in parts):
                return [SubTask(text=p, intent=self._classify_intent(p))
                        for p in parts if p]
        return None

    # ── Strategy 3: "and" with Different Intents ──────────────────────

    def _split_and_different_intents(self, query: str) -> Optional[List[SubTask]]:
        """
        Split on 'and' ONLY when both sides carry clearly different intents.
        "Explain X and write code for Y"   → split  (knowledge + generation)
        "Explain X and its benefits"       → no split (same knowledge ask)
        "Compare X and Y"                  → no split (single analysis ask)
        """
        segments = re.split(r'\band\b', query, flags=re.IGNORECASE)

        if len(segments) != 2:
            return None

        left  = segments[0].strip()
        right = segments[1].strip()

        if len(left) < MIN_SUBTASK_LENGTH or len(right) < MIN_SUBTASK_LENGTH:
            return None

        left_intent  = self._classify_intent(left)
        right_intent = self._classify_intent(right)

        if (left_intent != "unknown"
                and right_intent != "unknown"
                and left_intent != right_intent
                and self._looks_independent(right)):
            return [
                SubTask(text=left,  intent=left_intent),
                SubTask(text=right, intent=right_intent),
            ]
        return None

    # ── Helpers ───────────────────────────────────────────────────────

    def _classify_intent(self, text: str) -> str:
        lower = text.lower().strip()
        for category, keywords in self.intent_keywords.items():
            for kw in keywords:
                if lower.startswith(kw) or f" {kw} " in f" {lower} ":
                    return category
        return "unknown"

    def _looks_independent(self, text: str) -> bool:
        lower = text.lower().strip()
        if re.match(r'^(what|who|how|why|when|where|which|can|does|is|are)\b', lower):
            return True
        for kws in self.intent_keywords.values():
            for kw in kws:
                if lower.startswith(kw):
                    return True
        if re.search(
            r'\b(write|explain|describe|create|list|compare|solve|'
            r'calculate|generate|implement|analyze)\b', lower
        ):
            return True
        return False