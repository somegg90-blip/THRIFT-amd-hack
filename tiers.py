#!/usr/bin/env python3
"""
Tier implementations for the 3-tier cascade.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests

from config import (
    FIREWORKS_MODEL,
    FIREWORKS_API_BASE,
    FIREWORKS_API_KEY,
    AVAILABLE_MODELS,
    IMAGE_MODELS,
    COMPLEX_INTENTS,
    get_sorted_models_by_intent,
    FIREWORKS_MAX_TOKENS_SIMPLE,
    FIREWORKS_MAX_TOKENS_COMPLEX,
    FIREWORKS_TEMPERATURE,
    FIREWORKS_TIMEOUT_SEC,
    FIREWORKS_MAX_RETRIES,
    FIREWORKS_RETRY_BACKOFF_BASE,
    TIER2_MAX_TOTAL_ATTEMPTS,
    HEDGE_PHRASES,
    REFUSAL_PHRASES,
    MIN_ANSWER_LENGTH,
)
from safe_math import extract_and_evaluate

logger = logging.getLogger("thrift.tiers")


@dataclass
class TierResult:
    answer: str
    confidence: float
    tier: int
    tokens_used: int
    raw_response: str = ""
    error: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════
# Tier 0 — Rule-Based Heuristic
# ══════════════════════════════════════════════════════════════════════

class TierZero:
    def __init__(self):
        self.handled_count = 0
        self.attempted_count = 0

    def try_handle(self, query: str) -> Optional[TierResult]:
        self.attempted_count += 1
        query_clean = (query or "").strip()
        if not query_clean:
            return None

        result = extract_and_evaluate(query_clean)
        if result is not None:
            self.handled_count += 1
            answer = str(int(result)) if float(result).is_integer() else f"{result:.6g}"
            return TierResult(answer=answer, confidence=0.95, tier=0, tokens_used=0)

        return None

    def get_stats(self) -> dict:
        return {
            "attempted": self.attempted_count,
            "handled": self.handled_count,
            "hit_rate": round(self.handled_count / max(self.attempted_count, 1), 3),
        }


# ══════════════════════════════════════════════════════════════════════
# Tier 1 — Local Model (DISABLED)
# ═════════════════════════════════════════════════════════════════════

class TierOne:
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self._load_error = "Tier 1 disabled for submission"

    def _ensure_loaded(self):
        raise RuntimeError(self._load_error)

    def try_handle(self, query: str, context: str = "") -> TierResult:
        return TierResult(answer="", confidence=0.0, tier=1, tokens_used=0, error=self._load_error)


# ══════════════════════════════════════════════════════════════════════
# Confidence estimation
# ═════════════════════════════════════════════════════════════════════

def _heuristic_confidence(answer: str) -> float:
    lower = answer.lower()
    score = 1.0

    for phrase in REFUSAL_PHRASES:
        if phrase in lower:
            return 0.05

    hedge_hits = sum(1 for phrase in HEDGE_PHRASES if phrase in lower)
    if hedge_hits > 0:
        score -= min(0.5, 0.2 * hedge_hits)

    word_count = len(answer.split())
    if word_count < 5:
        score -= 0.35
    elif word_count < 10:
        score -= 0.15

    if answer.count("?") > 2:
        score -= 0.15

    has_numbers = bool(re.search(r'\b\d+\b', answer))
    if has_numbers:
        score += 0.05

    has_code = "```" in answer or "def " in answer or "function" in lower
    if has_code:
        score += 0.08

    if word_count > 60:
        score += 0.05

    return round(max(0.0, min(1.0, score)), 3)


# ══════════════════════════════════════════════════════════════════════
# Tier 2 — Fireworks Remote API
# ══════════════════════════════════════════════════════════════════════

FINAL_ANSWER_MARKER = "FINAL ANSWER:"
_FINAL_ANSWER_RE = re.compile(r'final answer\s*[:\-]\s*', re.IGNORECASE)


def _extract_final_answer(raw: str) -> Optional[str]:
    if not raw:
        return None
    matches = list(_FINAL_ANSWER_RE.finditer(raw))
    if not matches:
        return None
    tail = raw[matches[-1].end():].strip()
    return tail or None


class TierTwo:
    def __init__(self, api_key: str = FIREWORKS_API_KEY, model: str = FIREWORKS_MODEL):
        self.api_key = api_key
        self.default_model = model
        self.call_count = 0
        self.total_tokens_used = 0
        self.failure_count = 0
        self.available_models = [m for m in AVAILABLE_MODELS if m not in IMAGE_MODELS]

    def _build_specialized_prompt(self, query: str, intent: str = "unknown"):
        lower = query.lower()
        is_complex = intent in COMPLEX_INTENTS

        base = (
            f'You are a highly accurate assistant. When done, output your conclusion on its own line starting exactly with "{FINAL_ANSWER_MARKER}", followed by nothing but the answer itself.'
        )
        if is_complex:
            system = base + " You may reason briefly first."
        else:
            system = base + " Skip reasoning and go straight to the final line."

        if any(k in lower for k in ["extract", "named entit", "entities", "label each"]):
            system += f' Your {FINAL_ANSWER_MARKER} line must contain ONLY a valid JSON array with "entity" and "type" fields.'
        elif "sentiment" in lower and any(k in lower for k in ["classify", "what is the sentiment", "sentiment of"]):
            system += f' Your {FINAL_ANSWER_MARKER} line must be exactly one word: Positive, Negative, Neutral, or Mixed.'
        elif any(k in lower for k in ["write a function", "write a python", "implement", "debug", "fix this", "find the error", "write code"]):
            system += f' Put ONLY the complete code after {FINAL_ANSWER_MARKER}, inside a ```python code block.'
        elif any(k in lower for k in ["calculate", "how many", "what is the total", "percent", "arithmetic", "word problem"]):
            system += f' Your {FINAL_ANSWER_MARKER} line must contain ONLY the final number.'
        elif any(k in lower for k in ["summarize", "summarise", "summary", "one sentence"]):
            system += f' Put ONLY the summary after {FINAL_ANSWER_MARKER}.'

        return system, query

    def _post_process_answer(self, raw_answer: str, query: str) -> str:
        lower_query = query.lower()
        core = _extract_final_answer(raw_answer)
        answer = core if core is not None else raw_answer.strip()

        if any(k in lower_query for k in ["extract", "named entit", "entities", "label each"]):
            candidate = re.sub(r'```[a-zA-Z]*\n?|```', '', answer).strip()
            start, end = candidate.find("["), candidate.rfind("]")
            if start != -1 and end != -1 and end > start:
                snippet = candidate[start:end + 1]
                try:
                    json.loads(snippet)
                    return snippet
                except (ValueError, TypeError):
                    pass
            return candidate or raw_answer.strip()

        if "sentiment" in lower_query and any(k in lower_query for k in ["classify", "what is the sentiment", "sentiment of"]):
            lower_ans = answer.lower()
            if "mixed" in lower_ans or ("positive" in lower_ans and "negative" in lower_ans):
                return "Mixed"
            if "positive" in lower_ans:
                return "Positive"
            if "negative" in lower_ans:
                return "Negative"
            if "neutral" in lower_ans:
                return "Neutral"
            return answer

        if any(k in lower_query for k in ["calculate", "how many", "what is the total", "percent", "arithmetic", "word problem"]):
            matches = re.findall(r'-?\d[\d,]*(?:\.\d+)?', answer)
            if matches:
                return matches[-1].replace(",", "")
            return answer

        if any(k in lower_query for k in ["write a function", "write a python", "implement", "debug", "fix this", "find the error", "write code", "code"]):
            if "```" in answer:
                start = answer.find("```")
                end = answer.find("```", start + 3)
                if end != -1:
                    block = answer[start+3:end].strip()
                    if block.lower().startswith("python"):
                        block = block[6:].strip()
                    return f"```python\n{block}\n```"

        if answer.startswith("```"):
            answer = re.sub(r'^```[a-zA-Z]*\n?|```$', '', answer).strip()

        return answer or raw_answer.strip()

    def try_handle(self, query: str, context: str = "", intent: str = "unknown") -> TierResult:
        if not self.api_key:
            return TierResult(answer="", confidence=0.0, tier=2, tokens_used=0, error="FW_API_KEY not set")

        models_to_try = get_sorted_models_by_intent(intent, self.available_models)
        if not models_to_try:
            return TierResult(answer="", confidence=0.0, tier=2, tokens_used=0, error="No available models")

        system_prompt, user_content = self._build_specialized_prompt(query, intent)
        if context:
            user_content = f"Context: {context}\n\n{user_content}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        max_tokens = FIREWORKS_MAX_TOKENS_COMPLEX if intent in COMPLEX_INTENTS else FIREWORKS_MAX_TOKENS_SIMPLE

        last_error = None
        total_attempts = 0

        for model_to_try in models_to_try:
            if total_attempts >= TIER2_MAX_TOTAL_ATTEMPTS:
                break
            for attempt in range(1, FIREWORKS_MAX_RETRIES + 1):
                if total_attempts >= TIER2_MAX_TOTAL_ATTEMPTS:
                    break
                total_attempts += 1
                self.call_count += 1
                try:
                    response = requests.post(
                        f"{FIREWORKS_API_BASE}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json={"model": model_to_try, "messages": messages, "max_tokens": max_tokens, "temperature": FIREWORKS_TEMPERATURE},
                        timeout=FIREWORKS_TIMEOUT_SEC,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        raw_answer = data["choices"][0]["message"]["content"].strip()
                        tokens_used = data.get("usage", {}).get("total_tokens", 0)
                        self.total_tokens_used += tokens_used
                        answer = self._post_process_answer(raw_answer, query)
                        return TierResult(answer=answer, confidence=0.9, tier=2, tokens_used=tokens_used, raw_response=str(data))

                    if response.status_code == 429:
                        last_error = "Rate limited (429)"
                        self._backoff(attempt)
                        continue
                    if response.status_code >= 500:
                        last_error = f"Server error ({response.status_code})"
                        self._backoff(attempt)
                        continue

                    last_error = f"API error {response.status_code}: {response.text[:100]}"
                    break

                except requests.exceptions.Timeout:
                    last_error = f"Timeout after {FIREWORKS_TIMEOUT_SEC}s"
                    self._backoff(attempt)
                except Exception as e:
                    last_error = f"Unexpected error: {e}"
                    self._backoff(attempt)

        self.failure_count += 1
        logger.error(f"[Tier 2] Exhausted {total_attempts} attempts: {last_error}")
        return TierResult(answer="", confidence=0.0, tier=2, tokens_used=0, error=last_error)

    def _backoff(self, attempt: int):
        delay = FIREWORKS_RETRY_BACKOFF_BASE ** attempt
        logger.warning(f"[Tier 2] Retrying in {delay:.1f}s (attempt {attempt})")
        time.sleep(delay)

    def get_stats(self) -> dict:
        return {"calls": self.call_count, "failures": self.failure_count, "total_tokens_used": self.total_tokens_used}