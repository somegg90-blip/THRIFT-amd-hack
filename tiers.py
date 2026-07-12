#!/usr/bin/env python3
"""
Tier implementations for the 3-tier cascade.

Tier 0: Rule-based heuristics  — instant, free, pattern matching.
Tier 1: Small local model      — free (local tokens = 0 in scoring).
Tier 2: Fireworks remote API   — paid, highest quality, last resort.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests

from config import (
    LOCAL_MODEL_NAME,
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
    LOCAL_MAX_NEW_TOKENS,
    HEDGE_PHRASES,
    REFUSAL_PHRASES,
    MIN_ANSWER_LENGTH,
)
from safe_math import extract_and_evaluate

logger = logging.getLogger("thrift.tiers")


@dataclass
class TierResult:
    """Result from any tier."""
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
    """
    Instant pattern matching for trivially simple queries.
    No model, no tokens, no latency.
    Handles: basic arithmetic (safely, via AST — no eval()).
    """

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
            return TierResult(
                answer=answer,
                confidence=0.95,
                tier=0,
                tokens_used=0,
            )

        return None

    def get_stats(self) -> dict:
        return {
            "attempted": self.attempted_count,
            "handled": self.handled_count,
            "hit_rate": round(self.handled_count / max(self.attempted_count, 1), 3),
        }


# ══════════════════════════════════════════════════════════════════════
# Tier 1 — Local Small Model (SKIPPED in production to avoid timeouts)
# ══════════════════════════════════════════════════════════════════════

class ModelLoadError(RuntimeError):
    """Raised when the local model fails to load."""


class TierOne:
    """
    Small local model with confidence self-evaluation.
    All tokens are FREE (local = 0 in scoring).
    """

    def __init__(self, model_name: str = LOCAL_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False
        self._load_error: Optional[str] = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        if self._load_error:
            raise ModelLoadError(self._load_error)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            self._load_error = f"Missing dependency: {e}"
            raise ModelLoadError(self._load_error)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[Tier 1] Loading {self.model_name} on {self.device} ...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            from config import USE_4BIT
            model_kwargs = {"trust_remote_code": True}

            if USE_4BIT:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["device_map"] = "auto"
                logger.info("[Tier 1] Loading in 4-bit quantized mode")
            elif self.device == "cuda":
                model_kwargs["dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
                logger.info("[Tier 1] Loading in float16 mode (GPU)")
            else:
                model_kwargs["dtype"] = torch.float32
                logger.info("[Tier 1] Loading in float32 mode (CPU)")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs,
            )
            if self.device == "cpu" and not USE_4BIT:
                self.model = self.model.to("cpu")

            self.model.eval()
            self._loaded = True
            logger.info("[Tier 1] Model loaded successfully.")
        except Exception as e:
            self._load_error = f"Failed to load {self.model_name}: {e}"
            logger.error(f"[Tier 1] {self._load_error}")
            raise ModelLoadError(self._load_error)

    def try_handle(self, query: str, context: str = "") -> TierResult:
        if not query or not query.strip():
            return TierResult(answer="", confidence=0.0, tier=1, tokens_used=0,
                               error="Empty query")

        try:
            self._ensure_loaded()
        except ModelLoadError as e:
            return TierResult(answer="", confidence=0.0, tier=1, tokens_used=0,
                               error=str(e))

        try:
            is_complex = self._is_complex_query(query)
            prompt = self._build_prompt(query, context)
            raw = self._generate(prompt, complex_mode=is_complex)
            answer = self._clean_output(raw)

            if not answer or len(answer) < MIN_ANSWER_LENGTH:
                return TierResult(answer=answer, confidence=0.05, tier=1,
                                   tokens_used=0, raw_response=raw,
                                   error="Empty or near-empty generation")

            conf = self._estimate_confidence(query, answer)
            return TierResult(
                answer=answer, confidence=conf, tier=1,
                tokens_used=0, raw_response=raw,
            )
        except Exception as e:
            logger.warning(f"[Tier 1] Generation failed: {e}")
            return TierResult(answer="", confidence=0.0, tier=1, tokens_used=0,
                               error=f"Generation error: {e}")

    def _build_prompt(self, query: str, context: str = "") -> str:
        is_complex = self._is_complex_query(query)

        if is_complex:
            system = (
                "You are an expert assistant with deep knowledge across all domains. "
                "For complex questions, think through the problem step by step before "
                "giving your final answer. Show your reasoning process. "
                "Be thorough, precise, and confident. Never refuse to attempt a question."
            )
            user_content = (
                f"Think through this carefully step by step, then give a clear answer:\n\n"
                f"{query}"
            )
        else:
            system = (
                "You are a knowledgeable assistant. Answer directly, accurately, "
                "and completely. Be specific — give exact facts, dates, names, and "
                "numbers where relevant. Never refuse to attempt a question."
            )
            user_content = query

        messages = [{"role": "system", "content": system}]
        if context:
            messages.append({"role": "user",
                              "content": f"Context: {context}\n\n{user_content}"})
        else:
            messages.append({"role": "user", "content": user_content})

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _is_complex_query(self, query: str) -> bool:
        lower = query.lower()
        high_value_complex = [
            'write a', 'implement', 'code for', 'function that',
            'function to', 'algorithm', 'program that', 'script that',
            'debug', 'fix this code', 'optimize this',
            'prove that', 'derive', 'solve for', 'given that',
        ]
        return any(sig in lower for sig in high_value_complex)

    def _generate(self, prompt: str, max_new_tokens: int = LOCAL_MAX_NEW_TOKENS,
                  complex_mode: bool = False) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            if complex_mode:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=3,
                    repetition_penalty=1.15,
                    length_penalty=1.0,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _clean_output(self, raw: str) -> str:
        answer = (raw or "").strip()
        for marker in ["\nHuman:", "\nAssistant:", "\nUser:", "<|im_end|>",
                       "<|im_start|>", "</s>", "<|endoftext|>"]:
            idx = answer.find(marker)
            if idx != -1:
                answer = answer[:idx]
        return answer.strip()

    def _estimate_confidence(self, query: str, answer: str) -> float:
        return self._heuristic_confidence(answer)

    def _heuristic_confidence(self, answer: str) -> float:
        return _heuristic_confidence(answer)


# Module-level so both Tier 1 and Tier 2 score answers the same way,
# with zero extra token cost — it's pure text analysis on an
# already-generated answer, not another model call.
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

    has_steps = bool(re.search(
        r'(step \d|first[,:]|second[,:]|third[,:]|finally[,:]|\d+\.\s)',
        lower
    ))
    if has_steps:
        score += 0.05

    if word_count > 60:
        score += 0.05

    return round(max(0.0, min(1.0, score)), 3)


# ══════════════════════════════════════════════════════════════════════
# Tier 2 — Fireworks Remote API
# Smart model routing + a machine-parseable answer contract + bounded
# retries, so token spend and worst-case latency both stay predictable.
# ══════════════════════════════════════════════════════════════════════

class FireworksAPIError(RuntimeError):
    """Raised when the Fireworks API call fails after all retries."""


FINAL_ANSWER_MARKER = "FINAL ANSWER:"
_FINAL_ANSWER_RE = re.compile(r'final answer\s*[:\-]\s*', re.IGNORECASE)


def _extract_final_answer(raw: str) -> Optional[str]:
    """
    Pull out everything after the LAST 'FINAL ANSWER:' marker, if the
    model included one. This is the primary extraction path: far more
    reliable than guessing which number/word in a free-form explanation
    was the intended answer, and it's what lets complex tasks reason a
    little without breaking downstream parsing.
    """
    if not raw:
        return None
    matches = list(_FINAL_ANSWER_RE.finditer(raw))
    if not matches:
        return None
    tail = raw[matches[-1].end():].strip()
    return tail or None


class TierTwo:
    """
    Remote API call to Fireworks with:
      - Smart model selection (accuracy-first for hard tasks,
        efficiency-first for easy tasks) — drawn ONLY from the models
        actually present in this run's ALLOWED_MODELS.
      - A "FINAL ANSWER:" contract so extraction doesn't have to guess.
      - Per-complexity token budgets (less waste on simple tasks).
      - A hard cap on total attempts per subtask across every
        model+retry combination, so one bad subtask can't eat the
        whole run's time budget.
    """

    def __init__(self, api_key: str = FIREWORKS_API_KEY, model: str = FIREWORKS_MODEL):
        self.api_key = api_key
        self.default_model = model
        self.call_count = 0
        self.total_tokens_used = 0
        self.failure_count = 0

        # Single source of truth: config.AVAILABLE_MODELS (parsed once
        # from ALLOWED_MODELS), not a second env-var parse here.
        self.available_models = [m for m in AVAILABLE_MODELS if m not in IMAGE_MODELS]

        # If the harness gave us an image-capable model this run, keep
        # its name separately so image prompts can be routed to the
        # dedicated /images/generations endpoint instead of chat
        # completions (which would just describe the image in text and
        # fail any grading that expects an actual image).
        self.image_model = next((m for m in AVAILABLE_MODELS if m in IMAGE_MODELS), None)

    # ── Prompt construction ───────────────────────────────────────────

    def _build_specialized_prompt(self, query: str, intent: str = "unknown"):
        """
        Build a system prompt that always funnels down to one
        machine-parseable 'FINAL ANSWER:' line, with format constraints
        layered on for recognizable task shapes (JSON extraction,
        sentiment, code, arithmetic, summary). Complex intents are
        allowed brief reasoning first; simple intents skip straight to
        the answer, since forcing zero reasoning on hard tasks trades
        accuracy for tokens we can't afford to lose.
        """
        lower = query.lower()
        is_complex = intent in COMPLEX_INTENTS

        base = (
            "You are a highly accurate assistant. Correctness matters far "
            "more than brevity. When you are done, output your conclusion "
            f'on its own line starting exactly with "{FINAL_ANSWER_MARKER}", '
            "followed by nothing but the answer itself."
        )
        if is_complex:
            system = base + (
                " You may reason briefly first (a few sentences or a short "
                "worked calculation is fine) — but don't pad it, get to the "
                "final line efficiently."
            )
        else:
            system = base + (
                " This is a simple, direct question: skip any reasoning and "
                "go straight to the final line."
            )

        if any(k in lower for k in ["extract", "named entit", "entities", "label each"]):
            system += (
                f' Your {FINAL_ANSWER_MARKER} line must contain ONLY a valid JSON '
                'array of objects with "entity" and "type" fields, e.g. '
                '[{"entity": "Maria", "type": "PERSON"}]. No markdown fences.'
            )
        elif "sentiment" in lower and any(
            k in lower for k in ["classify", "what is the sentiment", "sentiment of"]
        ):
            system += (
                f' Your {FINAL_ANSWER_MARKER} line must be exactly one word: '
                'Positive, Negative, Neutral, or Mixed.'
            )
        elif any(k in lower for k in [
            "write a function", "write a python", "implement", "debug",
            "fix this", "find the error", "write code",
        ]):
            system += (
                f' Put ONLY the complete, corrected code after {FINAL_ANSWER_MARKER}, '
                'inside a ```python code block.'
            )
        elif any(k in lower for k in [
            "calculate", "how many", "what is the total", "percent",
            "arithmetic", "word problem",
        ]):
            system += (
                f' Your {FINAL_ANSWER_MARKER} line must contain ONLY the final '
                'number — plain digits, no units, no commas, no LaTeX.'
            )
        elif any(k in lower for k in ["summarize", "summarise", "summary", "one sentence"]):
            system += f' Put ONLY the summary after {FINAL_ANSWER_MARKER}.'

        return system, query

    # ── Post-processing ───────────────────────────────────────────────

    @staticmethod
    def _normalize_sentiment(text: str) -> Optional[str]:
        if not text:
            return None
        lower = text.strip().lower()
        for label in ("positive", "negative", "neutral", "mixed"):
            if lower == label:
                return label.capitalize()
        has_pos = "positive" in lower
        has_neg = "negative" in lower
        has_mix = "mixed" in lower or (has_pos and has_neg)
        has_neu = "neutral" in lower
        if has_mix:
            return "Mixed"
        if has_pos and not has_neg:
            return "Positive"
        if has_neg and not has_pos:
            return "Negative"
        if has_neu:
            return "Neutral"
        return None

    @staticmethod
    def _extract_number(text: str) -> Optional[str]:
        if not text:
            return None
        # (?:\.\d+)? instead of \.?\d* — a decimal point only counts as
        # part of the number when digits follow it, so a sentence-ending
        # period right after a number ("...is 36.") isn't swallowed in.
        anchored = re.search(
            r'(?:answer|result|total)\D{0,10}(-?\d[\d,]*(?:\.\d+)?)', text, re.IGNORECASE
        )
        if anchored:
            return anchored.group(1).replace(",", "")
        matches = re.findall(r'-?\d[\d,]*(?:\.\d+)?', text)
        if matches:
            return matches[-1].replace(",", "")
        return None

    @staticmethod
    def _extract_code(text: str) -> Optional[str]:
        if not text or "```" not in text:
            return None
        start = text.find("```")
        end = text.find("```", start + 3)
        if end == -1:
            return None
        block = text[start + 3:end].strip()
        if block.lower().startswith("python"):
            block = block[6:].strip()
        return f"```python\n{block}\n```" if block else None

    def _post_process_answer(self, raw_answer: str, query: str) -> str:
        """
        Extract the core answer using the FINAL ANSWER contract first,
        then apply light task-specific cleanup on that (much smaller,
        safer) string. Falls back to scanning the full raw answer only
        when the model didn't follow the marker instruction.
        """
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

        if "sentiment" in lower_query and any(
            k in lower_query for k in ["classify", "what is the sentiment", "sentiment of"]
        ):
            return (
                self._normalize_sentiment(answer)
                or self._normalize_sentiment(raw_answer)
                or answer
            )

        if any(k in lower_query for k in [
            "calculate", "how many", "what is the total", "percent",
            "arithmetic", "word problem",
        ]):
            return self._extract_number(answer) or self._extract_number(raw_answer) or answer

        if any(k in lower_query for k in [
            "write a function", "write a python", "implement", "debug",
            "fix this", "find the error", "write code", "code",
        ]):
            return self._extract_code(answer) or self._extract_code(raw_answer) or answer

        if answer.startswith("```"):
            answer = re.sub(r'^```[a-zA-Z]*\n?|```$', '', answer).strip()

        return answer or raw_answer.strip()

    # ── Image generation ──────────────────────────────────────────────

    def _is_image_prompt(self, query: str) -> bool:
        """Detects a request to generate an image, as opposed to merely
        discussing or explaining something visual in text."""
        lower = query.lower()
        image_keywords = [
            "generate an image", "generate a picture", "create an image",
            "create a picture", "draw a", "draw an", "paint a", "paint an",
            "sketch a", "sketch an", "make a picture", "make an image",
            "render an image", "produce an image",
        ]
        return any(k in lower for k in image_keywords)

    def _handle_image_generation(self, query: str) -> Optional[TierResult]:
        """
        Calls the Fireworks image-generation endpoint instead of
        chat/completions. Bounded to 2 attempts (image calls are
        effectively all-or-nothing — no point retrying a malformed
        response) so a stuck image call can't eat the time budget the
        text path is already protected from. Returns None on total
        failure so the caller can fall back to a text answer rather
        than returning nothing.
        """
        if not self.image_model:
            return None  # no image-capable model in this run's ALLOWED_MODELS

        last_error = None
        for attempt in range(1, 3):
            self.call_count += 1
            try:
                response = requests.post(
                    f"{FIREWORKS_API_BASE}/images/generations",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.image_model, "prompt": query, "n": 1},
                    timeout=FIREWORKS_TIMEOUT_SEC,
                )

                if response.status_code == 200:
                    data = response.json()
                    items = data.get("data", [])
                    item = items[0] if items else {}
                    if item.get("url"):
                        img_ref = item["url"]
                    elif item.get("b64_json"):
                        img_ref = f"data:image/png;base64,{item['b64_json']}"
                    else:
                        img_ref = None

                    if img_ref:
                        logger.info(
                            f"[Tier 2] Generated image with {self.image_model.split('/')[-1]}"
                        )
                        return TierResult(
                            answer=f"![generated image]({img_ref})",
                            confidence=0.9,
                            tier=2,
                            tokens_used=0,  # image calls don't consume text tokens
                            raw_response=str(data),
                        )
                    last_error = "Image response had no url/b64_json"
                    break  # malformed success response — retrying won't help

                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"Image API error ({response.status_code})"
                    self._backoff(attempt)
                    continue

                last_error = f"Image API error {response.status_code}: {response.text[:100]}"
                break

            except requests.exceptions.Timeout:
                last_error = f"Image generation timed out after {FIREWORKS_TIMEOUT_SEC}s"
                self._backoff(attempt)
            except Exception as e:
                last_error = f"Image generation error: {e}"
                break

        logger.warning(f"[Tier 2] Image generation failed, falling back to text: {last_error}")
        return None

    # ── Main handler ──────────────────────────────────────────────────

    def try_handle(self, query: str, context: str = "", intent: str = "unknown") -> TierResult:
        """
        Call Fireworks with smart model selection, bounded retries, and
        automatic fallback across models. Never raises — on total
        failure, returns a TierResult with confidence=0.0.
        """
        if not self.api_key:
            return TierResult(answer="", confidence=0.0, tier=2, tokens_used=0,
                               error="FW_API_KEY not set")

        if self.image_model and self._is_image_prompt(query):
            image_result = self._handle_image_generation(query)
            if image_result is not None:
                return image_result
            # Image generation unavailable/failed — fall through to the
            # text path below rather than returning nothing. A text
            # description is a worse answer than an image, but a much
            # better one than blank.
            logger.info("[Tier 2] No image produced; falling back to text description")

        models_to_try = get_sorted_models_by_intent(intent, self.available_models)
        if not models_to_try:
            return TierResult(answer="", confidence=0.0, tier=2, tokens_used=0,
                               error="No available (non-image) models to call")

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
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model_to_try,
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": FIREWORKS_TEMPERATURE,
                        },
                        timeout=FIREWORKS_TIMEOUT_SEC,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        raw_answer = data["choices"][0]["message"]["content"].strip()
                        tokens_used = data.get("usage", {}).get("total_tokens", 0)
                        self.total_tokens_used += tokens_used

                        answer = self._post_process_answer(raw_answer, query)

                        return TierResult(
                            answer=answer,
                            confidence=0.9,
                            tier=2,
                            tokens_used=tokens_used,
                            raw_response=str(data),
                        )

                    if response.status_code == 429:
                        last_error = "Rate limited (429)"
                        self._backoff(attempt)
                        continue

                    if response.status_code >= 500:
                        last_error = f"Server error ({response.status_code})"
                        self._backoff(attempt)
                        continue

                    # Non-retryable client error — move on to the next model
                    last_error = f"API error {response.status_code}: {response.text[:100]}"
                    logger.warning(
                        f"[Tier 2] {model_to_try.split('/')[-1]} failed "
                        f"({response.status_code}); trying next model"
                    )
                    break

                except requests.exceptions.Timeout:
                    last_error = f"Timeout after {FIREWORKS_TIMEOUT_SEC}s"
                    self._backoff(attempt)
                except requests.exceptions.ConnectionError as e:
                    last_error = f"Connection error: {e}"
                    self._backoff(attempt)
                except (KeyError, IndexError, ValueError) as e:
                    last_error = f"Malformed response: {e}"
                    break
                except Exception as e:
                    last_error = f"Unexpected error: {e}"
                    self._backoff(attempt)

        self.failure_count += 1
        logger.error(
            f"[Tier 2] Exhausted {total_attempts} attempt(s) across "
            f"{len(models_to_try)} model(s): {last_error}"
        )
        return TierResult(answer="", confidence=0.0, tier=2, tokens_used=0, error=last_error)

    def _backoff(self, attempt: int):
        delay = FIREWORKS_RETRY_BACKOFF_BASE ** attempt
        logger.warning(f"[Tier 2] Retrying in {delay:.1f}s (attempt {attempt})")
        time.sleep(delay)

    def get_stats(self) -> dict:
        return {
            "calls": self.call_count,
            "failures": self.failure_count,
            "total_tokens_used": self.total_tokens_used,
        }