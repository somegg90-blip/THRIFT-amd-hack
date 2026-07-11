#!/usr/bin/env python3
"""
Tier implementations for the 3-tier cascade.
"""

import logging
import os
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
    FIREWORKS_MAX_TOKENS,
    FIREWORKS_TEMPERATURE,
    FIREWORKS_TIMEOUT_SEC,
    FIREWORKS_MAX_RETRIES,
    FIREWORKS_RETRY_BACKOFF_BASE,
    LOCAL_MAX_NEW_TOKENS,
    HEDGE_PHRASES,
    REFUSAL_PHRASES,
    MIN_ANSWER_LENGTH,
    SELF_RATING_WEIGHT,
    HEURISTIC_WEIGHT,
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
        # Fast confidence estimation. We skip the slow self-rating step 
        # to ensure Tier 1 executes in < 5 seconds if it is ever used.
        return self._heuristic_confidence(answer)

    def _heuristic_confidence(self, answer: str) -> float:
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
# Tier 2 — Fireworks Remote API (with smart model selection & fallback)
# ══════════════════════════════════════════════════════════════════════

class FireworksAPIError(RuntimeError):
    """Raised when the Fireworks API call fails after all retries."""


class TierTwo:
    """
    Remote API call to Fireworks with smart model selection and specialized prompts.
    Uses the cheapest sufficient model based on task complexity.
    If a model fails (e.g., 404 Not Found), it automatically falls back to the next model.
    """

    def __init__(self, api_key: str = FIREWORKS_API_KEY, model: str = FIREWORKS_MODEL):
        self.api_key = api_key
        self.default_model = model
        self.call_count = 0
        self.total_tokens_used = 0
        self.failure_count = 0
        
        # Parse available models from ALLOWED_MODELS
        allowed = os.getenv("ALLOWED_MODELS", "")
        if allowed:
            self.available_models = [m.strip() for m in allowed.split(",") if m.strip()]
        else:
            self.available_models = [model]

    def _build_specialized_prompt(self, query: str):
        """
        Analyze the query to build a specialized prompt that enforces 
        conciseness and correct formatting (e.g., JSON for NER).
        This saves tokens and improves accuracy.
        """
        lower = query.lower()
        
        # Base system prompt for extreme conciseness
        system = (
            "You are a highly efficient assistant. "
            "CRITICAL: Be extremely concise. Provide ONLY the direct answer. "
            "No explanations, no thinking out loud, no preamble."
        )
        user_content = query
        
        # 1. NER (Named Entity Recognition)
        if any(k in lower for k in ["extract", "named entit", "entities", "label each"]):
            system += " Return ONLY a valid JSON array of objects with 'entity' and 'type' fields. Example: [{\"entity\": \"Maria\", \"type\": \"PERSON\"}]. No markdown, no other text."
            user_content = f"{query}\n\nReturn ONLY the JSON array."
            
        # 2. Sentiment Analysis
        elif any(k in lower for k in ["classify the sentiment", "sentiment of"]) and "sentiment" in lower:
            system += " Reply with ONLY one word: Positive, Negative, Mixed, or Neutral."
            
        # 3. Code Generation / Debugging
        elif any(k in lower for k in ["write a function", "write a python", "implement", "debug", "fix this", "find the error"]):
            system += " Provide ONLY the corrected/complete code in a ```python block. No explanation."
            
        # 4. Math Word Problems
        elif any(k in lower for k in ["calculate", "how many", "what is the total", "percent", "arithmetic", "word problem"]):
            system += " Provide the final numerical answer first (use plain numbers, no LaTeX). Be concise."
            
        # 5. Summarization
        elif any(k in lower for k in ["summarize", "summarise", "summary", "one sentence"]):
            system += " Provide ONLY the summary. No preamble."
            
        return system, user_content

    def try_handle(self, query: str, context: str = "", intent: str = "unknown") -> TierResult:
        """
        Call Fireworks with smart model selection and automatic fallback.
        If the cheapest model fails with a 404, it automatically tries the next largest model.
        """
        if not self.api_key:
            return TierResult(answer="", confidence=0.0, tier=2, tokens_used=0,
                               error="FW_API_KEY not set")

        # Get the ordered fallback list of models
        from config import get_sorted_models_by_intent
        models_to_try = get_sorted_models_by_intent(intent, self.available_models)
        
        # Build specialized prompt for conciseness and format compliance
        system_prompt, user_content = self._build_specialized_prompt(query)
        
        if context:
            user_content = f"Context: {context}\n\n{user_content}"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        last_error = None
        
        # Try each model in the fallback list
        for model_to_try in models_to_try:
            for attempt in range(1, FIREWORKS_MAX_RETRIES + 1):
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
                            "max_tokens": FIREWORKS_MAX_TOKENS,
                            "temperature": FIREWORKS_TEMPERATURE,
                        },
                        timeout=FIREWORKS_TIMEOUT_SEC,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer = data["choices"][0]["message"]["content"].strip()
                        tokens_used = data.get("usage", {}).get("total_tokens", 0)
                        self.total_tokens_used += tokens_used

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

                    # Client error (404 Not Found, 400 Bad Request, etc.)
                    # DO NOT RETRY. Break inner loop and try the NEXT model in the list.
                    last_error = f"API error {response.status_code}: {response.text[:100]}"
                    logger.warning(f"[Tier 2] Model {model_to_try.split('/')[-1]} failed with client error. Falling back to next model.")
                    break 

                except requests.exceptions.Timeout:
                    last_error = f"Timeout after {FIREWORKS_TIMEOUT_SEC}s"
                    self._backoff(attempt)
                except requests.exceptions.ConnectionError as e:
                    last_error = f"Connection error: {e}"
                    self._backoff(attempt)
                except Exception as e:
                    last_error = f"Unexpected error: {e}"
                    self._backoff(attempt)

        # If we exhausted all models and retries
        self.failure_count += 1
        logger.error(f"[Tier 2] All models and retries exhausted: {last_error}")
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