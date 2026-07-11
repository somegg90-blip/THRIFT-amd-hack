#!/usr/bin/env python3
"""
THRIFT — Single source of truth for all configuration.
"""
import os
import re
from dotenv import load_dotenv
load_dotenv()

# ── Hardcoded Fallbacks (safety net if judges forget to inject env vars) ──
FALLBACK_API_KEY = ""  # Your API key as safety net
FALLBACK_API_BASE = "https://api.fireworks.ai/inference/v1"

# ── Models ────────────────────────────────────────────────────────────
LOCAL_MODEL_NAME = os.getenv("THRIFT_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY") or os.getenv("FW_API_KEY", "") or FALLBACK_API_KEY
FIREWORKS_API_BASE = os.getenv("FIREWORKS_BASE_URL", FALLBACK_API_BASE)

# Models we should never call as a text model even if they show up in
# ALLOWED_MODELS (e.g. an image-generation model in the same account).
IMAGE_MODELS = {"accounts/fireworks/models/flux-1-schnell-fp8"}

_allowed = os.getenv("ALLOWED_MODELS", "")
if _allowed:
    AVAILABLE_MODELS = [m.strip() for m in _allowed.split(",") if m.strip()]
else:
    # Local-dev-only fallback. The harness always injects ALLOWED_MODELS
    # in the grading environment, so this branch should never fire there.
    AVAILABLE_MODELS = [
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/glm-5p1",
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/gpt-oss-120b",
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/kimi-k2p5",
    ]

# Single default model, derived from whatever was actually injected —
# never a hardcoded name that might not be in the allowed pool.
_usable_default = [m for m in AVAILABLE_MODELS if m not in IMAGE_MODELS]
FIREWORKS_MODEL = _usable_default[0] if _usable_default else "accounts/fireworks/models/deepseek-v4-pro"

# ── Smart Model Selection (accuracy-first for hard tasks, cost-first for easy) ──
HARD_TASK_MODEL_PRIORITY = [
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/gpt-oss-120b",
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/glm-5p2",
    "accounts/fireworks/models/glm-5p1",
    "accounts/fireworks/models/kimi-k2p5",
]
EASY_TASK_MODEL_PRIORITY = [
    "accounts/fireworks/models/glm-5p2",
    "accounts/fireworks/models/glm-5p1",
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/gpt-oss-120b",
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/kimi-k2p5",
]

COMPLEX_INTENTS = {"generation", "computation", "analysis"}


def _model_param_size(model_name: str) -> float:
    """Approximate parameter count in billions, parsed from the model name."""
    match = re.search(r'(\d+(?:\.\d+)?)b(?=[-_]|$)', model_name.lower())
    if not match:
        match = re.search(r'(\d+(?:\.\d+)?)b', model_name.lower())
    return float(match.group(1)) if match else float("inf")


def get_sorted_models_by_intent(intent: str, available_models: list) -> list:
    """
    Rank ONLY the models actually present in `available_models`, most-preferred first:
      - Complex intents: curated "smartest first" order, then any
        leftover models ranked largest-parameter-first.
      - Simple intents: curated "cheapest first" order, then any
        leftover models ranked smallest-parameter-first.
    """
    usable = [m for m in available_models if m not in IMAGE_MODELS]
    if not usable:
        return []

    is_complex = intent in COMPLEX_INTENTS
    known_priority = HARD_TASK_MODEL_PRIORITY if is_complex else EASY_TASK_MODEL_PRIORITY

    prioritized = [m for m in known_priority if m in usable]
    remaining = [m for m in usable if m not in prioritized]
    remaining_sorted = sorted(remaining, key=_model_param_size, reverse=is_complex)

    return prioritized + remaining_sorted


# ── Performance / environment tuning ────────────────────────────────────
SKIP_TIER_1 = os.getenv("THRIFT_SKIP_TIER_1", "true").lower() == "true"
USE_4BIT = os.getenv("THRIFT_USE_4BIT", "false").lower() == "true"

# ── Cascade Thresholds ────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("THRIFT_CONFIDENCE_THRESHOLD", "0.65"))

# ── Decomposition ─────────────────────────────────────────────────────
MIN_SUBTASK_LENGTH = 15
MAX_SUBTASKS = 5

# ── Intent Keywords ───────────────────────────────────────────────────
INTENT_KEYWORDS = {
    "knowledge": ["explain", "describe", "what is", "what are", "who is",
                  "define", "tell me about", "how does"],
    "generation": ["write", "create", "generate", "implement", "code",
                   "draft", "compose", "build", "debug", "fix"],
    "computation": ["calculate", "compute", "solve", "find the", "evaluate",
                    "determine", "what is the result", "how many"],
    "analysis": ["compare", "analyze", "contrast", "evaluate the pros",
                 "discuss", "assess", "examine"],
    "enumeration": ["list", "name", "give me", "enumerate", "provide"],
    "translation": ["translate", "convert", "rewrite in"],
    "summarization": ["summarize", "sum up", "brief", "overview"],
}

# ── Tier 0 ────────────────────────────────────────────────────────────
SIMPLE_MATH_PATTERN = r'[\d]+\s*[\+\-\*\/\%\^]\s*[\d]+'

# ── Tier 1 local model ────────────────────────────────────────────────
LOCAL_MAX_NEW_TOKENS = 100
LOCAL_GENERATION_TIMEOUT_SEC = 60

# ── Confidence estimation ─────────────────────────────────────────────
HEDGE_PHRASES = [
    "i'm not sure", "i am not sure", "i don't know", "i do not know",
    "i'm not certain", "i am not certain", "might be", "may be",
    "i think", "possibly", "not 100% sure", "i cannot be certain",
    "as an ai", "i don't have access", "i do not have access",
    "unable to determine", "it's hard to say", "it is hard to say",
    "i apologize, but", "i'm sorry, but i", "without more information",
]
REFUSAL_PHRASES = [
    "i cannot answer", "i can't answer", "i cannot help with",
    "i can't help with", "i won't", "i will not provide",
]
MIN_ANSWER_LENGTH = 3
SELF_RATING_WEIGHT = 0.5
HEURISTIC_WEIGHT = 0.5

# ── Tier 2 Fireworks ─────────────────────────────────────────────────
# Lower default token budgets to favour concise answers and reduce cost.
# These can still be overridden via env vars if you need longer outputs.
FIREWORKS_MAX_TOKENS_SIMPLE = int(os.getenv("THRIFT_MAX_TOKENS_SIMPLE", "120"))
FIREWORKS_MAX_TOKENS_COMPLEX = int(os.getenv("THRIFT_MAX_TOKENS_COMPLEX", "400"))
FIREWORKS_MAX_TOKENS = FIREWORKS_MAX_TOKENS_COMPLEX  # legacy alias

FIREWORKS_TEMPERATURE = float(os.getenv("THRIFT_TEMPERATURE", "0.1"))
FIREWORKS_TIMEOUT_SEC = int(os.getenv("THRIFT_TIMEOUT_SEC", "30"))
# Reduce retries by default to limit token and time spend
FIREWORKS_MAX_RETRIES = int(os.getenv("THRIFT_MAX_RETRIES", "1"))
FIREWORKS_RETRY_BACKOFF_BASE = float(os.getenv("THRIFT_RETRY_BACKOFF_BASE", "1.5"))

# Cap total attempts across models to keep token spend predictable
TIER2_MAX_TOTAL_ATTEMPTS = int(os.getenv("THRIFT_TIER2_MAX_TOTAL_ATTEMPTS", "1"))

# ── Eval harness ─────────────────────────────────────────────────────
THRESHOLD_SWEEP_START = 0.40
THRESHOLD_SWEEP_END = 0.95
THRESHOLD_SWEEP_STEP = 0.05