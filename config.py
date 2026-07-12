#!/usr/bin/env python3
"""
THRIFT — Configuration
"""
import os
import re
from dotenv import load_dotenv
load_dotenv()

# ── Fireworks API ──
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY") or os.getenv("FW_API_KEY", "")
FIREWORKS_API_BASE = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")

# ── Models ──
_allowed = os.getenv("ALLOWED_MODELS", "")
if _allowed:
    AVAILABLE_MODELS = [m.strip() for m in _allowed.split(",") if m.strip()]
else:
    AVAILABLE_MODELS = [
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/gpt-oss-120b",
        "accounts/fireworks/models/kimi-k2p6",
    ]

# Filter out image models
IMAGE_MODELS = {"accounts/fireworks/models/flux-1-schnell-fp8"}
AVAILABLE_MODELS = [m for m in AVAILABLE_MODELS if m not in IMAGE_MODELS]

FIREWORKS_MODEL = AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "accounts/fireworks/models/deepseek-v4-pro"

# ── Model Priorities (Accuracy-first) ──
HARD_TASK_MODELS = [
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/gpt-oss-120b",
    "accounts/fireworks/models/kimi-k2p6",
]
EASY_TASK_MODELS = [
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/gpt-oss-120b",
    "accounts/fireworks/models/kimi-k2p6",
]

COMPLEX_INTENTS = {"generation", "computation", "analysis"}

def get_sorted_models_by_intent(intent: str, available_models: list) -> list:
    """Return models in priority order based on task complexity."""
    usable = [m for m in available_models if m not in IMAGE_MODELS]
    if not usable:
        return []
    
    priority = HARD_TASK_MODELS if intent in COMPLEX_INTENTS else EASY_TASK_MODELS
    prioritized = [m for m in priority if m in usable]
    remaining = [m for m in usable if m not in prioritized]
    return prioritized + remaining

# ── Skip local model (CRITICAL for judging environment) ──
SKIP_TIER_1 = True  # ALWAYS skip local model for submission

# ── Cascade Thresholds ──
CONFIDENCE_THRESHOLD = 0.65

# ── Decomposition ──
MIN_SUBTASK_LENGTH = 15
MAX_SUBTASKS = 5

# ── Intent Keywords ──
INTENT_KEYWORDS = {
    "knowledge": ["explain", "describe", "what is", "what are", "who is", "define", "tell me about", "how does"],
    "generation": ["write", "create", "generate", "implement", "code", "draft", "compose", "build", "debug", "fix"],
    "computation": ["calculate", "compute", "solve", "find the", "evaluate", "determine", "what is the result", "how many"],
    "analysis": ["compare", "analyze", "contrast", "evaluate the pros", "discuss", "assess", "examine"],
    "enumeration": ["list", "name", "give me", "enumerate", "provide"],
    "translation": ["translate", "convert", "rewrite in"],
    "summarization": ["summarize", "sum up", "brief", "overview"],
}

# ── Tier 0 ──
SIMPLE_MATH_PATTERN = r'[\d]+\s*[\+\-\*\/\%\^]\s*[\d]+'

# ── Confidence estimation ──
HEDGE_PHRASES = [
    "i'm not sure", "i don't know", "i'm not certain", "might be", "may be",
    "i think", "possibly", "not 100% sure", "as an ai", "i don't have access",
    "unable to determine", "it's hard to say", "i apologize, but",
]
REFUSAL_PHRASES = ["i cannot answer", "i can't answer", "i cannot help with", "i won't"]
MIN_ANSWER_LENGTH = 3

# ── Tier 2 Fireworks ──
FIREWORKS_MAX_TOKENS_SIMPLE = 320
FIREWORKS_MAX_TOKENS_COMPLEX = 1100
FIREWORKS_TEMPERATURE = 0.1
FIREWORKS_TIMEOUT_SEC = 30
FIREWORKS_MAX_RETRIES = 2
FIREWORKS_RETRY_BACKOFF_BASE = 1.5
TIER2_MAX_TOTAL_ATTEMPTS = 3

# ── Eval harness ──
THRESHOLD_SWEEP_START = 0.40
THRESHOLD_SWEEP_END = 0.95
THRESHOLD_SWEEP_STEP = 0.05