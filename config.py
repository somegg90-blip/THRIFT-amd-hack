#!/usr/bin/env python3
"""
THRIFT — Single source of truth for all configuration.
"""
import os
import re
from dotenv import load_dotenv
load_dotenv()

# ── Hardcoded Fallbacks (Safety net if judges forget to inject env vars) ──
FALLBACK_API_KEY = ""
FALLBACK_API_BASE = "https://api.fireworks.ai/inference/v1"

# ── Models ────────────────────────────────────────────────────────────
# These are required by tiers.py
LOCAL_MODEL_NAME = os.getenv("THRIFT_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
FIREWORKS_MODEL = "accounts/fireworks/models/deepseek-v4-pro"

# The judges' system will inject FIREWORKS_API_KEY. If they don't, we use yours.
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY") or os.getenv("FW_API_KEY", "") or FALLBACK_API_KEY
FIREWORKS_API_BASE = os.getenv("FIREWORKS_BASE_URL", FALLBACK_API_BASE)

_allowed = os.getenv("ALLOWED_MODELS", "")
if _allowed:
    AVAILABLE_MODELS = [m.strip() for m in _allowed.split(",") if m.strip()]
else:
    # Fallback to the full list of models if none are injected
    AVAILABLE_MODELS = [
        "accounts/fireworks/models/deepseek-v4-pro",
        "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/glm-5p1",
        "accounts/fireworks/models/gpt-oss-120b",
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/kimi-k2p5"
    ]

# ── Model Priority List (Optimized for Token Efficiency & Quality) ────
# This is the exact order we want to try models in, based on your testing.
MODEL_PRIORITY = [         
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/glm-5p2",         
    "accounts/fireworks/models/glm-5p1", 
    "accounts/fireworks/models/gpt-oss-120b",    # Great coding, but verbose
    "accounts/fireworks/models/kimi-k2p6",       # Strong reasoning, long answers
    "accounts/fireworks/models/kimi-k2p5",       # Similar to K2.6
]

# Models to explicitly avoid (e.g., image generation models that will fail text tasks)
IMAGE_MODELS = {"accounts/fireworks/models/flux-1-schnell-fp8"}

def get_sorted_models_by_intent(intent: str, available_models: list) -> list:
    """
    Returns a fallback list of models to try, ordered by our custom priority list.
    Filters the available models against our priority list to ensure we only use
    the best models in the exact order we want.
    """
    if not available_models:
        return MODEL_PRIORITY
        
    prioritized = []
    
    # 1. First, add models that are in our strict priority list
    for model in MODEL_PRIORITY:
        if model in available_models:
            prioritized.append(model)
            
    # 2. Second, add any other available models that aren't image models
    for model in available_models:
        if model not in prioritized and model not in IMAGE_MODELS:
            prioritized.append(model)
            
    if not prioritized:
        return [m for m in available_models if m not in IMAGE_MODELS]
        
    return prioritized

# ── Performance Optimization ──────────────────────────────────────────
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
                   "draft", "compose", "build"],
    "computation": ["calculate", "compute", "solve", "find the", "evaluate",
                    "determine", "what is the result"],
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
FIREWORKS_MAX_TOKENS = 1024
FIREWORKS_TEMPERATURE = 0.1
FIREWORKS_TIMEOUT_SEC = 30
FIREWORKS_MAX_RETRIES = 3
FIREWORKS_RETRY_BACKOFF_BASE = 1.5

# ── Eval harness ─────────────────────────────────────────────────────
THRESHOLD_SWEEP_START = 0.40
THRESHOLD_SWEEP_END = 0.95
THRESHOLD_SWEEP_STEP = 0.05