# models.py
"""
Centralized model loader for Tier 1.

Keeps model-loading logic out of tiers.py so swapping models
(e.g. moving from Qwen 1.5B to a larger model on AMD Cloud)
is one change in config.py, not a code hunt across files.

TierOne in tiers.py handles its own lazy loading internally,
but you can also use this module to pre-warm the model at
startup (useful for the demo server so the first real query
isn't slow).
"""

import logging
from typing import Optional

logger = logging.getLogger("thrift.models")


def prewarm(model_name: Optional[str] = None):
    """
    Pre-load the local model at startup so the first query
    doesn't pay the cold-start penalty.

    Call this from app startup if you want:
        from models import prewarm
        prewarm()
    """
    from config import LOCAL_MODEL_NAME
    from tiers import TierOne

    name = model_name or LOCAL_MODEL_NAME
    logger.info(f"[models] Pre-warming local model: {name}")
    try:
        t1 = TierOne(model_name=name)
        t1._ensure_loaded()
        logger.info("[models] Pre-warm complete.")
        return t1
    except Exception as e:
        logger.warning(f"[models] Pre-warm failed (will load on first query): {e}")
        return None


def list_recommended_models() -> dict:
    """
    Reference list of models known to work well on AMD hardware
    for hackathon-scale deployments. Update after kickoff when
    Fireworks reveals the exact models to use.
    """
    return {
        "local_small": [
            "Qwen/Qwen2.5-0.5B-Instruct",    # fastest, lowest memory
            "Qwen/Qwen2.5-1.5B-Instruct",    # default — good balance
            "Qwen/Qwen2.5-3B-Instruct",      # step up if RAM allows
        ],
        "fireworks_remote": [
        "accounts/fireworks/models/llama4-maverick-instruct-basic",   # flagship — best quality
        "accounts/fireworks/models/llama-3.3-70b-instruct",           # solid fallback
        "accounts/fireworks/models/llama-3.1-8b-instruct",            # cheapest, fastest
        ],
        "notes": (
            "On AMD Developer Cloud with ROCm, install torch via: "
            "pip install torch --index-url https://download.pytorch.org/whl/rocm6.2 "
            "All other dependencies stay the same."
        )
    }