#!/usr/bin/env python3
"""
THRIFT — Token Heuristic Routing with Intelligent Fallback Trees (Track 1)

Uses the full THRIFT pipeline:
1. Decompose compound queries into subtasks
2. Route each subtask through Tier 0 → Tier 2 (skip slow Tier 1)
3. Smart model selection in Tier 2 based on task complexity
4. Reassemble subtask answers into final response
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("thrift.run")

INPUT_PATH  = Path(os.getenv("INPUT_PATH",  "/input/tasks.json"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "/output/results.json"))

def load_tasks(path: Path) -> list:
    if not path.exists():
        logger.error(f"Input file not found: {path}")
        sys.exit(1)
    tasks = json.loads(path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(tasks)} tasks from {path}")
    return tasks

def write_results(results: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Written {len(results)} results to {path}")

def process_task(agent, task: dict, task_num: int, total: int) -> dict:
    task_id = task.get("task_id", f"unknown_{task_num}")
    prompt = task.get("prompt", "")
    
    if not prompt:
        return {"task_id": task_id, "answer": ""}
    
    start = time.time()
    
    # Use the full THRIFT pipeline
    run = agent.answer(prompt)
    answer = run.final_answer.text.strip()
    
    elapsed = round(time.time() - start, 2)
    logger.info(
        f"[{task_num}/{total}] {task_id} | "
        f"subtasks={run.final_answer.subtask_count} "
        f"tiers={run.final_answer.tiers_used} "
        f"tokens={run.final_answer.total_tokens_used} "
        f"latency={elapsed}s"
    )
    
    return {"task_id": task_id, "answer": answer}

def main():
    start_total = time.time()
    logger.info("=" * 60)
    logger.info("THRIFT — Token-Efficient Routing Agent")
    logger.info("=" * 60)
    
    # Log environment
    fw_key = os.getenv("FIREWORKS_API_KEY", "")
    fw_url = os.getenv("FIREWORKS_BASE_URL", "")
    models = os.getenv("ALLOWED_MODELS", "")
    logger.info(f"FIREWORKS_API_KEY : {'SET' if fw_key else 'NOT SET'}")
    logger.info(f"FIREWORKS_BASE_URL: {fw_url or 'NOT SET'}")
    logger.info(f"ALLOWED_MODELS    : {models or 'NOT SET'}")
    
    if not fw_key:
        logger.warning("FIREWORKS_API_KEY not set — Tier 2 will be unavailable")
    
    # Load tasks
    tasks = load_tasks(INPUT_PATH)
    
    # Init agent
    logger.info("Initialising THRIFT agent...")
    from thrift import Thrift
    agent = Thrift()
    
    # Process all tasks
    results = []
    total = len(tasks)
    
    for i, task in enumerate(tasks, 1):
        result = process_task(agent, task, i, total)
        results.append(result)
        
        # Safety check for 9-minute limit
        if time.time() - start_total > 540 and i < total:
            logger.warning("Approaching 10-minute limit. Flushing remaining tasks.")
            for remaining in tasks[i:]:
                results.append({"task_id": remaining.get("task_id", "unknown"), "answer": ""})
            break
    
    # Write results
    write_results(results, OUTPUT_PATH)
    
    # Summary
    total_elapsed = round(time.time() - start_total, 1)
    stats = agent.get_stats()
    logger.info("=" * 60)
    logger.info(f"Done in {total_elapsed}s")
    logger.info(f"Tasks processed: {len(results)}/{total}")
    logger.info(f"Total tokens: {stats.get('total_tokens_used', 0)}")
    logger.info(f"Tier usage: {stats.get('cascade_stats', {}).get('tier_usage_counts', {})}")
    logger.info("=" * 60)
    
    sys.exit(0)

if __name__ == "__main__":
    main()