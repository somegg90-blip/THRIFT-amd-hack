#!/usr/bin/env python3
"""
THRIFT — Main entry point for grading harness.
"""
import json
import logging
import os
import sys
import time

from thrift import Thrift

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("thrift.run")


def main():
    logger.info("=" * 60)
    logger.info("THRIFT — Token-Efficient Routing Agent")
    logger.info("=" * 60)
    
    # Check environment
    from config import FIREWORKS_API_KEY, FIREWORKS_API_BASE, AVAILABLE_MODELS, SKIP_TIER_1
    logger.info(f"FIREWORKS_API_KEY : {'SET' if FIREWORKS_API_KEY else 'NOT SET'}")
    logger.info(f"FIREWORKS_BASE_URL: {FIREWORKS_API_BASE}")
    logger.info(f"ALLOWED_MODELS    : {','.join(AVAILABLE_MODELS[:3])}...")
    logger.info(f"SKIP_TIER_1       : {SKIP_TIER_1}")
    
    # Load tasks
    input_path = os.getenv("INPUT_PATH", "/input/tasks.json")
    output_path = os.getenv("OUTPUT_PATH", "/output/results.json")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    logger.info(f"Loaded {len(tasks)} tasks from {input_path}")
    
    # Initialize agent
    logger.info("Initialising THRIFT agent...")
    agent = Thrift()
    
    # Process tasks
    results = []
    start_time = time.time()
    total_tokens = 0
    
    for i, task in enumerate(tasks, 1):
        task_id = task.get("task_id", f"task-{i}")
        prompt = task.get("prompt", "")
        
        logger.info(f"[{i}/{len(tasks)}] Processing {task_id}...")
        
        try:
            run = agent.answer(prompt)
            total_tokens += run.tokens_used
            
            # Safely extract answer text
            answer_text = ""
            if hasattr(run, 'final_answer'):
                if hasattr(run.final_answer, 'text'):
                    answer_text = run.final_answer.text
                elif hasattr(run.final_answer, 'answer'):
                    answer_text = run.final_answer.answer
                elif isinstance(run.final_answer, str):
                    answer_text = run.final_answer
            
            results.append({
                "task_id": task_id,
                "answer": answer_text,
            })
            logger.info(f"  ✓ {task_id} | tokens={run.tokens_used}")
            
        except Exception as e:
            logger.error(f"  ✗ {task_id} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "task_id": task_id,
                "answer": "",
            })
    
    # Write results
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    logger.info(f"Written {len(results)} results to {output_path}")
    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()