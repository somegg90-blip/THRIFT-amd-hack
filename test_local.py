#!/usr/bin/env python3
"""
Local test — simulates the grading harness without Docker.
Usage:
    python test_local.py
    python test_local.py --tasks data/practice_tasks.json
"""
import json, os, sys, argparse
from pathlib import Path
from dotenv import load_dotenv

# Load .env FIRST
load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",  default="data/practice_tasks.json")
    parser.add_argument("--output", default="data/practice_results.json")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        print(f"Tasks file not found: {tasks_path}")
        sys.exit(1)

    # Simulate harness env vars
    os.environ["INPUT_PATH"]        = str(tasks_path)
    os.environ["OUTPUT_PATH"]       = args.output
    os.environ.setdefault("FIREWORKS_API_KEY",  os.getenv("FIREWORKS_API_KEY", ""))
    os.environ.setdefault("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    os.environ.setdefault("ALLOWED_MODELS",     "accounts/fireworks/models/deepseek-v4-pro")

    print(f"Running {len(json.loads(tasks_path.read_text()))} tasks...")
    print(f"API key: {'SET' if os.environ.get('FIREWORKS_API_KEY') else 'NOT SET'}")

    # Run the submission entry point directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("thrift_run", "run.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()

if __name__ == "__main__":
    main()