# THRIFT — Token Heuristic Routing with Intelligent Fallback Trees

> **AMD Developer Hackathon ACT II — Track 1: Hybrid Token-Efficient Routing Agent**

---

## 🚀 The Core Concept

Most routers make one decision per query: local model or remote model. **THRIFT makes a decision per *piece* of a query.**

A compound request like *"Explain how transformers work and write a Python function for multi-head attention"* gets split into two independent subtasks — the explanation routes to a free local tier, the code generation only reaches the paid API if necessary.

When routing to the paid API, THRIFT employs two additional layers of intelligence:

- **Smart Model Selection** — automatically sorts `ALLOWED_MODELS` by parameter size and tries the cheapest sufficient model first
- **Specialized Prompts** — injects task-specific system prompts that force concise, perfectly-formatted outputs and stop reasoning models from thinking out loud and wasting tokens

---

## 🏗 Architecture

```
Query
  └─ Decomposer        (Conservative split: numbered lists, semicolons, "and" with different intents)
       └─ per subtask → Cascade
                          ├─ Tier 0: Rule-based heuristic   (Free, instant — safe AST arithmetic, no eval())
                          ├─ Tier 1: Small local model       (Free — skipped in prod to guarantee <60s on 2 vCPU)
                          └─ Tier 2: Fireworks Remote API    (Smart routing — paid, last resort)
                             ├─ Model Priority Sorting       (Cheapest model first, based on param count)
                             ├─ Specialized Prompts          (JSON-only for NER, one-word for sentiment, etc.)
                             └─ Automatic Fallback           (404/400 → instantly tries next model in list)
       └─ Reassembler   (Stitches subtask answers into one coherent response)
```

---

## 🏆 Key Winning Features

### 1. Intent-Based Smart Model Routing
Parses the `ALLOWED_MODELS` environment variable at runtime, sorts models by parameter count extracted from their names, and routes each task to the smallest model capable of handling it. Cheaper models = fewer tokens = higher leaderboard rank.

### 2. Specialized Prompts for Token Efficiency
THRIFT analyses query intent before calling the API and injects strict system prompts tailored to each category:

| Category | System Prompt Strategy |
|---|---|
| NER | "Return ONLY a valid JSON array" — no preamble |
| Sentiment | "Reply with ONE word: Positive, Negative, Mixed, or Neutral" |
| Code | "Provide ONLY the code in a ```python block" |
| Math | "State the final numerical answer first" |
| Reasoning | "Work step by step, then state the answer clearly" |

This stops large reasoning models from outputting thousands of tokens of chain-of-thought before giving the actual answer.

### 3. Automatic Model Fallback
If the preferred cheap model returns a 404 or 400 (deprecated, not deployed, or rate-limited), THRIFT silently falls back to the next model in the priority list — guaranteeing 100% task completion with zero crashes.

### 4. Ultra-Lightweight Docker Image
By skipping local model weights in the production image, the Docker image stays **under 2GB** — avoiding the PULL_ERROR and TIMEOUT failures that plague teams bundling 8GB+ local models. The grading environment (4GB RAM, 2 vCPU) is treated as what it is: an API-calling environment, not a GPU server.

### 5. Safe Math via AST (Tier 0)
Simple arithmetic and word-based math (9 squared, sqrt(16), 10% of 200) are solved instantly using a safe AST-based evaluator — no eval(), no injection risk, zero API tokens.

### 6. Safety Net Fallbacks
Hardcoded fallback model lists and graceful error handling ensure the agent never crashes mid-run, even if the judging harness fails to inject environment variables correctly.

---

## ⚙️ Configuration

All values are read from environment variables (or a local `.env` file for development). Do not hardcode keys in the image.

| Variable | Description | Default |
|---|---|---|
| `FIREWORKS_API_KEY` | Fireworks API key (injected by harness) | — |
| `FIREWORKS_BASE_URL` | API endpoint or judging proxy | https://api.fireworks.ai/inference/v1 |
| `ALLOWED_MODELS` | Comma-separated list of permitted model IDs | Full priority list |
| `INPUT_PATH` | Path to input tasks JSON | /input/tasks.json |
| `OUTPUT_PATH` | Path to write results JSON | /output/results.json |

---

## 🏃 Running Locally

### Prerequisites
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Create .env file
```
FW_API_KEY=your-fireworks-key-here
FIREWORKS_API_KEY=your-fireworks-key-here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/deepseek-v4-pro
INPUT_PATH=data/practice_tasks.json
OUTPUT_PATH=data/practice_results.json
```

### Run against practice tasks
```bash
python run.py
```

### Run the demo dashboard
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000
```

### Run tests
```bash
python -m unittest discover -s tests -v
```

---

## 🐳 Docker Submission

The image is optimised for the judging environment: 4GB RAM, 2 vCPU, no GPU, 10-minute runtime limit.

### Build
```bash
docker buildx build --platform linux/amd64 -t sammegh/thrift:latest .
```

### Test locally
```bash
mkdir -p test_input test_output
cp data/practice_tasks.json test_input/tasks.json

docker run --rm \
  -v $(pwd)/test_input:/input \
  -v $(pwd)/test_output:/output \
  -e FIREWORKS_API_KEY="your-key" \
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
  -e ALLOWED_MODELS="accounts/fireworks/models/deepseek-v4-pro" \
  sammegh/thrift:latest

cat test_output/results.json
```

### Push
```bash
docker push sammegh/thrift:latest
```

**Submission image URL:**
```
docker.io/sammegh/thrift:latest
```

---

## 📁 Project Structure

```
thrift/
├── run.py                  ← Submission entry point (reads tasks, writes results, exits 0)
├── config.py               ← All settings, reads from env vars / .env
├── safe_math.py            ← AST-based safe arithmetic (Tier 0)
├── decomposer.py           ← Conservative query splitter
├── tiers.py                ← Tier 0/1/2 implementations
├── cascade.py              ← Tier escalation orchestrator
├── reassembler.py          ← Multi-subtask answer stitcher
├── thrift.py               ← Main agent
├── eval_harness.py         ← Threshold sweeper
├── app.py                  ← FastAPI demo dashboard
├── static/index.html       ← Live demo UI with token savings chart
├── Dockerfile              ← linux/amd64, lightweight, fast pull
├── requirements.txt
├── .env.example
├── data/
│   ├── sample_queries.json
│   └── practice_tasks.json
├── tests/
│   ├── test_decomposer.py
│   ├── test_cascade.py
│   └── test_integration.py
└── README.md
```

---

## 🎯 Scoring Strategy

**19 fixed tasks. Accuracy gate = 80% (16/19). Ranked by total tokens ascending.**

| Category | Strategy | Est. Tokens |
|---|---|---|
| Math (simple) | Tier 0 — AST evaluator | 0 |
| Math (word problem) | Fireworks, 200 token cap | ~200 |
| Factual | Fireworks, 150 token cap | ~150 |
| Sentiment | Fireworks, 20 token cap (one word) | ~20 |
| Summarization | Fireworks, 150 token cap | ~150 |
| NER | Fireworks, JSON-only prompt, 300 cap | ~200 |
| Code debug | Fireworks, code-only prompt, 800 cap | ~400 |
| Code generation | Fireworks, code-only prompt, 800 cap | ~500 |
| Reasoning | Fireworks, 300 token cap | ~300 |

**Estimated total: ~2,000–3,000 tokens for 19 tasks** — targeting top 3 on the leaderboard.

---

*Built with PyTorch, HuggingFace Transformers, FastAPI, and Fireworks AI on AMD infrastructure.*