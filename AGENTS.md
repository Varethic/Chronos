# AGENTS.md — Engineering Standards & Operational Protocol (v2)

This document defines the binding rules, architectural blueprints, and execution standards for any AI agent or engineer working on **Chronos Alpha** — a specialized conversational Python engineering model supporting **128k input** and **1M context window**.

---

## 1. Directory & File Structure Rules

The project workspace strictly permits ONLY two root directories. No other root folders are allowed.

```text
chronos/
├── AGENTS.md
├── docs/
│   ├── audit.md
│   └── report.md
└── src/
    ├── config.py
    ├── tokenizer.py
    ├── model.py
    ├── dataset.py
    ├── state.py                      # In-memory trajectory buffer & anti-regression state tracker
    ├── sandbox.py
    ├── antigravity.py
    ├── trainer.py
    ├── benchmark.py
    ├── main.py
    ├── data/                         # Serialized datasets only (.json)
    │   ├── traindata.json
    │   ├── testdata.json
    │   ├── manifest.json
    │   └── benchmark.json
    ├── json/                         # Model vocabulary, configs & run logs (.json only)
    │   ├── vocab.json
    │   ├── config.json
    │   ├── metrics.json
    │   ├── history.json
    │   └── antigravity.json
    └── models/                       # Binary model checkpoints & weights only (.pt)
        └── model.pt
```

### Naming Invariants:
- **Folder Names:** Only `src` and `docs` at the root; only `data`, `json`, and `models` inside `src`. No other folders or nested packages exist.
- **Artifact Placement:**
  - Serialized datasets (`traindata.json`, `testdata.json`, `manifest.json`, `benchmark.json`) must reside inside `src/data/`.
  - Model metadata, vocabulary, configuration, and execution logs (`vocab.json`, `config.json`, `metrics.json`, `history.json`, `antigravity.json`) must reside exclusively inside `src/json/`. No non-JSON file extensions are permitted in `src/json/`.
  - Binary model checkpoints and weights (`model.pt`) must reside inside `src/models/`.
  - Temporary buffers are written beside their target and atomically renamed.
- **File Names:** Strictly lowercase alphanumeric characters. **NO hyphens (`-`) and NO underscores (`_`)** are permitted in file names under `src/` or `docs/`.
- **Function Names:** Every custom function or method definition must start with a leading underscore and use strictly all-lowercase letters (e.g., `def _train():`, `def _step():`). For internal/nested helpers, use double underscores (e.g., `def __step():`).
  - *Mandatory Exceptions:* Standard Python dunder methods (`__init__`, `__len__`, `__getitem__`) and PyTorch `nn.Module.forward` must retain their exact framework-required names.
- **Imports:** Condense imports into single lines per module category. Never waste lines on multiline import blocks.
  ```python
  import os, sys, math, time, gc, json, ast, subprocess, re, csv, tempfile, random
  import torch, torch.nn as nn, torch.nn.functional as F
  ```

---

## 2. Model Specification: Chronos Alpha

- **Identity:** `Chronos Alpha`
- **Strict Domain Specialization & Guardrail:**
  1. **Python-Only Assistant:** Must converse, explain, debug, refactor, and write code exclusively for **Python** (Core Python, algorithms, debugging, `requests`, `httpx`, `urllib3`, `aiohttp`, `json`, `csv`, `re`, and standard library modules).
  2. **Strict Out-of-Domain Refusal:** Any prompt unrelated to Python (e.g., other programming languages like JavaScript/C++/Rust, cooking, politics, general trivia, or creative writing) MUST be politely and firmly refused with:
     `"I am Chronos Alpha, specialized exclusively in Python engineering. I can only assist with Python-related questions, debugging, and code."`
- **Conversation Format:** Chat-structured special tokens in `src/tokenizer.py`:
  `<|pad|>`, `<|bos|>`, `<|eos|>`, `<|unk|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|code|>`, `<|endcode|>`.
- **Long-Context Architecture (1M Context / 128k Input Ready):**
  - **Topology:** Memory-Bounded Decoder-Only Transformer with **Grouped-Query Attention (GQA)** and **Sliding Window Attention (SWA)** + **Chunked KV-Cache Prefill**.
  - **Parameters:** ~10M – 15M (regularized to prevent memorization).
  - **Dimensions:** `d_model=384`, `n_layers=8`, `n_heads=6`, `n_kv_heads=2`, `d_ff=1024`, `dropout=0.1`.
  - **Context & Input Limits:**
    - `max_input_tokens = 131072` (128k input capability via chunked prefill of size `2048`).
    - `max_context_tokens = 1048576` (1M context window capability via rolling SWA KV-cache with `window_size=2048` + `sink_tokens=16` Attention Sinks).
  - **Positional Encoding:** **Rotary Position Embeddings (RoPE)** with **NTK/YaRN Dynamic Scaling** (`rope_theta=500000.0`, `scaling_factor=32.0`) so models trained up to `4096` tokens extrapolate smoothly to `131072`–`1048576` positions without attention collapse or $O(N^2)$ RAM explosion.

---

## 3. Hardware Tuning & CPU Optimizations

- **Thread Binding:** Bind PyTorch to physical CPU cores to prevent hyperthreading thrashing:
  ```python
  def _bind():
      cores = max(1, (os.cpu_count() or 4) // 2)
      torch.set_num_threads(cores)
      torch.set_num_interop_threads(1)
  ```
- **Precision:** Use `torch.float32` as primary compute dtype on CPU (proven 50–100x faster than emulated CPU BF16 on non-AMX hardware). Only enable BF16 if an explicit micro-benchmark confirms throughput exceeds FP32.
- **Memory-Safe Chunked Inference:** Never allocate a full $N \times N$ attention mask for $N > 4096$. Process long inputs (up to 128k) in sequential chunks of `2048` tokens updating the rolling SWA + Attention Sink KV-cache.

---

## 4. Dataset & Evaluation Protocol (Anti-Overfitting & Long-Context)

To prevent memorization and build genuine Python reasoning, `src/dataset.py` must generate a diverse, parameterized corpus of **6,600 verified records**:

### Training Corpus (6,000 Unique Parameterized Samples):
1. **Pure Python Coding & Ecosystems (3,000 samples — 500 per domain):**
   - Loops & Iteration, Control Flow & Functions, Synchronous HTTP (`requests`, `httpx`, `urllib3`), Asynchronous HTTP (`aiohttp`), Nested JSON/Tree traversal, Built-in `csv` & `re` manipulation.
   - Must use randomized function names, variable identifiers, constants, and structural variations so no two samples share identical templates.
2. **Conversational Python Q&A & Debugging (1,800 samples):**
   - Multi-turn and single-turn `<|user|> ... <|assistant|>` dialogues explaining Python concepts (GIL, decorators, generators, async/await, memory management, error tracebacks) and fixing buggy snippets.
3. **Out-of-Domain Guardrail Refusals (600 samples):**
   - Non-Python prompts (JavaScript, SQL, C++, general knowledge, lifestyle, math puzzles without Python) paired with the exact Chronos Alpha domain-refusal response.
4. **Long-Context Needle-in-a-Haystack & Multi-File Retrieval (600 samples):**
   - Multi-module Python code contexts (`2,048` to `4,096` tokens during training) where the model must locate a specific bug, trace a variable across functions, or implement a missing helper conditioned on distant definitions.

### Holdout Evaluation Corpus (600 Unseen Samples):
- **300 Coding Holdout Challenges** (50 unseen problems per coding domain).
- **150 Conversational Python Q&A & Bug-Fix Prompts** (unseen).
- **100 Out-of-Domain Guardrail Prompts** (must achieve >= 95% refusal compliance).
- **50 Long-Context Needle-in-a-Haystack Tests** (scaled up to **128k tokens** input to verify RoPE extrapolation and chunked prefill retrieval).

---

## 5. State Management & Antigravity Self-Repair Cycle

Implement closed-loop autonomous correction inside `src/antigravity.py` backed by `src/state.py`:

```text
[Prompt Spec] ──► [Chronos Alpha Generates Code] ──► [AST Parse & Sandbox Run]
                                                               │
                       ┌───────────────────────────────────────┴───────────────────────────────────────┐
                       │                                                                               │
                       ▼ (Pass: Valid AST & Execution)                                                 ▼ (Fail: Syntax or Runtime Error)
         [Commit to Positive Buffer]                                                  [Isolate AST Node, Line No & Traceback]
                                                                                                       │
                                                                                                       ▼
                                                                                      [Check src/state.py: Seen Failure?]
                                                                                      (Prevent Repeating Identical Mutations)
                                                                                                       │
                                                                                                       ▼
                                                                                      [Construct Self-Correction Pair]
                                                                                 (<Prompt> + <Faulty Code> + <Traceback>
                                                                                              ──► <Fixed Code>)
                                                                                                       │
                                                                                                       ▼
                                                                                      [High-Priority Replay Step]
                                                                                      (Targeted Gradient Loss Update)
```

- **State Tracker (`src/state.py`):** Logs every attempt, traceback hash, and patch trajectory to detect loops and trigger mutation fallbacks.
- **Sandbox Isolation & Lazy Mocking (`src/sandbox.py`):** Execute candidate code inside `tempfile.TemporaryDirectory()` (`cwd=temp_dir`) via `subprocess.run` with strict 2.0-second timeouts. Use **lazy transport patching** (only importing/patching `requests`, `httpx`, `urllib3`, or `aiohttp` if referenced in code) to keep non-HTTP execution under `0.3s`.

---

## 6. Training Mandates, CLI Chat & Verification

1. **Two-Stage Curriculum Training (`src/trainer.py`):**
   - Use `AdamW` (`lr=3e-4`, `weight_decay=0.1`) with cosine decay, warmup, and gradient clipping at norm `1.0`.
   - **Stage 1 (Steps 1 – 6,000):** Train on `seq_len=1024` across coding, Q&A, and guardrail refusal datasets.
   - **Stage 2 (Steps 6,001 – 8,000):** Train on `seq_len=4096` with NTK/YaRN RoPE scaling on long-context multi-file Python retrieval and self-repair pairs.
2. **Atomic Persistence:** Save checkpoints as temporary files first, then execute `os.replace(tmp, path)` to `src/models/model.pt`.
3. **Interactive CLI Chat Mode (`src/main.py`):** Support `python src/main.py --chat` allowing interactive multi-turn Python conversation, file/long-context ingestion up to 128k tokens, rolling 1M context window management, and live domain-guardrail enforcement.
4. **Strict Verification (`src/benchmark.py`):** Evaluate against all 600 unseen holdout tests (coding pass rate, Q&A accuracy, out-of-domain refusal rate, and 128k needle-in-a-haystack retrieval). Append timestamped metrics to `docs/report.md` and log structural/runtime compliance in `docs/audit.md`. No mock metrics allowed.