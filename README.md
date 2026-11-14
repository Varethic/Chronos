# Chronos Alpha

A conversational **Python-only** language model built from scratch in PyTorch, designed for **128k-token inputs** and a **1M-token rolling context window** on ordinary CPU hardware.

Chronos Alpha is a 11.03M-parameter decoder-only transformer specialised in Python engineering: writing code, debugging tracebacks, explaining language internals, and refusing everything that is not Python.

---

## What makes it different

Most small models cannot handle long inputs on CPU: a 128k sequence needs an `N x N` attention matrix, which is ~68 GB per attention head in float32. Chronos Alpha avoids that entirely.

| Technique | Effect |
|---|---|
| **Grouped-Query Attention** (6 query / 2 KV heads) | KV cache is 3x smaller than multi-head |
| **Sliding Window Attention** (`window=2048`) + **16 attention sinks** | Attention memory is constant, not quadratic |
| **Chunked KV-cache prefill** (`chunk=2048`) | A 131,072-token input streams through in fixed blocks |
| **NTK/YaRN scaled RoPE** (`theta=5e5`, `factor=32`) | Trained at 4,096 positions, extrapolates to 131k+ |

Measured on the reference machine: processing a 131,072-token input added **0.25 GB** of RAM and kept the KV cache bounded at 2,064 cells. The equivalent full quadratic attention would need roughly **4,064x** more attention memory.

---

## Quick start

```bat
git clone https://github.com/Varethic/Chronos.git
cd Chronos
```

Chronos needs **Python 3.11+** and **PyTorch (CPU build)**. NumPy is optional; PyTorch warns when it is absent and nothing else is affected.

```bat
pip install torch
```

### 1. Verify the implementation

Checks naming invariants, corpus integrity, tokenizer losslessness, cache equivalence, masking, sandbox isolation, and atomic persistence:

```bat
python src\main.py verify
python src\main.py check
```

### 2. Build the datasets

Generates 6,000 training records and 600 held-out records, verifying every code sample by AST parse and sandboxed execution. Takes roughly 10-15 minutes because it builds 50 needle-in-a-haystack contexts of ~130k tokens each.

```bat
python src\main.py dataset
```

### 3. Train

Two-stage curriculum: 6,000 steps at sequence length 1,024, then 2,000 steps at 4,096 with short-corpus replay.

```bat
python src\main.py train
```

On 5 physical cores this takes about 4-5 hours. Checkpoints are written atomically every 50 steps, so interrupting and re-running resumes from the last completed optimizer step.

### 4. Benchmark

```bat
python src\main.py benchmark
```

Evaluates all 600 held-out records and prepends a timestamped result table to `docs/report.md`.

### 5. Chat

```bat
python src\main.py chat
```

```
you> /file src\tokenizer.py
ingested 12043 characters from src\tokenizer.py; ask a question to attach it
you> why does this fail on unicode input?
Chronos Alpha> ...
```

Commands: `/file <path>` ingests a Python file as context, `/status` shows window and checkpoint state, `/clear` resets the session, `/exit` quits.

---

## Architecture

```
Vocabulary        4,096 byte-level BPE merges + 9 chat control tokens
Layers            8 pre-norm decoder blocks
d_model           384
Attention         6 query heads / 2 KV heads (GQA), causal
Window            2,048 sliding window + 16 attention sinks
FFN               d_ff = 1,024 (SwiGLU-style gating)
Dropout           0.1
Positional        RoPE, theta = 500,000, YaRN scaling factor 32
Parameters        11,034,368
Precision         float32
```

### Chat format

```
<|bos|><|system|>...<|eos|><|user|>...<|eos|><|assistant|><|code|>...<|endcode|><|eos|>
```

---

## Domain guardrail

Chronos Alpha answers **only** Python questions. Anything else — JavaScript, C++, Rust, SQL, cooking, politics, general trivia, creative writing — receives this exact response:

> I am Chronos Alpha, specialized exclusively in Python engineering. I can only assist with Python-related questions, debugging, and code.

The guardrail is enforced twice: during training (600 refusal records with that exact string) and at runtime in `main.py`, which checks the input before the model ever sees it.

---

## Project layout

```
Chronos/
├── AGENTS.md              engineering standards this repo follows
├── README.md
├── SECURITY.md
├── CONTRIBUTING.md
├── LICENSE
├── .github/workflows/     continuous integration
├── docs/
│   ├── audit.md           compliance verification against AGENTS.md
│   └── report.md          timestamped benchmark results
└── src/
    ├── config.py          settings, curriculum schedule, atomic persistence
    ├── tokenizer.py       byte-level BPE with chat control tokens
    ├── model.py           GQA + sliding window + chunked prefill
    ├── dataset.py         corpus generation and verification
    ├── state.py           trajectory buffer and anti-regression tracking
    ├── sandbox.py         isolated subprocess code execution
    ├── antigravity.py     self-repair loop
    ├── trainer.py         two-stage curriculum training
    ├── benchmark.py       held-out evaluation
    ├── main.py            CLI entry point
    ├── data/              serialized datasets (.json)
    ├── json/              vocabulary, config, metrics, logs (.json)
    └── models/            checkpoints (.pt)
```

### Naming conventions

Filenames are lowercase alphanumeric with no hyphens or underscores. Every function starts with a leading underscore (`def _train():`). Frame-required dunders and `forward` are the only exceptions. These rules come from `AGENTS.md` and are enforced by `python src\main.py verify`.

---

## Large files and version control

`src/data/testdata.json` (~266 MB) and `src/models/model.pt` (~132 MB) exceed GitHub's 100 MB per-file limit and are excluded via `.gitignore`. Regenerate them with `python src\main.py dataset` and `python src\main.py train`.

---

## Hardware notes

- PyTorch is bound to `cpu_count() // 2` threads to avoid hyperthreading contention.
- float32 is used deliberately: CPU bfloat16 autocast is slower than float32 on hardware without AMX, and on some builds it fails outright.
- Length-bucketed batching keeps padding from dominating compute, which made a 46x difference here.

---

## Honest status

Chronos Alpha is a research prototype, not a production assistant. Evaluation results, including weaknesses, are published in [`docs/report.md`](docs/report.md) and the compliance ledger in [`docs/audit.md`](docs/audit.md). No metric in either file is estimated or invented; every number comes from running the checkpoint that is checked in.

See [`SECURITY.md`](SECURITY.md) for the threat model around sandboxed code execution.

---

## License

MIT. See [LICENSE](LICENSE).
