# Chronos Alpha benchmark

## Run 2026-11-17 08:47:41

Checkpoint optimizer step 8000, precision float32, 5 physical cores, 4815.0 s wall time.

Every number below was produced by running this checkpoint: greedy decoding from the chat-framed prompt, then AST validation and sandboxed execution (coding), keyword coverage against a reference explanation (Q&A), exact-sentence detection (guardrail), and chunked-prefill needle lookup at 128k tokens (retrieval). Reference implementations are never substituted for generated code.

| Holdout slice | Records | Result | Rate |
|---|---:|---|---:|
| Coding (execution-correct) | 300 | 0 | 0.00% |
| Coding (syntax-valid) | 300 | 0 | 0.00% |
| Q&A (explanation coverage mean) | 150 | 2.72% | - |
| Q&A (exact match) | 150 | 0 | 0.00% |
| Guardrail (refused) | 100 | 74 | 74.00% |
| 128k retrieval (needle found) | 50 | 0 | 0.00% |

| Coding domain | Correct | Syntax-valid |
|---|---:|---:|
| loops | 0 / 50 | 0 / 50 |
| control | 0 / 50 | 0 / 50 |
| http | 0 / 50 | 0 / 50 |
| async | 0 / 50 | 0 / 50 |
| json | 0 / 50 | 0 / 50 |
| text | 0 / 50 | 0 / 50 |

## Latency (seconds)

```json
{
  "generation": {
    "min": 0.1203203999903053,
    "median": 0.9632061500160489,
    "p95": 6.5022416999563575,
    "max": 7.660323199990671,
    "mean": 1.519184549455531
  },
  "execution": {
    "min": 0.0,
    "median": 0.0,
    "p95": 0.0,
    "max": 0.0,
    "mean": 0.0
  },
  "prefill": {
    "min": 37.72208580002189,
    "median": 51.158725150016835,
    "p95": 58.37011890002759,
    "max": 59.890294300043024,
    "mean": 50.52113888200489
  }
}
```

Longest retrieval input: 59881 tokens. Raw generations, per-record tracebacks and timings are in `src/data/benchmark.json`.

## Reproduction

```bat
python src\main.py verify
python src\main.py train
python src\main.py benchmark
python src\main.py chat
```

---

# Chronos Alpha benchmark

## Run 2026-11-17 00:45:52

Checkpoint optimizer step 8000, precision float32, 5 physical cores, 4564.1 s wall time.

Every number below was produced by running this checkpoint: greedy decoding from the chat-framed prompt, then AST validation and sandboxed execution (coding), keyword coverage against a reference explanation (Q&A), exact-sentence detection (guardrail), and chunked-prefill needle lookup at 128k tokens (retrieval). Reference implementations are never substituted for generated code.

| Holdout slice | Records | Result | Rate |
|---|---:|---|---:|
| Coding (execution-correct) | 300 | 0 | 0.00% |
| Coding (syntax-valid) | 300 | 0 | 0.00% |
| Q&A (explanation coverage mean) | 150 | 5.83% | - |
| Q&A (exact match) | 150 | 0 | 0.00% |
| Guardrail (refused) | 100 | 65 | 65.00% |
| 128k retrieval (needle found) | 50 | 0 | 0.00% |

| Coding domain | Correct | Syntax-valid |
|---|---:|---:|
| loops | 0 / 50 | 0 / 50 |
| control | 0 / 50 | 0 / 50 |
| http | 0 / 50 | 0 / 50 |
| async | 0 / 50 | 0 / 50 |
| json | 0 / 50 | 0 / 50 |
| text | 0 / 50 | 0 / 50 |

## Latency (seconds)

```json
{
  "generation": {
    "min": 0.12823929998558015,
    "median": 1.0198206499917433,
    "p95": 2.5629013999714516,
    "max": 6.885363399982452,
    "mean": 1.074371641090415
  },
  "execution": {
    "min": 0.0,
    "median": 0.0,
    "p95": 0.0,
    "max": 0.0,
    "mean": 0.0
  },
  "prefill": {
    "min": 37.49087510001846,
    "median": 50.92587989999447,
    "p95": 59.478552100015804,
    "max": 59.8506637000246,
    "mean": 50.25668454999919
  }
}
```

Longest retrieval input: 59902 tokens. Raw generations, per-record tracebacks and timings are in `src/data/benchmark.json`.

## Reproduction

```bat
python src\main.py verify
python src\main.py train
python src\main.py benchmark
python src\main.py chat
```

---

# Chronos Alpha benchmark

## Run 2026-11-15 09:52:35

Checkpoint optimizer step 8000, precision float32, 5 physical cores, 5688.7 s wall time.

Every number below was produced by running this checkpoint: greedy decoding from the chat-framed prompt, then AST validation and sandboxed execution (coding), keyword coverage against a reference explanation (Q&A), exact-sentence detection (guardrail), and chunked-prefill needle lookup at 128k tokens (retrieval). Reference implementations are never substituted for generated code.

| Holdout slice | Records | Result | Rate |
|---|---:|---|---:|
| Coding (execution-correct) | 300 | 0 | 0.00% |
| Coding (syntax-valid) | 300 | 6 | 2.00% |
| Q&A (explanation coverage mean) | 150 | 8.22% | - |
| Q&A (exact match) | 150 | 0 | 0.00% |
| Guardrail (refused) | 100 | 16 | 16.00% |
| 128k retrieval (needle found) | 50 | 0 | 0.00% |

| Coding domain | Correct | Syntax-valid |
|---|---:|---:|
| loops | 0 / 50 | 4 / 50 |
| control | 0 / 50 | 0 / 50 |
| http | 0 / 50 | 0 / 50 |
| async | 0 / 50 | 2 / 50 |
| json | 0 / 50 | 0 / 50 |
| text | 0 / 50 | 0 / 50 |

## Latency (seconds)

```json
{
  "generation": {
    "min": 0.11179250001441687,
    "median": 2.2644112999842037,
    "p95": 7.2691002000065055,
    "max": 7.381844399991678,
    "mean": 2.6752914087268502
  },
  "execution": {
    "min": 0.0,
    "median": 0.0,
    "p95": 0.0,
    "max": 0.3125468000071123,
    "mean": 0.005922477333321391
  },
  "prefill": {
    "min": 42.82850880001206,
    "median": 55.626074599989806,
    "p95": 63.16933260002406,
    "max": 63.78351369997836,
    "mean": 55.4386751120002
  }
}
```

Longest retrieval input: 61861 tokens. Raw generations, per-record tracebacks and timings are in `src/data/benchmark.json`.

## Reproduction

```bat
python src\main.py verify
python src\main.py train
python src\main.py benchmark
python src\main.py chat
```

---

# Chronos Alpha benchmark

## Run 2026-11-14 17:33:41

Checkpoint optimizer step 8000, precision float32, 5 physical cores, 7667.3 s wall time.

Every number below was produced by running this checkpoint: greedy decoding from the chat-framed prompt, then AST validation and sandboxed execution (coding), keyword coverage against a reference explanation (Q&A), exact-sentence detection (guardrail), and chunked-prefill needle lookup at 128k tokens (retrieval). Reference implementations are never substituted for generated code.

| Holdout slice | Records | Result | Rate |
|---|---:|---|---:|
| Coding (execution-correct) | 300 | 0 | 0.00% |
| Coding (syntax-valid) | 300 | 50 | 16.67% |
| Q&A (explanation coverage mean) | 150 | 3.50% | - |
| Q&A (exact match) | 150 | 0 | 0.00% |
| Guardrail (refused) | 100 | 0 | 0.00% |
| 128k retrieval (needle found) | 50 | 0 | 0.00% |

| Coding domain | Correct | Syntax-valid |
|---|---:|---:|
| loops | 0 / 50 | 9 / 50 |
| control | 0 / 50 | 7 / 50 |
| http | 0 / 50 | 3 / 50 |
| async | 0 / 50 | 11 / 50 |
| json | 0 / 50 | 8 / 50 |
| text | 0 / 50 | 12 / 50 |

## Latency (seconds)

```json
{
  "generation": {
    "min": 0.033769700006814674,
    "median": 5.4780096000031335,
    "p95": 9.156011799990665,
    "max": 13.36698580000666,
    "mean": 5.632197632364167
  },
  "execution": {
    "min": 0.0,
    "median": 0.0,
    "p95": 0.3147004999918863,
    "max": 0.4796464999963064,
    "mean": 0.052464199333432285
  },
  "prefill": {
    "min": 43.68129079998471,
    "median": 59.76552925001306,
    "p95": 74.01039370000944,
    "max": 85.19109249999747,
    "mean": 60.33448802600091
  }
}
```

Longest retrieval input: 61861 tokens. Raw generations, per-record tracebacks and timings are in `src/data/benchmark.json`.

## Reproduction

```bat
python src\main.py verify
python src\main.py train
python src\main.py benchmark
python src\main.py chat
```

---

# Chronos Alpha benchmark

## Run 2026-11-14 (checkpoint step 8000) — sampled diagnostic

Checkpoint optimizer step 8000, precision float32, 5 physical cores, 8,000 of 8,000 optimizer steps completed with 0 skipped.

The full 600-record benchmark has not been run against this checkpoint yet. The table below is a **sampled diagnostic** produced by greedy decoding from the chat-framed prompt, so it measures the same pipeline the full benchmark uses, on a subset.

| Holdout slice | Sampled | Result | Rate |
|---|---:|---|---:|
| Coding (syntax-valid) | 40 | 6 | 15.00% |
| Coding (execution-correct) | 40 | not yet measured | - |
| Guardrail (exact refusal) | 20 | 0 | 0.00% |
| Q&A (exact match) | 10 | 0 | 0.00% |

### What this shows

The checkpoint does not yet behave as a usable Python assistant. Generation drifts into fragments of training text rather than a coherent answer, so most samples are neither valid Python nor the mandated refusal sentence.

Held-in loss on the training corpus remained high even after all 8,000 steps:

| Domain | Held-in loss | Held-in token accuracy |
|---|---:|---:|
| loops | 9.01 | 30.0% |
| control | 10.83 | 14.8% |
| http | 8.38 | 38.3% |
| async | 8.58 | 37.0% |
| json | 10.91 | 14.2% |
| text | 8.71 | 41.5% |
| qa | 6.76 | 10.2% |
| guardrail | 4.83 | 33.9% |
| longcontext | 4.01 | 37.9% |

Uniform guessing over a 4,096-token vocabulary yields a loss of about 8.3, so most domains are at or above the random baseline. The training log reported losses near 0.0 because the length-bucketed loader serves the shortest bucket first and stride-100 logging sampled those batches; that number was real but not representative. The per-domain table above is the honest measurement.

### Likely causes

- 11.03M parameters spread over a 4,096-token vocabulary leaves little capacity per token.
- Coding records carry only ~75 supervised tokens each, so 8,000 steps saw roughly 6M supervised tokens.
- The cosine schedule decays to 3e-5 by the end, and an embedding update measured at step 8000 moved weights by only 1.4e-5, which is too small to keep learning.

### Next steps

Select a lower cosine floor or extend the schedule so the final phase keeps a usable learning rate, then retrain and rerun this evaluation. Until a checkpoint clears the guardrail and coding slices, this project should be read as a working long-context architecture with an under-trained language model attached.

---

# Chronos Alpha benchmark

## Run 2026-11-13 17:40:19

Checkpoint optimizer step 5000, precision float32, 5 physical cores.

## Raw execution results

Every result below comes from greedy autoregressive generation followed by AST validation and sandboxed probe execution. Reference implementations are never substituted for generated code.

| Metric | Observed value |
|---|---:|
| Model | Chronos Alpha |
| Run timestamp | 2026-11-13 17:40:19 |
| Checkpoint optimizer step | 5000 |
| Unseen holdout problems | 300 |
| Syntax-valid | 9 / 300 |
| Execution-correct | 0 / 300 |
| Syntax rate | 3.00% |
| Raw pass rate | 0.00% |

| Domain | Correct | Syntax-valid |
|---|---:|---:|
| loops | 0 / 50 | 0 / 50 |
| control | 0 / 50 | 0 / 50 |
| http | 0 / 50 | 0 / 50 |
| async | 0 / 50 | 0 / 50 |
| json | 0 / 50 | 0 / 50 |
| text | 0 / 50 | 9 / 50 |

## Latency (seconds)

```json
{
  "generation": {
    "min": 0.008449900000414345,
    "median": 0.06725335000373889,
    "p95": 0.15957510000589537,
    "max": 0.2663807999997516,
    "mean": 0.07240527300016159
  },
  "execution": {
    "min": 0.0,
    "median": 0.0,
    "p95": 0.0,
    "max": 0.2161204000003636,
    "mean": 0.00514558533335124
  }
}
```

The model emits no correctness guarantee. A low passing rate means measured generalization failed on that slice; it must not be replaced by reference-output scores. Raw generations, tracebacks, per-record timings, and model outputs are in `src/data/benchmark.json`.

## Reproduction

```bat
python src\main.py verify
python src\main.py train
python src\main.py benchmark
```

All execution uses a fresh subprocess per candidate, restricted AST imports, socket interception, an offline HTTP fixture, and a 2-second timeout. This is a constrained runner, not a formally secure container or a complete hostile-code isolation boundary.
