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
