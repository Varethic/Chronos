# Contributing to Chronos Alpha

Thanks for taking a look. This is a research prototype, so the bar is different from a production library: contributions are welcome, but **honesty about results matters more than impressive numbers**.

## Ground rules

1. **Never fabricate or estimate a metric.** If you change the model, retrain it and publish what you measured. A worse-but-real number beats a better-but-invented one. This is the single most important rule in this project.
2. **No mock evaluation.** `src/benchmark.py` scores generated output only. Never submit a change that substitutes a reference implementation, caches expected answers, or skips records.
3. **Keep the sandbox boundary.** Generated code is parsed and executed only inside `src/sandbox.py`, never in the parent process.

## Naming invariant (enforced in CI)

The project follows `AGENTS.md`, and `python src\main.py verify` fails the build when these are violated:

| Rule | Detail |
|---|---|
| Root directories | exactly `docs/` and `src/` |
| `src/` subdirectories | exactly `data/`, `json/`, `models/` |
| Filenames | lowercase alphanumeric plus extension: `no_hyphens.py`, no `under_scores.py` |
| Function names | must start with `_` and be lowercase: `def _train():` |
| Nested helpers | double underscore: `def __step():` |
| Exceptions | dunders (`__init__`, `__len__`, `__getitem__`) and `nn.Module.forward` |
| Imports | one line per module category, never a multiline block |

```python
import os, sys, math, time, gc, json, ast, subprocess, re, csv, tempfile, random
import torch, torch.nn as nn, torch.nn.functional as F
```

Artifact placement is also fixed: datasets as `.json` in `src/data/`, metadata and logs as `.json` in `src/json/`, checkpoints as `.pt` in `src/models/`. Temporary files are written beside their target and moved with `os.replace`.

## Development setup

```bat
git clone https://github.com/Varethic/Chronos.git
cd Chronos
pip install torch
python src\main.py verify
python src\main.py check
```

`check` runs the fast unit-level proofs (tokenizer losslessness, cache equivalence, masking, sandbox isolation, atomic persistence) in about a minute. Run it before every commit.

## What to work on

The most valuable contributions right now:

- **Evaluation honesty.** Anything that makes `docs/report.md` more truthful: better scoring, clearer per-domain breakdowns, failure analysis.
- **Long-context correctness.** Tests that prove the sliding-window cache and attention sinks stay equivalent to full attention at 8k, 32k, and 128k tokens.
- **Sandbox hardening.** Additional AST guards, an optional OS-level memory cap, or a container-based execution backend.
- **Smaller footprint.** Reductions in parameter count that do not cost accuracy are very welcome, since the model currently sits above its original size target.

If you want to change the architecture or the curriculum, open an issue first. Training takes hours, and a rejected architecture change is expensive to discover afterwards.

## Submitting a change

1. Fork and branch from `main`.
2. Keep changes focused; one concern per pull request.
3. Run `python src\main.py verify` and `python src\main.py check`. Both must pass.
4. If you changed the model, data, or training: retrain, run the full 600-record benchmark, and include the real before/after numbers in the PR description, including regressions.
5. If you changed `docs/report.md`, append; do not rewrite existing runs. History is the point.

### Pull request checklist

- [ ] `python src\main.py verify` passes
- [ ] `python src\main.py check` passes
- [ ] No invented or estimated metrics anywhere
- [ ] Naming invariants respected (no new violations)
- [ ] If behaviour changed, `docs/report.md` or `docs/audit.md` updated with measured numbers
- [ ] New code has a docstring explaining why, not just what

## Reporting a bug

Include the command you ran, the full traceback, your Python and PyTorch versions, and how many CPU cores PyTorch reported. For training issues, include the last few `src/json/history.json` entries so the loss trajectory is visible.

Security issues go through [SECURITY.md](SECURITY.md), not the public issue tracker.

## Code style

Match the existing files. Docstrings are one line and describe intent. Prefer explicitness over cleverness: this code is read far more often than it is run. Comments explain *why* a decision was made, especially where a naive approach is faster but wrong (for example, why length-bucketed batching exists).

## License

By contributing you agree that your work is licensed under the MIT License in [LICENSE](LICENSE).
