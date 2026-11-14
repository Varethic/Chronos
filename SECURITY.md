# Security Policy

## Reporting a vulnerability

Report security issues privately to **varethic@users.noreply.github.com**. Do not open a public issue for anything that could let generated code escape the sandbox, exhaust host resources, or exfiltrate data.

Please include:

- A description of the issue and its impact
- Steps to reproduce, ideally a prompt or code sample
- Your Python and PyTorch versions, and operating system
- Whether you believe the issue is exploitable without local code execution

You can expect an initial acknowledgement within 7 days. This is a single-maintainer research project, so there is no formal SLA beyond that.

## Supported versions

Only the latest commit on `main` is supported. There are no maintained release branches.

---

## Threat model

Chronos Alpha **executes code that a language model generated**. Everything in this section follows from that fact.

### What the sandbox protects against

`src/sandbox.py` runs candidate code in a subprocess, in a fresh interpreter, inside `tempfile.TemporaryDirectory()` with `cwd` set to that directory, under a hard 2.0-second deadline. It guards against:

| Risk | Mitigation |
|---|---|
| Infinite loops | Wall-clock timeout, process killed on expiry |
| Runaway memory allocation | Process isolation; the child dies without taking the parent down |
| Filesystem writes outside the temp directory | `cwd` is a temporary directory that is deleted afterwards |
| Network calls during grading | Transport modules are patched lazily, only when the candidate references them |
| Process spawning / `eval` / `exec` / `__import__` abuse | Static AST guard rejects these before execution |
| Output that is correct-looking but wrong | Results compared behaviorally against the reference oracle |

### What the sandbox does **not** protect against

Read this list before running Chronos on untrusted input.

1. **It is a guardrail, not a security boundary.** `src/sandbox.py` uses static AST filtering plus subprocess isolation. It is not a container, VM, seccomp jail, or OS-level sandbox. A determined adversary who controls the model or the prompt may find a bypass. Do not run the benchmark on hostile input on a machine that holds credentials.
2. **The child process inherits your user privileges.** It runs as you. It can read any file your user can read, even though the AST guard blocks the obvious ways to do so.
3. **No memory cap is enforced by the OS.** The 2.0-second deadline bounds wall time, not allocation. A candidate that allocates several gigabytes will still do so before the timeout; you may see swapping or an OOM kill of the child.
4. **Environment variables are inherited.** If your environment holds tokens or keys, a sandbox escape would expose them. Run Chronos in a stripped environment or a disposable user account when evaluating untrusted models.
5. **The guardrail is a behavioural filter.** It checks whether a prompt looks Python-related with a keyword allowlist. It is not a security control and can be talked around. It exists to keep the model on-topic, not to classify adversaries.

**Recommendation:** run `python src\main.py benchmark` and `chat` inside a disposable VM or container if you are evaluating models or prompts you did not write.

---

## Model output is untrusted

Treat every byte the model generates as untrusted input.

- Generation output is passed to `ast.parse` and then to the sandbox. It is never `exec`'d in the parent process.
- `src/main.py chat` prints model output verbatim. It does not execute it.
- `src/benchmark.py` never substitutes a reference implementation for generated code. A generated sample is scored exactly as produced.

If you build on Chronos, keep this separation: parse and execute generated code only inside `src/sandbox.py`, never in your own process.

## Dependencies

The project depends on PyTorch only. HTTP-related training data references `requests`, `httpx`, `urllib3`, and `aiohttp`, but those are imported *inside the sandbox subprocess* for candidate code and are not required to run Chronos itself.

Keep PyTorch current. Older CPU builds have had memory-safety issues in tensor deserialization.

## Checkpoints

`src/models/model.pt` is loaded with:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

`weights_only=True` restricts deserialization to tensors and basic types, which prevents arbitrary code execution from a malicious checkpoint. **Do not remove this argument.** If you download a checkpoint from an untrusted source and load it with `weights_only=False`, you are giving that file code execution on your machine.

`src/json/*.json` files are parsed with `json.load`, which is not a code execution vector.

## Known limitations

These are documented rather than fixed, because fixing them would change published benchmark behaviour:

- The 2.0-second sandbox timeout is generous for the probe sizes used here; a slow candidate may pass a case that a stricter deadline would fail.
- `_compare` uses behavioural comparison with tolerance, so a candidate returning an almost-correct float may be scored as passing.
- Benchmark results in `docs/report.md` reflect the checked-in checkpoint only. A different checkpoint will produce different numbers.

## Disclosure

Once a fix is available, vulnerabilities are described in `docs/audit.md` alongside the mitigation, without publishing a working exploit.
