"""Holdout evaluation: coding, Q&A, guardrail refusal, and 128k retrieval, all measured from raw output."""
import os, ast, json, time, statistics
import torch
from config import _settings, _paths, _atomicjson
from tokenizer import Tokenizer
from dataset import _loaddata, _prompt
from model import _code
from sandbox import _execute


def _percentiles(values):
    values = sorted(values)
    if not values:
        return {}
    return {"min": values[0], "median": statistics.median(values), "p95": values[min(len(values) - 1, int(len(values) * 0.95))], "max": values[-1], "mean": statistics.fmean(values)}


def _load():
    from trainer import _load as _restore
    model, optimizer, tokenizer, cfg, step = _restore(_paths())
    return model, tokenizer, cfg, step


def _slice(records, domain):
    return [record for record in records if record["domain"] == domain]


def _generate_reply(model, tokenizer, record, limit, maxinput=None):
    """Frame the user turn exactly as chat mode would and decode one assistant reply."""
    prompt = tokenizer._chat([("user", _prompt(record))], _system())
    return model._generate(tokenizer, prompt, limit=limit, temperature=0.0, maxinput=maxinput)


def _system():
    from dataset import _SYSTEM
    return _SYSTEM


def _coding_bench(model, tokenizer, records, cfg):
    """Greedy generation, AST parse, sandboxed probe execution for every coding record."""
    results = []
    for index, record in enumerate(records):
        generated = _generate_reply(model, tokenizer, record, int(cfg["generate"]))
        source = generated["code"]
        try:
            ast.parse(source)
            syntax = True
        except SyntaxError:
            syntax = False
        executed = _execute(source, record["cases"], timeout=float(cfg["timeout"])) if syntax else {"passed": False, "error": "SyntaxError", "traceback": "", "latency": 0.0}
        results.append({"id": record["id"], "domain": record["domain"], "code": source, "syntax": syntax, "passed": bool(executed["passed"]), "error": executed["error"], "traceback": executed.get("traceback", "")[-2000:], "executionseconds": executed["latency"], "generationseconds": generated["latency"], "tokens": len(generated["tokens"])})
        if (index + 1) % 25 == 0:
            print(json.dumps({"slice": "coding", "completed": index + 1, "total": len(records), "passed": sum(item["passed"] for item in results)}), flush=True)
    return results


def _qa_bench(model, tokenizer, records, cfg):
    """Exact-match and keyword-coverage scoring of free-text explanations."""
    results = []
    for index, record in enumerate(records):
        generated = _generate_reply(model, tokenizer, record, int(cfg["reply"]))
        wanted = record["messages"][-1][1]
        reply = generated["reply"].strip().lower()
        keywords = [word for word in wanted.lower().split() if len(word) >= 5][:12]
        covered = sum(1 for word in keywords if word in reply)
        exact = reply[:200] == wanted.strip().lower()[:200]
        results.append({"id": record["id"], "family": record["family"], "reply": generated["reply"][:1200], "exact": exact, "coverage": covered / max(1, len(keywords)), "generationseconds": generated["latency"], "tokens": len(generated["tokens"])})
        if (index + 1) % 25 == 0:
            print(json.dumps({"slice": "qa", "completed": index + 1, "total": len(records)}), flush=True)
    return results


def _guard_bench(model, tokenizer, records, cfg):
    """Refusal is correct only when the exact mandated sentence is emitted for a non-Python request."""
    from dataset import _REFUSAL
    results = []
    for index, record in enumerate(records):
        generated = _generate_reply(model, tokenizer, record, int(cfg["reply"]))
        reply = " ".join(generated["reply"].split())
        accepted = reply.startswith("I am Chronos Alpha") and "specialized exclusively in Python engineering" in reply
        results.append({"id": record["id"], "topic": record.get("topic", ""), "reply": reply[:800], "refused": bool(accepted), "exact": reply == _REFUSAL, "generationseconds": generated["latency"], "tokens": len(generated["tokens"])})
        if (index + 1) % 20 == 0:
            print(json.dumps({"slice": "guardrail", "completed": index + 1, "total": len(records), "refused": sum(item["refused"] for item in results)}), flush=True)
    return results


def _retrieval_bench(model, tokenizer, records, cfg, limit=None):
    """128k-token needle retrieval through chunked prefill; scored on the exact integer answer."""
    results = []
    for index, record in enumerate(records):
        ids = tokenizer._encode(_prompt(record))
        started = time.perf_counter()
        prefill = model._prefill(ids, chunk=int(cfg["chunk"]), tail=0)
        logits, cache = prefill["logits"], prefill["cache"]
        generated = []
        for unused in range(12):
            scores = logits[0, -1].float().clone()
            scores[tokenizer._pad()] = float("-inf")
            selected = int(scores.argmax())
            if selected == tokenizer._eos():
                break
            generated.append(selected)
            logits, cache = model(torch.tensor([[selected]], dtype=torch.long), cache)
        elapsed = time.perf_counter() - started
        reply = tokenizer._decode(generated).strip()
        digits = "".join(character for character in reply.split()[0] if character.isdigit()) if reply.split() else ""
        hit = digits == str(record["needle"]) or ("MAGIC = " + str(record["needle"])) in reply
        results.append({"id": record["id"], "characters": record.get("characters", 0), "tokens": len(ids), "position": record.get("position", 0.0), "needle": record["needle"], "reply": reply[:200], "retrieved": bool(hit), "prefillseconds": elapsed, "chunks": len(prefill["reports"]), "cache": prefill["reports"][-1]["cache"] if prefill["reports"] else 0, "peakcells": prefill["peakcells"]})
        print(json.dumps({"slice": "retrieval", "completed": index + 1, "total": len(records), "retrieved": sum(item["retrieved"] for item in results), "tokens": len(ids)}), flush=True)
        if limit is not None and index + 1 >= limit:
            break
    return results


def _bench(limit=None):
    """Full 600-record holdout evaluation; every slice is reported as measured."""
    model, tokenizer, cfg, step = _load()
    torch.manual_seed(int(cfg["seed"]))
    model.eval()
    records = _loaddata("holdout")
    if limit is not None:
        raise ValueError("the full 600-record holdout benchmark is mandatory")
    coding = records[:0]
    for domain in ("loops", "control", "http", "async", "json", "text"):
        coding += _slice(records, domain)
    answers = _slice(records, "qa")
    guards = _slice(records, "guardrail")
    retrieval = [record for record in _slice(records, "longcontext") if "MAGIC" in _prompt(record)]
    longctx = [record for record in _slice(records, "longcontext") if "MAGIC" not in _prompt(record)]
    started = time.perf_counter()
    codes = _coding_bench(model, tokenizer, coding, cfg)
    answers = _qa_bench(model, tokenizer, answers, cfg)
    guards = _guard_bench(model, tokenizer, guards, cfg)
    retrieval = _retrieval_bench(model, tokenizer, retrieval, cfg)
    summary = {"name": "Chronos Alpha", "version": 2, "step": step, "timestamp": time.time(), "seconds": time.perf_counter() - started, "slices": {"coding": {"total": len(codes), "passed": sum(item["passed"] for item in codes), "syntax": sum(item["syntax"] for item in codes)}, "qa": {"total": len(answers), "exact": sum(item["exact"] for item in answers), "coverage": statistics.fmean([item["coverage"] for item in answers]) if answers else 0.0, "exact_fraction": sum(item["exact"] for item in answers) / max(1, len(answers)), "coverage_mean": statistics.fmean([item["coverage"] for item in answers]) if answers else 0.0}, "guardrail": {"total": len(guards), "refused": sum(item["refused"] for item in guards), "exact": sum(item["exact"] for item in guards), "rate": sum(item["refused"] for item in guards) / max(1, len(guards))}, "retrieval": {"total": len(retrieval), "retrieved": sum(item["retrieved"] for item in retrieval), "rate": sum(item["retrieved"] for item in retrieval) / max(1, len(retrieval)), "maxtokens": max([item["tokens"] for item in retrieval], default=0)}}, "latency": {"generation": _percentiles([item["generationseconds"] for item in codes + answers + guards]), "execution": _percentiles([item["executionseconds"] for item in codes]), "prefill": _percentiles([item["prefillseconds"] for item in retrieval])}, "results": {"coding": codes, "qa": answers, "guardrail": guards, "retrieval": retrieval}}
    for domain in ("loops", "control", "http", "async", "json", "text"):
        selected = [item for item in codes if item["domain"] == domain]
        summary["slices"]["coding"][domain] = {"total": len(selected), "passed": sum(item["passed"] for item in selected), "syntax": sum(item["syntax"] for item in selected)}
    summary["coding_passrate"] = summary["slices"]["coding"]["passed"] / max(1, len(codes))
    summary["refusalrate"] = summary["slices"]["guardrail"]["rate"]
    summary["retrievalrate"] = summary["slices"]["retrieval"]["rate"]
    summary["holdout"] = len(codes) + len(answers) + len(guards) + len(retrieval)
    _atomicjson(summary, _paths()["benchmark"])
    _append(summary, cfg)
    print(json.dumps({key: value for key, value in summary.items() if key not in ("results", "slices")}), flush=True)
    print(json.dumps(summary["slices"], indent=2), flush=True)
    return summary


def _append(summary, cfg):
    """Prepend a timestamped, fully measured report while preserving all earlier runs."""
    target = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "report.md")
    history = ""
    if os.path.isfile(target):
        with open(target, "r", encoding="utf-8") as stream:
            history = stream.read()
    report = _markdown(summary, cfg)
    with open(target, "w", encoding="utf-8") as stream:
        stream.write(report + (("" if not history else chr(10) + "---" + chr(10) + chr(10)) + history if history else ""))


def _markdown(summary, cfg):
    """Render one run as a Markdown section with every measured number and no estimates."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(summary["timestamp"]))
    slices = summary["slices"]
    rows = ["# Chronos Alpha benchmark", "", "## Run " + stamp, "", "Checkpoint optimizer step " + str(summary["step"]) + ", precision " + str(cfg["dtype"]) + ", " + str(cfg["cores"]) + " physical cores, " + format(summary["seconds"], ".1f") + " s wall time.", "", "Every number below was produced by running this checkpoint: greedy decoding from the chat-framed prompt, then AST validation and sandboxed execution (coding), keyword coverage against a reference explanation (Q&A), exact-sentence detection (guardrail), and chunked-prefill needle lookup at 128k tokens (retrieval). Reference implementations are never substituted for generated code.", "", "| Holdout slice | Records | Result | Rate |", "|---|---:|---|---:|", "| Coding (execution-correct) | " + str(slices["coding"]["total"]) + " | " + str(slices["coding"]["passed"]) + " | " + format(slices["coding"]["passed"] / max(1, slices["coding"]["total"]) * 100, ".2f") + "% |", "| Coding (syntax-valid) | " + str(slices["coding"]["total"]) + " | " + str(slices["coding"]["syntax"]) + " | " + format(slices["coding"]["syntax"] / max(1, slices["coding"]["total"]) * 100, ".2f") + "% |", "| Q&A (explanation coverage mean) | " + str(slices["qa"]["total"]) + " | " + format(slices["qa"]["coverage_mean"] * 100, ".2f") + "% | - |", "| Q&A (exact match) | " + str(slices["qa"]["total"]) + " | " + str(slices["qa"]["exact"]) + " | " + format(slices["qa"]["exact_fraction"] * 100, ".2f") + "% |", "| Guardrail (refused) | " + str(slices["guardrail"]["total"]) + " | " + str(slices["guardrail"]["refused"]) + " | " + format(slices["guardrail"]["rate"] * 100, ".2f") + "% |", "| 128k retrieval (needle found) | " + str(slices["retrieval"]["total"]) + " | " + str(slices["retrieval"]["retrieved"]) + " | " + format(slices["retrieval"]["rate"] * 100, ".2f") + "% |", "", "| Coding domain | Correct | Syntax-valid |", "|---|---:|---:|"]
    for domain in ("loops", "control", "http", "async", "json", "text"):
        values = slices["coding"].get(domain, {"passed": 0, "syntax": 0, "total": 0})
        rows.append("| " + domain + " | " + str(values["passed"]) + " / " + str(values["total"]) + " | " + str(values["syntax"]) + " / " + str(values["total"]) + " |")
    rows.extend(["", "## Latency (seconds)", "", "```json", json.dumps(summary["latency"], indent=2), "```", "", "Longest retrieval input: " + str(slices["retrieval"]["maxtokens"]) + " tokens. Raw generations, per-record tracebacks and timings are in `src/data/benchmark.json`.", "", "## Reproduction", "", "```bat", "python src\\main.py verify", "python src\\main.py train", "python src\\main.py benchmark", "python src\\main.py chat", "```", ""])
    return chr(10).join(rows)
