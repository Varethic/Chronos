"""Closed-loop error attribution, verified correction, and high-priority replay."""
import os, ast, time, traceback, difflib
import torch, torch.nn.functional as F
from sandbox import _execute
from config import _atomicjson
from state import State, _hash



def _cause(source, error, tb):
    """Locate the most specific executable AST node indicated by a traceback."""
    line = None
    for text in reversed(tb.splitlines()):
        text = text.strip()
        if text.startswith("line ") and text[5:].split(",", 1)[0].isdigit():
            line = int(text[5:].split(",", 1)[0])
            break
    node = None
    try:
        tree = ast.parse(source)
        candidates = [item for item in ast.walk(tree) if hasattr(item, "lineno") and line is not None and item.lineno <= line <= getattr(item, "end_lineno", item.lineno)]
        if candidates:
            node = next(iter(sorted(candidates, key=lambda item: (getattr(item, "end_lineno", item.lineno) - item.lineno, getattr(item, "end_col_offset", 0) - item.col_offset))))
    except SyntaxError:
        pass
    return {"line": line, "node": type(node).__name__ if node else "Source", "message": error, "traceback": tb}


def _repair(source, error):
    """Deterministic repairs act on the diagnosed source, never hidden references."""
    fixed = source
    rule = "diagnosed"
    if "unexpected EOF while parsing" in error or "expected an indented block" in error:
        fixed = fixed.rstrip() + "\n    return None\n"
        rule = "add-missing-return"
    elif "return _solve(" in fixed and "recursion" in error.lower():
        raise ValueError("unsafe automatic recursive repair requires human review")
    elif "NameError" in error:
        missing = error.split("name '", 1)[-1].split("'", 1)[0]
        if missing.isidentifier() and not missing.startswith("__"):
            fixed = fixed.replace(missing, "0")
            rule = "name-initialization"
    elif "return(out)" in fixed:
        fixed = fixed.replace("return(out)", "return out")
        rule = "return-call"
    elif "TypeError: _solve()" in error and "takes 0 positional arguments" in error:
        signature = _arguments(error)
        if signature:
            fixed = fixed.replace("def _solve():", "def _solve(" + signature + "):")
            rule = "argument-signature"
    elif "WrongAnswer" in error:
        raise ValueError("semantic assertion failures require validated supervision")
    if fixed == source:
        raise ValueError("no verified deterministic repair rule for " + error)
    ast.parse(fixed)
    return fixed, rule


def _arguments(error):
    if "but " not in error:
        return ""
    count = error.split("but ", 1)[-1].split(" ", 1)[0].strip()
    if not count.isdigit() or int(count) > 8:
        return ""
    return ", ".join("arg" + str(index) for index in range(int(count)))


def _trajectory(record, source, failure, fixed, history="", tokenizer=None):
    cause = _cause(source, failure["error"], failure["traceback"])
    small = tokenizer._encode(record["prompt"] + history + failure["error"] + str(cause["line"]) + cause["node"], controls=False)
    ids = tokenizer._encode(source, controls=False) + [3] + tokenizer._encode(fixed, controls=False) + [4]
    start = max(0, len(ids) + len(small) - 254)
    return {"tokens": small + ids[start:], "prompt": record["prompt"], "history": history, "faulty": source, "fixed": fixed}


def _trainpair(pair, tokenizer, model, optimizer, cfg):
    model.train()
    ids = list(pair) if not isinstance(pair, str) else tokenizer._encode(pair, controls=False)
    if len(ids) - 1 > cfg["sequence"]:
        raise ValueError("repair trajectory exceeds context; no truncated supervision")
    inputs = ids[:-1]
    targets = ids[1:]
    if len(inputs) < 2:
        raise ValueError("repair trajectory too short")
    ids = torch.tensor([inputs], dtype=torch.long)
    labels = torch.tensor([targets], dtype=torch.long)
    weights = torch.ones_like(ids, dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    loss = model._loss(ids, weights, labels) * 2.0
    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return {"loss": str(loss.detach()), "updated": False, "nonfiniteloss": True}
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
    if not torch.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        return {"loss": float(loss.item()), "updated": False, "nonfinitegradient": True}
    optimizer.step()
    return {"loss": float(loss.item()), "updated": True, "priority": 2.0, "length": len(inputs)}


def _autorepair(record, model, tokenizer, optimizer, cfg, source=None, fixed=None, supervised=False, state=None):
    """Verify each stateful repair before replay; never train a rejected correction."""
    if supervised and fixed is None:
        raise ValueError("verified supervision must supply candidate corrected code")
    generated = source if source is not None else model._generate(tokenizer, record["prompt"], temperature=0.2)["code"]
    memory, attempts, current = State() if state is None else state, [], generated
    for cycle in range(3):
        failure = _execute(current, record["cases"], timeout=cfg["timeout"])
        cause = _cause(current, failure["error"], failure["traceback"])
        repeated = memory._seen(current, failure["error"], failure["traceback"])
        attempt = {"cycle": cycle, "code": current, "failure": failure, "cause": cause, "repeated": repeated, "memoryused": len(memory._attempts)}
        attempts.append(attempt)
        if failure["passed"]:
            memory._commit(current, {"passed": True})
            return {"passed": True, "cycles": cycle, "attempts": attempts, "training": None, "state": memory._items()}
        if repeated:
            attempt["repairerror"] = "RepeatedFailure: identical source/error rejected"
            break
        memory._record(current, failure["error"], failure["traceback"], cause["line"], cause["node"])
        try:
            correction, rule = (fixed, "verified-supervision") if supervised else _repair(current, failure["error"])
            memory._patch(current, failure, correction, None)
            attempt["rule"], attempt["fixed"] = rule, correction
            verified = _execute(correction, record["cases"], timeout=cfg["timeout"])
            if verified["passed"]:
                memory._patch(current, failure, correction, True)
                pair = _trajectory(record, current, failure, correction, memory._history(), tokenizer)
                attempt["trajectory"] = pair
                attempt["training"] = _trainpair(pair["tokens"], tokenizer, model, optimizer, cfg)
                if attempt["training"]["updated"]:
                    memory._commit(correction, verified)
                return {"passed": True, "cycles": cycle + 1, "attempts": attempts, "training": attempt["training"], "state": memory._items(), "rollback": memory._rollback()}
            current = correction
        except (ValueError, SyntaxError, AttributeError, IndexError) as problem:
            attempt["repairerror"] = type(problem).__name__ + ": " + str(problem)
    return {"passed": False, "cycles": len(attempts), "attempts": attempts, "training": None, "state": memory._items(), "rollback": memory._rollback()}
