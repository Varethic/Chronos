"""Two-stage curriculum training with real AdamW, cosine decay, and atomic persistence."""
import os, math, time, json, gc
import torch
from config import _settings, _paths, _atomicjson, _atomicmodel, _fingerprint, _stage, _schedule, _stages
from tokenizer import Tokenizer
from dataset import _generate, _prepare, _batch


def _state(cfg):
    """Model plus AdamW configured exactly as the protocol states."""
    from model import Chronos
    model = Chronos(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["decay"]), betas=(0.9, 0.95), eps=1e-8)
    return model, optimizer


def _save(model, optimizer, tokenizer, step, history, cfg, paths, attempts, skipped, elapsed):
    """Persist a checkpoint only when every optimizer state counter agrees with the step."""
    counts = [int(state.get("step", 0)) for state in optimizer.state.values()]
    if counts and min(counts) != step:
        raise RuntimeError("checkpoint diverged from real optimizer update counts")
    _atomicmodel({"name": "Chronos Alpha", "step": step, "_step": step, "attempts": attempts, "skipped": skipped, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg}, paths["model"])
    tokenizer._save(paths["vocab"])
    _atomicjson(cfg, paths["config"])
    _atomicjson(history, paths["history"])
    finite = [event["loss"] for event in history if not event.get("skipped")]
    metrics = {"name": "Chronos Alpha", "version": 2, "_step": step, "step": step, "attempts": attempts, "skipped": skipped, "parameters": model._paramcount, "dtype": cfg["dtype"], "cores": cfg["cores"], "threads": torch.get_num_threads(), "history_length": len(history), "elapsedseconds": elapsed, "initialwindowloss": sum(finite[:100]) / len(finite[:100]) if finite else None, "finalwindowloss": sum(finite[-100:]) / len(finite[-100:]) if finite else None, "stepspersecond": step / elapsed if elapsed else 0.0, "optimizerstates": len(counts), "stages": [{"stage": event.get("stage"), "sequence": event.get("length")} for event in history if event.get("ils")], "fingerprint": _fingerprint()}
    _atomicjson(metrics, paths["metrics"])


def _load(paths):
    """Restore checkpoint, tokenizer, and configuration together; refuse any mismatch."""
    with open(paths["config"], "r", encoding="utf-8") as stream:
        cfg = json.load(stream)
    state = torch.load(paths["model"], map_location="cpu", weights_only=True)
    if cfg.get("name") != "Chronos Alpha" or state.get("name") != "Chronos Alpha":
        raise ValueError("checkpoint identity mismatch")
    step = int(state.get("_step", -1))
    if state.get("step") != step or step < 0:
        raise ValueError("invalid optimizer step")
    tokenizer = Tokenizer()._load(paths["vocab"])
    if cfg["vocab"] != len(tokenizer):
        raise ValueError("checkpoint vocabulary mismatch")
    model, optimizer = _state(cfg)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    counts = [int(value.get("step", 0)) for value in optimizer.state.values()]
    if counts and min(counts) != step:
        raise ValueError("optimizer and checkpoint step mismatch")
    return model, optimizer, tokenizer, cfg, step


def _step(model, optimizer, data, cfg, step):
    """One optimizer step; `data` is either a loader or an explicit list of row groups."""
    stage = _stage(step, cfg)
    if isinstance(data, list):
        groups = data
    else:
        groups = [_batch(data, cfg, step)]
    width = int(stage["sequence"])
    trimmed = []
    for tokens, targets, masks in groups:
        if tokens.shape[1] > width:
            tokens, targets, masks = tokens[:, :width], targets[:, :width], masks[:, :width]
        trimmed.append((tokens, targets, masks))
    rate = _schedule(step, cfg)
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=True)
    total = sum(int(masks.sum()) for unused_tokens, unused_targets, masks in trimmed)
    loss, weighted = None, 0.0
    for tokens, targets, masks in trimmed:
        if int(masks.sum()) == 0:
            continue
        part = model._loss(tokens, masks, targets)
        weighted = weighted + part * (int(masks.sum()) / max(1, total))
        loss = part if loss is None else loss
    if loss is None:
        return {"attempt": step, "stage": stage["stage"], "tokens": 0, "length": 0, "lr": rate, "step": step, "skipped": True, "reason": "empty-batch"}
    loss = weighted
    event = {"attempt": step, "stage": stage["stage"], "tokens": total, "length": max(int(part[0].shape[1]) for part in trimmed), "lr": rate}
    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        event.update({"skipped": True, "reason": "nonfinite-loss", "loss": str(loss.detach())})
        return event
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip"]))
    if not torch.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        event.update({"skipped": True, "reason": "nonfinite-gradient", "loss": float(loss.detach())})
        return event
    before = model._embedding.weight.detach().clone()
    optimizer.step()
    with torch.no_grad():
        change = (model._embedding.weight - before).abs().max()
    if not torch.isfinite(change):
        raise RuntimeError("nonfinite model parameters after optimizer step")
    event.update({"loss": float(loss.detach()), "norm": float(norm), "changed": float(change), "updated": True, "ils": True})
    return event


def _stagebatch(cfg, stage, tokenizer, data):
    """Build the stage-two stream: long sequences interleaved with short replay so nothing is forgotten."""
    if int(stage["stage"]) != 2:
        return data
    from dataset import _tensors
    longs = [record for record in data["records"] if record["domain"] == "longcontext"]
    shorts = [record for record in data["records"] if record["domain"] != "longcontext"]
    if not longs:
        return data
    width = int(stage["sequence"])
    ltokens, ltargets, lmasks = _tensors(tokenizer, longs, cfg, width)
    stokens, stargets, smasks = _tensors(tokenizer, shorts, cfg, int(cfg["sequence"]))
    return {"records": longs + shorts, "long": {"tokens": ltokens, "targets": ltargets, "masks": lmasks}, "short": {"tokens": stokens, "targets": stargets, "masks": smasks}, "width": width, "batches": len(longs), "totaltokens": int(lmasks.sum())}


def _stagebatch_sample(source, generator, longsize, shortsize):
    """Draw a replay mix: long-context rows plus a rotation of short rows at equal weight."""
    picks = []
    for name, count in (("long", int(longsize)), ("short", int(shortsize))):
        part = source[name]
        pool = part["masks"].shape[0]
        if pool == 0 or count <= 0:
            continue
        index = torch.randint(0, pool, (min(count, pool),), generator=generator) if count < pool else torch.randperm(pool, generator=generator)[:count]
        rows = index.tolist()
        stack = torch.tensor(rows, dtype=torch.long) if rows else None
        if stack is None:
            continue
        cutoff = 1
        for row in rows:
            occupied = torch.nonzero(part["masks"][row] > 0).flatten()
            if occupied.numel():
                cutoff = max(cutoff, int(occupied[-1]) + 2)
        cutoff = max(1, min(cutoff, part["tokens"].shape[1]))
        picks.append((part["tokens"].index_select(0, stack)[:, :cutoff], part["targets"].index_select(0, stack)[:, :cutoff], part["masks"].index_select(0, stack)[:, :cutoff]))
    if not picks:
        return None
    return picks


def _train(target=None, resume=True, verify=True):
    """Run the mandated two-stage curriculum to completion with real optimizer steps only."""
    cfg = _settings()
    paths = _paths()
    if target is not None:
        cfg["stagetwo"] = max(int(cfg["stagetwo"]), int(target))
        cfg["steps"] = min(int(cfg["steps"]), int(cfg["stagetwo"]) - 1)
    completed = all(os.path.isfile(paths[name]) for name in ("model", "vocab", "config"))
    history = []
    if completed and resume:
        model, optimizer, tokenizer, cfg, start = _load(paths)
        if os.path.isfile(paths["history"]):
            with open(paths["history"], "r", encoding="utf-8") as stream:
                history = json.load(stream)
        print("resuming Chronos Alpha at real optimizer step", start, "target", cfg["stagetwo"], flush=True)
    else:
        start = 0
        records, holdout = _generate(None, int(cfg["retrievaltokens"]))
        print("fitting byte BPE on training records", flush=True)
        tokenizer = Tokenizer()._fit([_render(record) for record in records], int(cfg["vocab"]))
        cfg["vocab"] = len(tokenizer)
        model, optimizer = _state(cfg)
    data = _prepare(tokenizer, cfg, verify=verify)
    skipped = sum(1 for event in history if event.get("skipped"))
    _save(model, optimizer, tokenizer, start, history, cfg, paths, len(history), skipped, 0.0)
    model.train()
    torch.set_num_threads(int(cfg["cores"]))
    print(json.dumps({"name": "Chronos Alpha", "parameters": model._paramcount, "dtype": cfg["dtype"], "cores": cfg["cores"], "target": cfg["stagetwo"], "stages": _stages(cfg)}), flush=True)
    started = time.perf_counter()
    longdata, stage = None, _stage(max(1, start), cfg)["stage"]
    mixer = torch.Generator().manual_seed(int(cfg["seed"]) + 2)
    step = start
    while step < int(cfg["stagetwo"]):
        current = _stage(step + 1, cfg)
        if current["stage"] != stage:
            stage = current["stage"]
            longdata = _stagebatch(cfg, current, tokenizer, data)
            print(json.dumps({"entering": "stage" + str(stage), "sequence": current["sequence"], "longbatches": longdata["batches"]}), flush=True)
        if longdata is None:
            source = data
        else:
            parts = _stagebatch_sample(longdata, mixer, current["batch"], int(cfg["replayshort"]))
            source = parts if parts else data
        batch_started = time.perf_counter()
        event = _step(model, optimizer, source, cfg, step + 1)
        if not event.get("skipped"):
            step += 1
        else:
            skipped += 1
        event["step"] = step
        event["seconds"], event["elapsed"] = time.perf_counter() - batch_started, time.perf_counter() - started
        history.append(event)
        if len(history) % int(cfg["saveevery"]) == 0 or step == int(cfg["stagetwo"]):
            _save(model, optimizer, tokenizer, step, history, cfg, paths, len(history), skipped, time.perf_counter() - started)
        if step % 100 == 0 and event.get("updated"):
            print(json.dumps({"step": step, "stage": event["stage"], "loss": round(event["loss"], 5), "norm": round(event["norm"], 3), "lr": round(event["lr"], 8), "length": event["length"], "elapsed": round(event["elapsed"], 1)}), flush=True)
    _save(model, optimizer, tokenizer, step, history, cfg, paths, len(history), skipped, time.perf_counter() - started)
    print("training_complete", step, "attempts", len(history), "skipped", skipped, round(time.perf_counter() - started, 3), "seconds", flush=True)
    return {"model": model, "optimizer": optimizer, "tokenizer": tokenizer, "cfg": cfg, "data": data, "history": history, "step": step}


def _render(record):
    """Corpus text used to fit the vocabulary."""
    if record.get("messages"):
        return record["messages"][0][1] + record.get("code", "")
    return record.get("prompt", "") + record.get("code", "")


def _correction(model, optimizer, tokenizer, cfg, data, paths):
    """Supervised fault injection proving the self-repair loop really updates weights."""
    from antigravity import _autorepair
    record = next(item for item in data["records"] if item.get("code") and item.get("cases") and "range(1," in item["code"])
    failure = record["code"].replace("range(1,", "range(1,").replace("range(1,", "range(1, )", 1)
    if failure == record["code"]:
        failure = record["code"].replace(":", ";", 1)
    if failure == record["code"]:
        raise ValueError("fault injection did not alter the candidate source")
    before = model._embedding.weight.detach().clone()
    event = _autorepair(record, model, tokenizer, optimizer, cfg, failure, fixed=record["code"], supervised=True)
    if not event["passed"] or not event.get("training") or not event["training"].get("updated"):
        raise RuntimeError("supervised repair control must pass and update weights")
    event["injected"] = failure
    event["weightchanged"] = not torch.equal(before, model._embedding.weight)
    if not event["weightchanged"]:
        raise RuntimeError("supervised repair updated no weights")
    event["kind"] = "verified supervised fault injection"
    _atomicjson(event, paths["antigravity"])
    return event


if __name__ == "__main__":
    result = _train(None, resume=True)
    print("antigravity_control", json.dumps(_correction(result["model"], result["optimizer"], result["tokenizer"], result["cfg"], result["data"], _paths())), flush=True)
