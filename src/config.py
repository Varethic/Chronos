"""Chronos Alpha v2 runtime configuration: topology, long-context limits, curriculum, and artifact paths."""
import os, json, math, time, random, hashlib
import torch


def _bind():
    """Bind PyTorch to physical CPU cores so hyperthread siblings are never oversubscribed."""
    cores = max(1, (os.cpu_count() or 4) // 2)
    torch.set_num_threads(cores)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return cores


def _speed(dtype):
    """Time a fixed square matmul in one precision; lower is faster."""
    left, right = torch.randn(512, 512), torch.randn(512, 512)

    def __run():
        left @ right
        for unused in range(6):
            left @ right

    if dtype == torch.bfloat16:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            started = time.perf_counter()
            __run()
            return time.perf_counter() - started
    started = time.perf_counter()
    __run()
    return time.perf_counter() - started


def _precision():
    """Return bfloat16 only when this CPU actually accelerates it, otherwise float32."""
    if not torch.backends.mkldnn.is_available():
        return torch.float32
    try:
        probe = torch.nn.Linear(16, 16).to(torch.bfloat16)(torch.zeros(1, 16, dtype=torch.bfloat16))
        if probe.dtype != torch.bfloat16:
            return torch.float32
        return torch.bfloat16 if _speed(torch.bfloat16) < _speed(torch.float32) * 0.9 else torch.float32
    except (RuntimeError, TypeError):
        return torch.float32


def _settings():
    """Resolve the frozen Chronos Alpha v2 topology, long-context limits, and curriculum."""
    cores = _bind()
    seed = 17041
    random.seed(seed)
    torch.manual_seed(seed)
    return {
        "name": "Chronos Alpha", "version": 2, "seed": seed, "cores": cores, "dtype": str(_precision()).split(".")[-1],
        "vocab": 4096, "dim": 384, "layers": 8, "heads": 6, "kvheads": 2, "ff": 1024, "dropout": 0.1,
        "sequence": 1024, "stagesequence": 4096, "chunk": 2048, "attnblock": 512, "window": 2048, "sinks": 16,
        "maxinput": 131072, "maxcontext": 1048576, "theta": 500000.0, "scaling": 32.0, "ropebase": 4096,
        "batch": 16, "stagebatch": 2, "replayshort": 12, "bucket": 256, "stagelr": 0.00003, "steps": 6000, "stagetwo": 8000, "lr": 0.0003, "warmup": 400, "decay": 0.1, "clip": 1.0,
        "timeout": 2.0, "replay": 3.0, "generate": 384, "reply": 256, "temperature": 0.0, "saveevery": 50,
        "trainrecords": 6000, "holdoutrecords": 600, "needles": 50, "retrievaltokens": 152000, "checkpoints": 4,
    }


def _ensure():
    """Create the three permitted artifact directories beneath src."""
    root = os.path.dirname(os.path.abspath(__file__))
    for name in ("data", "json", "models"):
        os.makedirs(os.path.join(root, name), exist_ok=True)
    return root


def _paths():
    """Resolve every artifact to its mandated location under src/data, src/json, or src/models."""
    root = _ensure()
    stored = {"vocab": "vocab.json", "config": "config.json", "metrics": "metrics.json", "history": "history.json", "antigravity": "antigravity.json", "curriculum": "curriculum.json"}
    binary = {"model": "model.pt"}
    corpus = {"manifest": "manifest.json", "train": "traindata.json", "holdout": "testdata.json", "benchmark": "benchmark.json"}
    return {**{key: os.path.join(root, "json", name) for key, name in stored.items()}, **{key: os.path.join(root, "models", name) for key, name in binary.items()}, **{key: os.path.join(root, "data", name) for key, name in corpus.items()}}


def _atomicjson(value, path):
    """Write JSON beside its target, fsync, then atomically replace the destination."""
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomicmodel(value, path):
    """Write a binary checkpoint beside its target, fsync, then atomically replace it."""
    temporary = path + ".tmp"
    with open(temporary, "wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _digest(texts):
    return hashlib.sha256("\\0".join(texts).encode("utf-8")).hexdigest()


def _fingerprint():
    """Hash every module so metrics can prove which sources produced a checkpoint."""
    root = os.path.dirname(os.path.abspath(__file__))
    names = ("config.py", "tokenizer.py", "model.py", "dataset.py", "state.py", "sandbox.py", "antigravity.py", "trainer.py", "benchmark.py", "main.py")
    versions = []
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            versions.append("missing:" + name)
            continue
        with open(path, "rb") as stream:
            versions.append(hashlib.sha256(stream.read()).hexdigest())
    return {"names": list(names), "versions": versions, "digest": _digest(versions)}


def _stages(cfg):
    """Two-stage curriculum: stage one is sequence-length 1024, stage two jumps to 4096."""
    if int(cfg["stagetwo"]) <= int(cfg["steps"]):
        raise ValueError("stage two must extend beyond stage one")
    return [{"stage": 1, "first": 1, "last": int(cfg["steps"]), "sequence": int(cfg["sequence"]), "batch": int(cfg["batch"])}, {"stage": 2, "first": int(cfg["steps"]) + 1, "last": int(cfg["stagetwo"]), "sequence": int(cfg["stagesequence"]), "batch": int(cfg["stagebatch"])}]


def _stage(step, cfg):
    """Map a 1-based optimizer step to its curriculum stage record."""
    for stage in _stages(cfg):
        if stage["first"] <= step <= stage["last"]:
            return stage
    raise ValueError("optimizer step outside the declared curriculum")


def _schedule(step, cfg):
    """Cosine decay with linear warmup over the whole two-stage schedule."""
    total = max(1, int(cfg["stagetwo"]))
    warmup = max(1, int(cfg["warmup"]))
    if step <= warmup:
        return float(cfg["lr"]) * step / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    floor = float(cfg["lr"]) * float(cfg["decay"])
    cosine = floor + 0.5 * (float(cfg["lr"]) - floor) * (1.0 + math.cos(math.pi * progress))
    ceiling = float(cfg["stagelr"]) if int(step) > int(cfg["steps"]) else float(cfg["lr"])
    return min(cosine, ceiling)
