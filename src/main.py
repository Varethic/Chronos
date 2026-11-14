"""Chronos Alpha command entry point: verification, dataset build, training, benchmark, and interactive chat."""
import os, sys, ast, json, time, hashlib, argparse
import torch
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from config import _settings, _paths, _ensure
from state import State

_USAGE = "usage: python src/main.py [verify|check|dataset|train|benchmark|stepcheck|chat]"


def _verify():
    """Assert reproducible corpus constraints and strict naming invariants from AGENTS.md."""
    from dataset import _generate, _TRAINCODE, _TRAINQA, _TRAINOOD, _TRAINLONG, _HOLDCODE, _HOLDQA, _HOLDOOD, _HOLDLONG
    fail = []
    paths = _paths()
    project = os.path.dirname(_ROOT)
    rootdirs = sorted(name for name in os.listdir(project) if os.path.isdir(os.path.join(project, name)) and not name.startswith(".") and name != "__pycache__")
    if rootdirs != ["docs", "src"]:
        fail.append("root directories must be exactly docs and src: " + ",".join(rootdirs))
    subdirs = sorted(name for name in os.listdir(_ROOT) if os.path.isdir(os.path.join(_ROOT, name)) and name != "__pycache__")
    if subdirs != ["data", "json", "models"]:
        fail.append("src subdirectories must be exactly data, json, models: " + ",".join(subdirs))
    train, hold = _generate(None, 0)
    if len(train) != _TRAINCODE + _TRAINQA + _TRAINOOD + _TRAINLONG:
        fail.append("training partition size " + str(len(train)))
    if len(hold) != _HOLDCODE + _HOLDQA + _HOLDOOD + _HOLDLONG:
        fail.append("holdout partition size " + str(len(hold)))
    if {record["id"] for record in train} & {record["id"] for record in hold}:
        fail.append("partitions share identifiers")
    trainprompts = {record["prompt"] for record in train if record.get("prompt")}
    holdprompts = {record["prompt"] for record in hold if record.get("prompt")}
    if trainprompts & holdprompts:
        fail.append("partition prompt overlap")
    for key, expected in (("train", len(train)), ("holdout", len(hold))):
        if os.path.isfile(paths[key]):
            with open(paths[key], "r", encoding="utf-8") as stream:
                stored = json.load(stream)
            if len(stored) != expected:
                fail.append("serialized partition " + key + " holds " + str(len(stored)) + " records, expected " + str(expected))
    functions = 0
    for base in (_ROOT, os.path.join(project, "docs")):
        for directory, dirs, files in os.walk(base):
            dirs[:] = [name for name in dirs if name != "__pycache__"]
            for name in files:
                if name.lower() != name or not all(character.isalnum() or character == "." for character in name):
                    fail.append("bad filename " + os.path.relpath(os.path.join(directory, name), project))
                if name.endswith(".py"):
                    try:
                        with open(os.path.join(directory, name), "r", encoding="utf-8") as stream:
                            tree = ast.parse(stream.read())
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                functions += 1
                                allowed = node.name.startswith("__") and node.name.endswith("__") or node.name == "forward"
                                if not allowed and (not node.name.startswith("_") or not node.name.replace("_", "").islower()):
                                    fail.append("bad function " + name + ":" + node.name)
                    except (OSError, UnicodeError, SyntaxError) as error:
                        fail.append(str(error))
    for folder, suffix in (("data", ".json"), ("json", ".json"), ("models", ".pt")):
        for name in os.listdir(os.path.join(_ROOT, folder)):
            if name.endswith(".tmp"):
                continue
            if not name.endswith(suffix):
                fail.append(folder + " holds a non-" + suffix + " artifact: " + name)
    missing = [name for name in ("model", "vocab", "config", "metrics") if not os.path.isfile(paths[name])]
    report = {"structural": not fail, "failures": fail, "train": len(train), "holdout": len(hold), "functiondefs": functions, "modelpresent": not missing}
    print(json.dumps(report, indent=2), flush=True)
    return fail


def _check():
    """Unit-level proofs of tokenizer losslessness, cache equivalence, masking, sandbox and persistence."""
    from tokenizer import Tokenizer
    from model import Chronos
    from sandbox import _execute
    from antigravity import _autorepair
    cfg = _settings()
    tokenizer = Tokenizer()._fit(["def _solve(x):\n    return x + 1\n" * 20, "π 😀 plain text" * 20], 400)
    for text in ("", "hello", "π 😀 é" + chr(10), "def _solve(x):" + chr(10) + chr(9) + "return x + 1" + chr(10), chr(0x10000) * 3):
        assert tokenizer._decode(tokenizer._encode(text, False)) == text
    framed = tokenizer._encode("<|user|>hi<|assistant|>", False)
    assert tokenizer._user() in framed and tokenizer._assistant() in framed
    assert tokenizer._decode(framed) == "hi"
    stored = tokenizer._tokens
    paths = _paths()
    tokenizer._save(paths["vocab"] + ".check.json")
    assert Tokenizer()._load(paths["vocab"] + ".check.json")._tokens == stored
    os.remove(paths["vocab"] + ".check.json")
    for index, name in enumerate(("<pad>", "<bos>", "<eos>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>", "<|code|>", "<|endcode|>")):
        assert tokenizer._tokens[index] == name
    cfg["vocab"] = len(tokenizer)
    model = Chronos(cfg).eval()
    full = Chronos(_settings()).eval()
    assert 10000000 <= full._paramcount <= 15000000, full._paramcount
    assert cfg["cores"] == torch.get_num_threads()
    tokens = torch.randint(9, cfg["vocab"], (1, 256))
    whole, _ = model(tokens)
    cache, pieces = None, []
    for position in range(256):
        piece, cache = model(tokens[:, position:position + 1], cache)
        pieces.append(piece)
    cached = torch.cat(pieces, dim=1)
    error = float((whole - cached).abs().max().detach())
    assert error < 1e-3, error
    chunked, cache = model(tokens[:, :128], None)
    tail, cache = model(tokens[:, 128:], cache)
    assert float((whole - torch.cat((chunked, tail), dim=1)).abs().max()) < 1e-3
    for layer in cache["blocks"]:
        assert int(layer["recent"][0].shape[2]) <= int(cfg["window"])
    uncapped = dict(cfg)
    uncapped["window"] = 32
    uncapped["sinks"] = 4
    small = Chronos(uncapped).eval()
    long = torch.randint(9, cfg["vocab"], (1, 200))
    _, state = small(long, None)
    for layer in state["blocks"]:
        assert int(layer["recent"][0].shape[2]) <= 32
        assert int(layer["span"]) <= 4
    normal = _execute("def _solve(values):\n    return [x * 3 for x in values]\n", [{"arguments": [[2, -1]], "expected": [6, -3]}])
    assert normal["passed"]
    assert not _execute("def _solve(:\n", [])["passed"]
    timed = _execute("def _solve():\n    while True:\n        pass\n", [{"arguments": [], "expected": None}], 0.8)
    assert not timed["passed"] and "TimeoutExpired" in timed["error"]
    blocked = _execute("import socket\ndef _solve():\n    return socket.create_connection(('127.0.0.1', 1234), 0.1)\n", [])
    assert not blocked["passed"]
    assert not _execute("def _solve():\n    return None\n", [{"arguments": [], "expected": 1}])["passed"]
    model = Chronos(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    record = {"prompt": "Return one.", "cases": [{"arguments": [], "expected": 1}], "messages": [("user", "Return one.")]}
    before = model._embedding.weight.detach().clone()
    event = _autorepair(record, model, tokenizer, optimizer, cfg, "def _solve():\n    return None\n", fixed="def _solve():\n    return 1\n", supervised=True)
    assert event["training"] and event["training"]["updated"]
    assert not torch.equal(before, model._embedding.weight)
    from trainer import _save, _load
    probes = {name: value + ".probe" for name, value in paths.items()}
    _save(model, optimizer, tokenizer, 1, [{"loss": 1.0, "updated": True}], cfg, probes, 1, 0, 1.0)
    restored, restoredoptimizer, restoredtokens, restoredcfg, restoredstep = _load(probes)
    assert restoredstep == 1 and restoredcfg["name"] == "Chronos Alpha" and restoredtokens._tokens == tokenizer._tokens
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    for path in probes.values():
        if os.path.isfile(path):
            os.remove(path)
    print(json.dumps({"tests": "passed", "cacheerror": error, "parameters": full._paramcount, "network": blocked["error"], "timeout": timed["error"]}), flush=True)
    return True


def _stepcheck(target=None):
    """Prove the persisted checkpoint reports a real step count inside the mandated curriculum."""
    from trainer import _load
    paths = _paths()
    model, optimizer, tokenizer, cfg, step = _load(paths)
    want = int(target if target is not None else cfg["stagetwo"])
    state = torch.load(paths["model"], map_location="cpu", weights_only=True)
    valid = step >= want and 10000000 <= model._paramcount <= 15000000 and state["name"] == "Chronos Alpha" and int(state["step"]) == step
    print(json.dumps({"step": step, "target": want, "parameters": model._paramcount, "valid": valid}), flush=True)
    if not valid:
        raise RuntimeError("checkpoint does not prove the required real optimizer step count")
    return True


_MARKERS = ("python", "def ", "import ", "class ", "asyncio", "requests", "httpx", "urllib3", "aiohttp", "json", "csv", "regex", "re.", "traceback", "decorator", "generator", "gil", "interpreter", "list", "dict", "tuple", "set ", "lambda", "exception", "module", "function", "variable", "loop", "recursion", "syntax", "pandas", "numpy", "iterat", "async", "await", "memory", "reference", "context manager", "sort", "string", "integer", "float", "file", "virtualenv", "pip", "venv", "typing", "annotation", "yield", "comprehension")


def _domain(text):
    """Conservative gate: only explicit Python context passes; everything else is refused."""
    lowered = " " + text.lower() + " "
    return any(marker in lowered for marker in _MARKERS)


def _chat(maxinput=None, limit=None):
    """Interactive multi-turn Python-only conversation with guardrail enforcement and long-context ingestion."""
    from tokenizer import Tokenizer
    from trainer import _load
    from dataset import _SYSTEM, _REFUSAL
    cfg = _settings()
    model, optimizer, tokenizer, saved, step = _load(_paths())
    cfg = saved
    model.eval()
    room = int(maxinput if maxinput is not None else cfg["maxinput"])
    cap = int(limit if limit is not None else cfg["reply"])
    history = []
    print("Chronos Alpha chat. Checkpoint step " + str(step) + ". Python questions only. Commands: /file <path>, /clear, /exit, /status", flush=True)
    pending = None
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        if line == "/clear":
            history, pending = [], None
            print("context cleared", flush=True)
            continue
        if line == "/status":
            print(json.dumps({"step": step, "turns": len(history), "maxinput": room, "maxcontext": int(cfg["maxcontext"]), "window": int(cfg["window"]), "sinks": int(cfg["sinks"]), "parameters": model._paramcount}), flush=True)
            continue
        if line.startswith("/file"):
            path = line[5:].strip().strip('"')
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as stream:
                    pending = stream.read(room * 6)
                print("ingested " + str(len(pending)) + " characters from " + path + "; ask a question to attach it", flush=True)
            except OSError as error:
                print("cannot read file: " + str(error), flush=True)
            continue
        question = line if pending is None else ("Here is a Python file for context:" + chr(10) + pending + chr(10) + line)
        pending = None
        if not _domain(question):
            print("Chronos Alpha> " + _REFUSAL, flush=True)
            history.append(("user", question))
            history.append(("assistant", _REFUSAL))
            continue
        ids = tokenizer._chat(history + [("user", question)], _SYSTEM)
        if len(ids) > room:
            trimmed = [("user", question)]
            ids = tokenizer._chat(trimmed, _SYSTEM)
            while len(ids) > room and len(question) > 200:
                question = question[len(question) // 4:]
                ids = tokenizer._chat([("user", question)], _SYSTEM)
        started = time.perf_counter()
        if len(ids) > int(cfg["chunk"]):
            prefill = model._prefill(ids, chunk=int(cfg["chunk"]), tail=0)
            logits, cache = prefill["logits"], prefill["cache"]
        else:
            logits, cache = model(torch.tensor([ids], dtype=torch.long), None)
        generated = []
        with torch.inference_mode():
            for unused in range(cap):
                scores = logits[0, -1].float().clone()
                for banned in (tokenizer._pad(), tokenizer._bos(), tokenizer._user(), tokenizer._system(), tokenizer._code(), tokenizer._endcode(), tokenizer._unk()):
                    scores[banned] = float("-inf")
                selected = int(scores.argmax())
                if selected == tokenizer._eos():
                    break
                generated.append(selected)
                logits, cache = model(torch.tensor([[selected]], dtype=torch.long), cache)
        reply = tokenizer._decode(generated).strip()
        elapsed = time.perf_counter() - started
        print("Chronos Alpha> " + (reply if reply else "(no output)"), flush=True)
        print("  [" + str(len(generated)) + " tokens in " + format(elapsed, ".2f") + "s]", flush=True)
        history.append(("user", question))
        history.append(("assistant", reply))
        if len(tokenizer._chat(history, _SYSTEM)) > room:
            history = history[-4:]


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if action in ("--chat", "chat"):
        parser = argparse.ArgumentParser(prog="main.py chat")
        parser.add_argument("--chat", action="store_true")
        parser.add_argument("--maxinput", type=int, default=None)
        parser.add_argument("--limit", type=int, default=None)
        known, unused = parser.parse_known_args(sys.argv[2:])
        _chat(known.maxinput, known.limit)
    elif action == "verify":
        if _verify():
            raise SystemExit(1)
    elif action == "check":
        _check()
    elif action == "dataset":
        from dataset import _generate, _verify as _vdata, _storedata
        from tokenizer import Tokenizer
        cfg = _settings()
        train, hold = _generate(None, 0)
        tokenizer = Tokenizer()._fit([record.get("prompt", "") + record.get("code", "") for record in train], int(cfg["vocab"]))
        train, hold = _generate(tokenizer, int(cfg["retrievaltokens"]))
        _vdata(train + hold)
        print(json.dumps(_storedata(train, hold)), flush=True)
    elif action == "train":
        from trainer import _train, _correction
        result = _train(None, resume=True, verify=False)
        print(json.dumps(_correction(result["model"], result["optimizer"], result["tokenizer"], result["cfg"], result["data"], _paths())), flush=True)
    elif action == "benchmark":
        from benchmark import _bench
        _bench()
    elif action == "stepcheck":
        _stepcheck()
    else:
        raise SystemExit(_USAGE)
