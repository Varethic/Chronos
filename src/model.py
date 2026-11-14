"""Chronos Alpha: GQA decoder with sliding-window attention, attention sinks, chunked KV prefill and NTK/YaRN RoPE."""
import ast, math, time
import torch, torch.nn as nn, torch.nn.functional as F


def _base(theta, dim, scaled, trained):
    """NTK-aware base scaling: raise theta so every frequency stretches by the requested factor."""
    if scaled is None or trained is None or scaled <= trained or scaled <= 1.0:
        return theta
    return theta * (scaled / trained) ** (dim / max(2, dim - 2))


def _frequencies(dim, theta, positions, scaled=None, trained=None):
    """Rotary angles (n, dim) at the given integer positions, expanded to interleaved pairs."""
    base = _base(theta, dim, scaled, trained)
    half = dim // 2
    inverse = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    angles = torch.outer(positions.float(), inverse)
    return angles.repeat_interleave(2, dim=-1)


def _rotate(values, angles):
    """Rotate interleaved (even, odd) pairs of a [batch, heads, length, width] tensor."""
    cosine, sine = torch.cos(angles[..., ::2]), torch.sin(angles[..., ::2])
    while cosine.ndim < values.ndim:
        cosine, sine = cosine.unsqueeze(0), sine.unsqueeze(0)
    even, odd = values[..., ::2], values[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Grouped-query causal attention over a bounded sliding window plus a permanent sink span."""

    def __init__(self, cfg):
        super().__init__()
        self._heads = int(cfg["heads"])
        self._kvheads = int(cfg["kvheads"])
        self._dim = int(cfg["dim"])
        self._width = self._dim // self._heads
        if self._dim % self._heads or self._heads % self._kvheads or self._width % 2:
            raise ValueError("invalid grouped-query head geometry")
        self._window = int(cfg["window"])
        self._sinks = int(cfg["sinks"])
        self._query = nn.Linear(self._dim, self._heads * self._width, bias=False)
        self._key = nn.Linear(self._dim, self._kvheads * self._width, bias=False)
        self._value = nn.Linear(self._dim, self._kvheads * self._width, bias=False)
        self._project = nn.Linear(self._heads * self._width, self._dim, bias=False)

    def _split(self, tensor, heads, batch, length):
        return tensor.view(batch, length, heads, self._width).transpose(1, 2)

    def _expand(self, tensor):
        group = self._heads // self._kvheads
        return tensor if group == 1 else tensor.repeat_interleave(group, dim=1)

    def _attend(self, values, cache, cfg):
        """Attend over sinks plus the causal window; returns context and the trimmed rolling cache."""
        batch, length, _ = values.shape
        offset = 0 if cache is None else int(cache["length"])
        query = self._split(self._query(values), self._heads, batch, length)
        key = self._split(self._key(values), self._kvheads, batch, length)
        value = self._split(self._value(values), self._kvheads, batch, length)
        angles = _frequencies(self._width, float(cfg["theta"]), torch.arange(offset, offset + length), cfg["maxinput"], cfg["ropebase"]).unsqueeze(0)
        query, key = _rotate(query, angles), _rotate(key, angles)
        if cache is None:
            spansink = min(self._sinks, length)
            sinkk, sinkv = key[:, :, :spansink], value[:, :, :spansink]
            recentk, recentv, first = key[:, :, spansink:], value[:, :, spansink:], spansink
        else:
            sinkk, sinkv = cache["sinks"]
            recentk = torch.cat((cache["recent"][0], key), dim=2)
            recentv = torch.cat((cache["recent"][1], value), dim=2)
            first = int(cache["first"])
        keys, stacks = self._expand(recentk), self._expand(recentv)
        broad = self._expand(sinkk)
        broadv = self._expand(sinkv)
        span = broad.shape[2]
        block = max(1, min(int(cfg["attnblock"]), length))
        scale = 1.0 / math.sqrt(self._width)
        context = torch.empty(batch, self._heads, length, self._width)
        for begin in range(0, length, block):
            stop = min(begin + block, length)
            positions = torch.arange(offset + begin, offset + stop)
            low = max(0, offset + begin - self._window + 1 - first)
            high = offset + stop - first
            board = torch.arange(first + low, first + high)
            allowed = (board.unsqueeze(0) <= positions.unsqueeze(1)) & (board.unsqueeze(0) >= (positions.unsqueeze(1) - self._window + 1))
            scores = torch.matmul(query[:, :, begin:stop], keys[:, :, low:high].transpose(-1, -2)) * scale
            scores = scores.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), float("-inf"))
            part = stacks[:, :, low:high]
            if span:
                nearby = torch.matmul(query[:, :, begin:stop], broad.transpose(-1, -2)) * scale
                keep = torch.arange(span).unsqueeze(0) <= positions.unsqueeze(1)
                nearby = nearby.masked_fill(~keep.unsqueeze(0).unsqueeze(0), float("-inf"))
                scores = torch.cat((nearby, scores), dim=-1)
                part = torch.cat((broadv, part), dim=2)
            context[:, :, begin:stop] = torch.matmul(torch.softmax(scores, dim=-1), part)
        total = offset + length
        drop = max(0, recentk.shape[2] - self._window)
        state = {"sinks": (sinkk[:, :, :span], sinkv[:, :, :span]), "recent": (recentk[:, :, drop:], recentv[:, :, drop:]), "first": first + drop, "length": total, "span": span}
        context = context.transpose(1, 2).contiguous().view(batch, length, self._dim)
        return self._project(context), state

    def _report(self, state):
        return {"window": int(state["recent"][0].shape[2]), "sink": int(state["span"]), "cache": int(state["recent"][0].shape[2] + state["span"])}


class Block(nn.Module):
    """Pre-norm transformer block: grouped attention sublayer then feed-forward sublayer."""

    def __init__(self, cfg):
        super().__init__()
        self._normone, self._normtwo = nn.LayerNorm(cfg["dim"]), nn.LayerNorm(cfg["dim"])
        self._attention = Attention(cfg)
        self._feed = nn.Sequential(nn.Linear(cfg["dim"], cfg["ff"]), nn.GELU(), nn.Linear(cfg["ff"], cfg["dim"]))
        self._drop = nn.Dropout(float(cfg["dropout"]))

    def _forward(self, values, cache, cfg):
        attended, state = self._attention._attend(self._normone(values), cache, cfg)
        values = values + self._drop(attended)
        return values + self._drop(self._feed(self._normtwo(values))), state

    forward = _forward


class Chronos(nn.Module):
    """Memory-bounded causal decoder: rotary positions, grouped-query sliding attention, tied embeddings."""

    def __init__(self, cfg):
        super().__init__()
        if cfg["name"] != "Chronos Alpha" or cfg["dim"] % cfg["heads"]:
            raise ValueError("invalid Chronos Alpha topology")
        self._cfg = dict(cfg)
        self._embedding = nn.Embedding(cfg["vocab"], cfg["dim"], padding_idx=0)
        self._blocks = nn.ModuleList([Block(cfg) for unused in range(cfg["layers"])])
        self._norm = nn.LayerNorm(cfg["dim"])
        self._output = nn.Linear(cfg["dim"], cfg["vocab"], bias=False)
        self._output.weight = self._embedding.weight
        self._reset()
        self._total = sum(parameter.numel() for parameter in self.parameters())
        self._paramcount = self._total

    def _reset(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _forward(self, tokens, cache=None, collect=False):
        """Score a span; `cache` carries per-layer keys, values and absolute positions from earlier spans."""
        if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.device.type != "cpu":
            raise ValueError("a two-dimensional CPU long tensor is required")
        batch, length = tokens.shape
        offset = 0 if cache is None else int(cache["blocks"][0]["length"])
        if length < 1:
            raise ValueError("a span must hold at least one token")
        if offset + length > int(self._cfg["maxcontext"]):
            raise ValueError("requested span exceeds max_context_tokens")
        values = self._embedding(tokens)
        updated, reports = [], []
        for index, block in enumerate(self._blocks):
            values, state = block(values, None if cache is None else cache["blocks"][index], self._cfg)
            updated.append(state)
            reports.append(block._attention._report(state))
        logits = self._output(self._norm(values))
        out = {"blocks": updated, "length": offset + length}
        return (logits, out, reports) if collect else (logits, out)

    forward = _forward

    def _loss(self, tokens, mask, targets):
        """Masked cross entropy in the configured compute precision (float32 on this hardware)."""
        logits, unused = self(tokens)
        weights = mask.reshape(-1).float()
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), reduction="none")
        return (losses * weights).sum() / weights.sum().clamp_min(1.0)

    def _prefill(self, ids, chunk=None, tail=0):
        """Chunked prefill in fixed blocks over the rolling sink+window cache, so attention memory stays O(window)."""
        size = int(chunk if chunk is not None else self._cfg["chunk"])
        cache, reports, kept = None, [], max(0, len(ids) - int(tail))
        logits, cells = None, 0
        with torch.inference_mode():
            for begin in range(0, max(1, kept), size):
                span = ids[begin:min(kept, begin + size)]
                if not span:
                    break
                logits, cache, report = self._forward(torch.tensor([span], dtype=torch.long), cache, True)
                cells = max(cells, len(span) * int(report[0]["cache"]))
                reports.append({"begin": begin, "length": len(span), **report[0]})
        return {"cache": cache, "logits": logits, "reports": reports, "consumed": kept, "peakcells": cells}

    @torch.inference_mode()
    def _generate(self, tokenizer, prompt, limit=None, temperature=0.0, maxinput=None):
        """Greedy or sampled decode of one assistant reply after a chat-framed prompt."""
        was = self.training
        self.eval()
        try:
            cap = int(limit if limit is not None else self._cfg["generate"])
            room = int(maxinput if maxinput is not None else self._cfg["maxinput"])
            ids = tokenizer._encode(prompt, controls=False) if isinstance(prompt, str) else [int(value) for value in prompt]
            if len(ids) > room:
                ids = ids[-room:]
            if not ids:
                ids = [tokenizer._bos()]
            started = time.perf_counter()
            if len(ids) > int(self._cfg["chunk"]):
                prefill = self._prefill(ids, tail=1)
                logits, cache = prefill["logits"], prefill["cache"]
            else:
                logits, cache = self(torch.tensor([ids], dtype=torch.long), None)
            first, generated = None, []
            for unused in range(cap):
                scores = logits[0, -1].float().clone()
                for banned in (tokenizer._pad(), tokenizer._bos(), tokenizer._unk(), tokenizer._user(), tokenizer._system(), tokenizer._code(), tokenizer._endcode()):
                    scores[banned] = float("-inf")
                if temperature <= 0.01:
                    selected = int(scores.argmax())
                else:
                    selected = int(torch.multinomial(torch.softmax(scores / temperature, dim=-1), 1))
                if first is None:
                    first = time.perf_counter() - started
                if selected == tokenizer._eos():
                    break
                generated.append(selected)
                logits, cache = self(torch.tensor([[selected]], dtype=torch.long), cache)
            elapsed = time.perf_counter() - started
            text = tokenizer._decode(generated)
            return {"code": _code(text), "reply": text, "tokens": generated, "latency": elapsed, "first": first if first is not None else elapsed, "rate": len(generated) / max(elapsed, 1e-9), "parameters": self._total, "mode": "greedy" if temperature <= 0.01 else "sampled"}
        finally:
            self.train(was)


def _code(text):
    """Extract the longest syntax-valid Python prefix the model actually emitted."""
    for marker in ("<|endcode|>", "<|eos|>", "<|user|>", "<|unk|>", "<|assistant|>", "<|system|>", "<|code|>"):
        text = text.split(marker, 1)[0]
    fence = text.find("```")
    if fence != -1:
        text = text[fence + 3:]
        if text[:6].lower() == "python":
            text = text[6:]
        end = text.find("```")
        if end != -1:
            text = text[:end]
    lines = [line for line in text.splitlines() if line.strip().lower() not in ("python", "py")]
    for cut in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:cut]).rstrip() + "\n"
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        return candidate
    return text.strip() + ("\n" if text.strip() else "")
