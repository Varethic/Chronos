"""Deterministic byte-level BPE plus chat control tokens: lossless for arbitrary Unicode, no out-of-vocabulary bytes."""
import json, re, heapq
from config import _atomicjson

_CONTROLS = ["<pad>", "<bos>", "<eos>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>", "<|code|>", "<|endcode|>"]
_BASE = len(_CONTROLS)
_SPLIT = re.compile("(<\\|[a-z]+\\|>|<pad>|<bos>|<eos>|<unk>)")


class Tokenizer:
    """Byte-pair encoder with an explicit merge table and reserved chat control tokens."""

    def __init__(self):
        self._tokens = list(_CONTROLS) + [chr(index) for index in range(256)]
        self._merges = []
        self._ranks = {}
        self._letters = {}
        self._control = {name: index for index, name in enumerate(_CONTROLS)}

    def _piece(self, text):
        """Split text into control-token identifiers and raw byte runs so specials are never merged away."""
        output = []
        for part in _SPLIT.split(text):
            if not part:
                continue
            if part in self._control:
                output.append(self._control[part])
            else:
                output.extend(value + _BASE for value in part.encode("utf-8"))
        return output

    def _fit(self, texts, size=4096):
        """Greedy BPE over byte sequences; deterministic given the corpus and vocabulary size."""
        if size < _BASE + 256 or size > _BASE + 4088:
            raise ValueError("vocabulary size must lie between " + str(_BASE + 256) + " and " + str(_BASE + 4088))
        words = [self._piece(text) for text in texts if isinstance(text, str) and text]
        counts, holders = {}, {}
        for position, word in enumerate(words):
            tallies = {}
            for pair in zip(word, word[1:]):
                tallies[pair] = tallies.get(pair, 0) + 1
            for pair, amount in tallies.items():
                counts[pair] = counts.get(pair, 0) + amount
                holders.setdefault(pair, set()).add(position)
        while len(self._tokens) < size:
            if not counts:
                break
            pair = max(counts, key=lambda item: (counts[item], -item[0], -item[1]))
            if counts[pair] < 2:
                break
            identifier = len(self._tokens)
            self._tokens.append(self._tokens[pair[0]] + self._tokens[pair[1]])
            self._merges.append(pair)
            self._ranks[pair] = identifier
            for position in sorted(holders.pop(pair, ())):
                word = words[position]
                tallies = {}
                for old in zip(word, word[1:]):
                    tallies[old] = tallies.get(old, 0) + 1
                for old, amount in tallies.items():
                    counts[old] = counts.get(old, 0) - amount
                    if counts[old] <= 0:
                        counts.pop(old, None)
                        shared = holders.get(old)
                        if shared is not None:
                            shared.discard(position)
                            if not shared:
                                holders.pop(old, None)
                merged, index = [], 0
                while index < len(word):
                    if index + 1 < len(word) and (word[index], word[index + 1]) == pair:
                        merged.append(identifier)
                        index += 2
                    else:
                        merged.append(word[index])
                        index += 1
                words[position] = merged
                added = {}
                for new in zip(merged, merged[1:]):
                    added[new] = added.get(new, 0) + 1
                for new, amount in added.items():
                    counts[new] = counts.get(new, 0) + amount
                    holders.setdefault(new, set()).add(position)
        return self

    def _encode(self, text, controls=True):
        """Merge byte symbols in learned order using a linked list and a rank heap."""
        if not isinstance(text, str):
            raise TypeError("text must be str")
        word = self._piece(text)
        reserved = [index for index, value in enumerate(word) if value < _BASE]
        if not word:
            return ([1] + [2]) if controls else []
        length = len(word)
        previous = [index - 1 for index in range(length)]
        following = [index + 1 for index in range(length)]
        following[length - 1] = -1
        alive = [True] * length
        heap = []
        for index in range(length - 1):
            rank = self._ranks.get((word[index], word[index + 1]))
            if rank is not None:
                heap.append((rank, index))
        heapq.heapify(heap)
        while heap:
            rank, index = heapq.heappop(heap)
            forward = following[index]
            if not alive[index] or forward == -1 or not alive[forward]:
                continue
            if self._ranks.get((word[index], word[forward])) != rank:
                continue
            word[index] = rank
            alive[forward] = False
            after = following[forward]
            following[index] = after
            if after != -1:
                previous[after] = index
            for position in (previous[index], index):
                if position == -1:
                    continue
                neighbour = following[position]
                if neighbour == -1:
                    continue
                sibling = self._ranks.get((word[position], word[neighbour]))
                if sibling is not None:
                    heapq.heappush(heap, (sibling, position))
        merged = []
        cursor = 0
        while cursor != -1:
            merged.append(word[cursor])
            cursor = following[cursor]
        return ([1] + merged + [2]) if controls else merged

    def _expand(self, identifier):
        """Recursively rebuild the byte string of a merged token."""
        if identifier in self._letters:
            return self._letters[identifier]
        left, right = self._merges[identifier - (_BASE + 256)]
        self._letters[identifier] = self._bytes(left) + self._bytes(right)
        return self._letters[identifier]

    def _bytes(self, identifier):
        return bytes([identifier - _BASE]) if identifier < _BASE + 256 else self._expand(identifier)

    def _decode(self, ids, controls=False):
        """Concatenate byte expansions; control tokens are omitted unless requested."""
        pieces = []
        for value in ids:
            identifier = int(value)
            if identifier < 0 or identifier >= len(self._tokens):
                continue
            if identifier < _BASE:
                if controls:
                    pieces.append(self._tokens[identifier].encode("utf-8"))
            else:
                pieces.append(self._bytes(identifier))
        return b"".join(pieces).decode("utf-8", errors="replace")

    def _system(self):
        return self._control["<|system|>"]

    def _user(self):
        return self._control["<|user|>"]

    def _assistant(self):
        return self._control["<|assistant|>"]

    def _code(self):
        return self._control["<|code|>"]

    def _endcode(self):
        return self._control["<|endcode|>"]

    def _eos(self):
        return self._control["<eos>"]

    def _bos(self):
        return self._control["<bos>"]

    def _unk(self):
        """Identifier of the reserved unknown byte token."""
        return 3

    def _pad(self):
        return self._control["<pad>"]

    def _chat(self, messages, system):
        """Serialize a dialogue as chat control tokens; returns ids ready for next-token prediction."""
        ids = [self._bos(), self._system()]
        ids += self._encode(system, controls=False) + [self._eos(), self._user()]
        for index, (role, content) in enumerate(messages):
            if role == "user":
                ids += self._encode(content, controls=False) + [self._eos(), self._assistant()]
            else:
                # Only Python source is fenced with the code control tokens. A refusal
                # or a prose explanation is plain text; fencing every answer made
                # <|code|> mean "some answer starts here", so the model could not tell
                # a code reply from a refusal and emitted one when asked for the other.
                source = isinstance(content, str) and content.lstrip().startswith(("def ", "import ", "async def ", "@", "from "))
                if source:
                    ids += [self._code()] + self._encode(content, controls=False) + [self._endcode(), self._eos()]
                else:
                    ids += self._encode(content, controls=False) + [self._eos()]
                if index != len(messages) - 1:
                    ids += [self._user()]
        return ids

    def _save(self, path):
        _atomicjson({"name": "Chronos Alpha", "kind": "bytelevelbpe", "version": 2, "controls": list(_CONTROLS), "tokens": self._tokens, "merges": [[left, right, new] for (left, right), new in self._ranks.items()]}, path)

    def _load(self, path):
        with open(path, "r", encoding="utf-8") as stream:
            state = json.load(stream)
        if not isinstance(state, dict) or state.get("name") != "Chronos Alpha" or state.get("kind") != "bytelevelbpe":
            raise ValueError("not a Chronos Alpha vocabulary")
        tokens, merges = state.get("tokens"), state.get("merges")
        if not isinstance(tokens, list) or not isinstance(merges, list) or len(tokens) < _BASE + 256 or tokens[:_BASE] != _CONTROLS:
            raise ValueError("invalid vocabulary payload")
        self._tokens = tokens
        self._merges = []
        self._ranks = {}
        self._letters = {}
        for left, right, new in sorted(merges, key=lambda item: int(item[2])):
            self._merges.append((int(left), int(right)))
            self._ranks[(int(left), int(right))] = int(new)
        if sorted(self._ranks.values()) != list(range(_BASE + 256, _BASE + 256 + len(self._merges))):
            raise ValueError("merge identifiers must be contiguous and ordered")
        return self

    def __len__(self):
        return len(self._tokens)
