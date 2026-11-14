"""Bounded in-memory correction trajectories and anti-regression state."""
import hashlib


def _hash(text):
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]


class State:
    """Process-local bounded records with replay detection and accepted rollback points."""

    def __init__(self, maximum=128, context=1600):
        self._maximum, self._context = max(4, int(maximum)), max(200, int(context))
        self._attempts, self._accepted, self._patches, self._failed = [], [], {}, set()

    def _record(self, code, error, traceback="", line=None, node=""):
        failure = _hash(error + "\n" + traceback)
        event = {"code": code, "codehash": _hash(code), "error": error, "errorhash": failure, "line": line, "node": node, "traceback": traceback}
        self._attempts.append(event)
        if len(self._attempts) > self._maximum:
            self._attempts = self._attempts[-self._maximum:]
        return {name: value for name, value in event.items() if name not in ("code", "traceback")}

    def _seen(self, code, error, traceback="", correction=None):
        failure = _hash(error + "\n" + traceback)
        if correction is not None and (_hash(code), failure, _hash(correction)) in self._failed:
            return True
        return any(item["codehash"] == _hash(code) and item["errorhash"] == failure for item in self._attempts)

    def _patch(self, source, failure, correction, passed):
        key = (_hash(source), _hash(failure.get("error", "") + "\n" + failure.get("traceback", "")), _hash(correction))
        if key in self._failed and passed is not True:
            return False
        if key in self._patches and passed is not None:
            self._patches[key]["passed"] = passed
            self._failed.discard(key) if passed else self._failed.add(key)
            return True
        if key in self._patches:
            return False
        self._patches[key] = {"source": key[0], "failure": key[1], "correction": correction, "correctionhash": key[2], "passed": passed}
        if passed is False:
            self._failed.add(key)
        return True
    def _commit(self, code, result=None):
        self._accepted.append({"code": code, "codehash": _hash(code), "result": result})
        if len(self._accepted) > self._maximum:
            self._accepted = self._accepted[-self._maximum:]

    def _rollback(self):
        return self._accepted[-1]["code"] if self._accepted else None

    def _history(self):
        lines = ["<previous>code " + event["codehash"] + " error " + event["error"][:240] for event in self._attempts[-3:]]
        rejected = [event["correctionhash"] for event in self._patches.values() if not event["passed"]][-3:]
        lines.extend("<rejectedpatch>" + digest for digest in rejected)
        return "\n".join(lines)[-self._context:]

    def _items(self):
        return {"attempts": len(self._attempts), "accepted": len(self._accepted), "patches": len(self._patches), "failed": len(self._failed), "history": self._history()}
