"""Isolated execution: temp-directory cwd, strict timeouts, offline HTTP transport mocking."""
import os, sys, ast, json, time, socket, subprocess, tempfile, traceback, types, inspect, asyncio, urllib.parse

_MARKER = "__chronos_result__"
_ALLOWED = {"json", "time", "math", "asyncio", "itertools", "functools", "collections", "statistics", "re", "csv", "io", "requests", "httpx", "urllib3", "aiohttp"}
_DENIED = {"eval", "exec", "compile", "open", "input", "__import__", "globals", "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit", "breakpoint"}
_DENIEDATTRS = {"system", "popen", "remove", "unlink", "rmdir", "rmtree", "chdir", "stat", "walk", "listdir"}
_API = {"base": 42, "items": [{"value": 2}, {"value": 3}], "next": None, "values": [42, 43]}
_LOOPBACK = ("127.0.0.1", "::1", "localhost", "0.0.0.0", "")


def _cause(error):
    return type(error).__name__ + ": " + str(error)


def _guard(source):
    """Parse the candidate and reject workspace-touching operations (defense in depth)."""
    if not isinstance(source, str) or len(source) > 200000:
        raise ValueError("candidate source must be bounded Python text")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED:
                    raise ValueError("forbidden import on line " + str(node.lineno) + ": " + alias.name)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in _ALLOWED:
                raise ValueError("forbidden import on line " + str(node.lineno) + ": " + str(node.module))
        elif isinstance(node, ast.Name) and node.id in _DENIED:
            raise ValueError("forbidden operation on line " + str(node.lineno) + ": " + node.id)
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in _DENIEDATTRS):
            raise ValueError("forbidden attribute on line " + str(node.lineno) + ": " + node.attr)
    return tree


def _body(value):
    """Normalize any client-library body into a plain dictionary."""
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        if text[:1] in "[{":
            try:
                decoded = json.loads(text)
            except ValueError:
                return {}
            return decoded if isinstance(decoded, dict) else {}
        try:
            return {key: values[-1] for key, values in urllib.parse.parse_qs(text).items()}
        except ValueError:
            return {}
    return dict(value) if isinstance(value, dict) else {}


def _fixture(method, url, headers=None, params=None, body=None):
    """Deterministic in-memory REST fixture: no socket, no DNS, no live network."""
    text = str(url).split("?", 1)[0].rstrip("/")
    tail = text.rsplit("/", 1)[-1]
    identity = int(tail) if tail.isdigit() else 1
    options = dict(params or {})
    fields = _body(body)
    lookup = {str(name).lower(): value for name, value in dict(headers or {}).items()}

    def __number(name):
        try:
            return int(options[name])
        except (KeyError, TypeError, ValueError):
            return identity

    amount = fields.get("amount", identity)
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        amount = identity
    return {"base": _API["base"], "items": [dict(item) for item in _API["items"]], "next": _API["next"], "method": str(method).upper(), "token": lookup.get("authorization"), "page": __number("page"), "offset": __number("offset"), "amount": amount}


def _stream(text):
    return "stream" in str(text)


class Reply:
    """requests-compatible response object."""

    def __init__(self, request=None, **unused):
        self._request = request
        method = getattr(request, "method", "GET")
        url = getattr(request, "url", "http://offline.test/api/1")
        headers = dict(getattr(request, "headers", {}) or {})
        params = dict(getattr(getattr(request, "url", None), "params", {}) or {})
        payload = _fixture(method, url, headers, params, getattr(request, "body", None))
        self.status_code, self.reason, self.encoding = 200, "OK", "utf-8"
        self.headers = {"Content-Type": "application/json"}
        self.url = str(url)
        self.request = request
        if _stream(url):
            self._content = b"\n".join(json.dumps({"value": value}).encode("utf-8") for value in _API["values"])
        else:
            self._content = json.dumps(payload).encode("utf-8")

    @property
    def _body(self):
        return self._content

    @property
    def _text(self):
        return self._content.decode("utf-8", errors="replace")

    @property
    def _ok(self):
        return True

    def _json(self, **unused):
        return json.loads(self._content.decode("utf-8"))

    def _raise(self):
        return None

    def _lines(self, **unused):
        return iter(self._content.splitlines())

    content, text, ok = property(_body.fget), property(_text.fget), property(_ok.fget)
    json, raise_for_status, iter_lines = _json, _raise, _lines


class Stream:
    """Async line reader matching aiohttp's StreamReader content interface."""

    def __init__(self, lines):
        self._lines = list(lines)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        return line


def _send(self, request, **unused):
    return Reply(request)


async def _asend(self, request, **unused):
    return Reply(request)


class _PoolReply:
    """urllib3-compatible response object."""

    status, reason = 200, "OK"

    def __init__(self, payload):
        self._payload = payload
        self._content = json.dumps(payload).encode("utf-8")
        self.headers = {"Content-Type": "application/json"}

    def _data(self):
        return self._content

    def _json(self, **unused):
        return self._payload

    def _read(self, *unused):
        return self._content

    data, json, read = property(_data), _json, _read


def _urlopen(self, method, url, *unused, **options):
    return _PoolReply(_fixture(method, str(url), dict(options.get("headers") or {}), {}, options.get("body")))


async def _aiorequest(self, method, url, **options):
    headers = {**dict(getattr(self, "headers", {}) or {}), **dict(options.get("headers") or {})}
    return AioReply(method, str(url), options.get("params"), options.get("json", options.get("data")), headers)


class AioReply:
    """aiohttp-compatible response: async context manager, async json, async line content."""

    def __init__(self, method, url, params=None, body=None, headers=None):
        self._payload = _fixture(method, url, headers, params, body)
        self._lines = [json.dumps({"value": value}).encode("utf-8") + b"\n" for value in _API["values"]]
        self._streamed = _stream(url)
        self.status, self.reason, self.method, self.url = 200, "OK", str(method).upper(), str(url)
        self.headers = {"Content-Type": "application/json"}
        self.content = Stream(self._lines) if self._streamed else b""

    async def _json(self, **unused):
        return self._payload

    async def _text(self, **unused):
        return json.dumps(self._payload)

    async def _read(self):
        return json.dumps(self._payload).encode("utf-8")

    def _raise(self):
        return None

    def _release(self):
        return None

    def _close(self):
        return None

    json, text, read, raise_for_status, release, close = _json, _text, _read, _raise, _release, _close

    async def __aenter__(self):
        return self

    async def __aexit__(self, *unused):
        return False


def _refuse(*unused, **ignored):
    raise OSError("live networking is disabled inside the Chronos sandbox")


def _attach(name):
    """Bind one client library's transport layer to the in-memory fixture."""
    if name == "requests":
        requests = sys.modules[name]
        requests.Session.send = _send
        requests.sessions.Session.send = _send
    elif name == "httpx":
        httpx = sys.modules[name]
        httpx.Client.send = _send
        httpx.AsyncClient.send = _asend
    elif name == "urllib3":
        urllib3 = sys.modules[name]
        urllib3.PoolManager.urlopen = _urlopen
        urllib3.connectionpool.HTTPConnectionPool.urlopen = _urlopen
    elif name == "aiohttp":
        aiohttp = sys.modules[name]
        aiohttp.ClientSession._request = _aiorequest


def _patch():
    """Redirect every client transport to the in-memory fixture, importing only libraries in use."""
    for name in ("requests", "httpx", "urllib3", "aiohttp"):
        if name in sys.modules:
            _attach(name)
    original, resolve = socket.socket, socket.getaddrinfo

    class _Socket(original):
        def _connect(self, address, *unused, **ignored):
            host = address[0] if isinstance(address, (tuple, list)) and address else None
            if host is not None and str(host) not in _LOOPBACK:
                raise OSError("live networking is disabled inside the Chronos sandbox")
            return original.connect(self, address, *unused, **ignored)

        def _connect_ex(self, address, *unused, **ignored):
            host = address[0] if isinstance(address, (tuple, list)) and address else None
            if host is not None and str(host) not in _LOOPBACK:
                raise OSError("live networking is disabled inside the Chronos sandbox")
            return original.connect_ex(self, address, *unused, **ignored)

        connect, connect_ex = _connect, _connect_ex

    def __resolve(host, port, *unused, **ignored):
        if str(host) not in _LOOPBACK:
            raise OSError("live name resolution is disabled inside the Chronos sandbox")
        return resolve(host, port, *unused, **ignored)

    socket.socket = _Socket
    socket.getaddrinfo = __resolve


def _compare(actual, expected):
    """Structural comparison tolerating int/float width and tuple/list containers."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) and actual == expected
    if isinstance(expected, float) or isinstance(actual, float):
        return isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and abs(actual - expected) <= 1e-9
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(_compare(one, two) for one, two in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(_compare(actual[key], expected[key]) for key in expected)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual == expected
    if actual is None or expected is None:
        return actual is None and expected is None
    return type(actual) is type(expected) and actual == expected


def _runner():
    """Worker entry point executed as a fresh interpreter inside the temporary directory."""
    started = time.perf_counter()
    try:
        request = json.loads(sys.stdin.read())
        tree = _guard(request["code"])
        scope = {"__name__": "chronos"}
        exec(compile(tree, "<chronos>", "exec"), scope, scope)
        _patch()
        solver = scope.get("_solve")
        if not callable(solver):
            raise NameError("the candidate does not define a callable _solve")
        results = []
        for index, case in enumerate(request["cases"]):
            entry = scope.get(case.get("entry") or "_solve")
            if not callable(entry):
                raise NameError("the candidate does not define a callable " + str(case.get("entry")))
            actual = entry(*case.get("arguments", []))
            if inspect.isawaitable(actual):
                actual = asyncio.run(actual)
            expected = case.get("expected")
            results.append({"index": index, "passed": bool(_compare(actual, expected)), "actual": repr(actual)[:2000], "expected": repr(expected)[:2000]})
        passed = bool(results) and all(result["passed"] for result in results)
        outcome = {"passed": passed, "error": "" if passed else "WrongAnswer: behavioral comparison failed", "traceback": "" if passed else "\n".join(result["actual"] + " != " + result["expected"] for result in results if not result["passed"]), "results": results}
    except BaseException as error:
        outcome = {"passed": False, "error": _cause(error), "traceback": "".join(traceback.format_exception(error))[-6000:], "results": []}
    outcome["worker"] = time.perf_counter() - started
    sys.stdout.write(_MARKER + json.dumps(outcome))


def _execute(code, cases, timeout=2.0):
    """Run one candidate against its probes in a temporary directory with a hard deadline."""
    try:
        _guard(code)
        if not isinstance(cases, list) or not cases or any(not isinstance(case, dict) or not isinstance(case.get("arguments", []), list) for case in cases):
            raise ValueError("cases must be a non-empty list of argument dictionaries")
    except (SyntaxError, ValueError, TypeError) as error:
        return {"passed": False, "error": _cause(error), "traceback": traceback.format_exc(), "latency": 0.0, "results": []}
    worker, started = os.path.abspath(__file__), time.perf_counter()
    environment = {key: value for key, value in os.environ.items() if key in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "NUMBER_OF_PROCESSORS")}
    environment.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    with tempfile.TemporaryDirectory(prefix="chronos") as directory:
        try:
            finished = subprocess.run([sys.executable, "-I", "-B", worker], input=json.dumps({"code": code, "cases": cases}), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=directory, env=environment, check=False)
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "TimeoutExpired: candidate exceeded " + str(timeout) + " seconds", "traceback": "the worker was terminated by the hard deadline", "latency": time.perf_counter() - started, "results": []}
        except (OSError, TypeError) as error:
            return {"passed": False, "error": _cause(error), "traceback": traceback.format_exc(), "latency": time.perf_counter() - started, "results": []}
    if _MARKER not in finished.stdout:
        return {"passed": False, "error": "WorkerError: " + (finished.stderr or "unstructured worker output")[-2000:], "traceback": finished.stderr[-4000:], "latency": time.perf_counter() - started, "results": []}
    try:
        outcome = json.loads(finished.stdout.rsplit(_MARKER, 1)[1])
    except (ValueError, TypeError) as error:
        return {"passed": False, "error": _cause(error), "traceback": finished.stdout[-2000:], "latency": time.perf_counter() - started, "results": []}
    outcome["latency"] = time.perf_counter() - started
    return outcome


if __name__ == "__main__":
    _runner()
