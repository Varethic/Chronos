"""Chronos Alpha corpus: 6,000 parameterized training records and 600 unseen holdout records."""
import os, ast, json, csv, io, re, random, string, concurrent.futures
import torch, torch.utils.data
from config import _paths, _atomicjson, _digest
from sandbox import _execute

_DOMAINS = ("loops", "control", "http", "async", "json", "text")
_SYSTEM = "You are Chronos Alpha, a Python engineering specialist. Answer only Python questions."
_REFUSAL = "I am Chronos Alpha, specialized exclusively in Python engineering. I can only assist with Python-related questions, debugging, and code."
_TIMEOUT = 2.0
_API = {"base": 42, "items": [{"value": 2}, {"value": 3}], "next": None, "values": [42, 43]}
_RESERVED = {"and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield", "None", "True", "False", "self", "async", "await", "sum", "max", "min", "len", "sorted", "list", "dict", "int", "str", "float", "set", "map", "filter", "zip", "enumerate", "range", "isinstance", "abs", "print", "open", "input", "type", "bool", "tuple", "round", "reversed", "any", "all", "getattr", "hasattr"}


def _names(seed, count, prefix=""):
    """Deterministic unique Python identifiers, never colliding with builtins or dunder names."""
    rng = random.Random(seed)
    chosen, output = set(), []
    while len(output) < count:
        length = rng.randint(3, 9)
        name = prefix + "".join(rng.choice(string.ascii_lowercase) for unused in range(length))
        if name in chosen or name in _RESERVED or name.startswith("__"):
            continue
        chosen.add(name)
        output.append(name)
    return output


def _url(variant, stream=False):
    return ("http://offline.test/api/stream/" if stream else "http://offline.test/api/") + str(variant)


def _running(values, start):
    result, total = [], start
    for value in values:
        total = total + value
        result.append(total)
    return result


def _ceiling(values, start):
    result, best = [], start
    for value in values:
        best = value if value > best else best
        result.append(best)
    return result


def _position(values, keep, last=False):
    found = -1
    for index, value in enumerate(values):
        if keep(value):
            found = index
            if not last:
                return found
    return found


def _factorial(number, base):
    return base if number <= 1 else number * _factorial(number - 1, base)


def _find(data, key, offset):
    if isinstance(data, dict):
        if key in data:
            return data[key] + offset
        for value in data.values():
            found = _find(value, key, offset)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _find(value, key, offset)
            if found is not None:
                return found
    return None


def _flatten(data, separator):
    result = {}

    def __walk(item, prefix):
        if isinstance(item, dict):
            for name, value in item.items():
                __walk(value, prefix + (separator if prefix else "") + str(name))
        elif isinstance(item, list):
            for index, value in enumerate(item):
                __walk(value, prefix + separator + str(index))
        else:
            result[prefix] = item

    __walk(data, "")
    return result


def _leafsum(data, offset):
    if isinstance(data, dict):
        return sum(_leafsum(value, offset) for value in data.values())
    if isinstance(data, list):
        return sum(_leafsum(value, offset) for value in data)
    if isinstance(data, bool):
        return 0
    return data + offset if isinstance(data, int) else 0


def _numscale(data, factor):
    if isinstance(data, dict):
        return sum(_numscale(value, factor) for value in data.values())
    if isinstance(data, list):
        return sum(_numscale(value, factor) for value in data)
    if isinstance(data, bool):
        return 0
    return data * factor if isinstance(data, (int, float)) else 0


def _valid(data, keys, minimum):
    if not isinstance(data, dict):
        return False
    for name in keys:
        value = data.get(name)
        if not isinstance(value, str) or len(value.strip()) < minimum:
            return False
    return True


def _collect(data, key):
    result = []
    if isinstance(data, dict):
        for name, value in data.items():
            if name == key:
                result.extend(value if isinstance(value, list) else [value])
            else:
                result.extend(_collect(value, key))
    elif isinstance(data, list):
        for value in data:
            result.extend(_collect(value, key))
    return result


def _setpath(data, path, score, offset):
    current = data
    for name in path.split("."):
        if not isinstance(current.get(name), dict):
            current[name] = {}
        current = current[name]
    current["score"] = score + offset
    return data


def _summary(records, offset, key="group"):
    result = {}
    for record in records:
        group = record.get(key, "none")
        result[group] = result.get(group, 0) + record["value"] + offset
    return result


def _rewrite(text, variant):
    rows = list(csv.reader(io.StringIO(text)))
    output = io.StringIO(newline="")
    csv.writer(output, delimiter=";" if variant % 2 == 0 else "\t").writerows(rows)
    return output.getvalue()


def _loops():
    """Loop and iteration families: 0-4 train on this pattern, 5-9 are unseen variants."""
    return [
        {"code": "def _solve(values):\n    return [value * $K$ for value in values]\n",
         "spec": "multiply every value of a list by $K$ and return the resulting list",
         "inputs": lambda k: [[[1, -2, 3]], [[0, 5]], [[7]]],
         "oracle": lambda a, k: list(map(lambda item: item * k, a[0]))},
        {"code": "def _solve(n):\n    return sum(value for value in range(1, n + 1) if value % $K$ == 0)\n",
         "spec": "sum every integer from 1 through n that is exactly divisible by $K$",
         "inputs": lambda k: [[12], [25], [0]],
         "oracle": lambda a, k: sum(filter(lambda item: item % k == 0, range(1, a[0] + 1)))},
        {"code": "def _solve(values):\n    total = $K$\n    out = []\n    for value in values:\n        total += value\n        out.append(total)\n    return out\n",
         "spec": "start a running total at $K$, add each list value in order, and return the running totals",
         "inputs": lambda k: [[[1, 2, 3]], [[]], [[-4, 10]]],
         "oracle": lambda a, k: _running(a[0], k)},
        {"code": "def _solve(values):\n    for index, value in enumerate(values):\n        if value > $K$:\n            return index\n    return -1\n",
         "spec": "return the index of the first value strictly greater than $K$, or -1 when there is none",
         "inputs": lambda k: [[[1, 2, k + 20]], [[k, k]], [[]]],
         "oracle": lambda a, k: _position(a[0], lambda item: item > k)},
        {"code": "def _solve(values):\n    if not values:\n        return $K$\n    return max(value * 2 for value in values)\n",
         "spec": "return twice the largest value of a list, or $K$ when the list is empty",
         "inputs": lambda k: [[[3, 9, 4]], [[]]],
         "oracle": lambda a, k: max(v * 2 for v in a[0]) if a[0] else k},
        {"code": "def _solve(values):\n    return sum(value * value for value in values if value % $K$ == 0)\n",
         "spec": "sum the squares of every list value that is exactly divisible by $K$",
         "inputs": lambda k: [[[2, 3, 4, 9]], [[]], [list(range(-6, 13))]],
         "oracle": lambda a, k: sum(map(lambda v: v * v, filter(lambda v: v % k == 0, a[0])))},
        {"code": "def _solve(values):\n    best = $K$\n    out = []\n    for value in values:\n        if value > best:\n            best = value\n        out.append(best)\n    return out\n",
         "spec": "start a running maximum at $K$ and return the running maxima across the list",
         "inputs": lambda k: [[[1, 9, 2]], [[]], [[-5, k + 3]]],
         "oracle": lambda a, k: _ceiling(a[0], k)},
        {"code": "def _solve(values):\n    return sorted((value * $K$ for value in values), reverse=True)\n",
         "spec": "multiply every value by $K$ and return the products in descending order",
         "inputs": lambda k: [[[3, 1, 2]], [[]], [[-2, 7]]],
         "oracle": lambda a, k: sorted((v * k for v in a[0]), reverse=True)},
        {"code": "def _solve(values):\n    found = -1\n    for index, value in enumerate(values):\n        if value < $K$:\n            found = index\n    return found\n",
         "spec": "return the index of the last value strictly less than $K$, or -1 when there is none",
         "inputs": lambda k: [[[k + 5, 0, k + 9, k - 3]], [[k, k + 1]], [[]]],
         "oracle": lambda a, k: _position(a[0], lambda item: item < k, True)},
        {"code": "def _solve(values):\n    if not values:\n        return 0\n    return min(value - $K$ for value in values)\n",
         "spec": "subtract $K$ from every list value and return the smallest result, or 0 when the list is empty",
         "inputs": lambda k: [[[10, 3, 25]], [[]]],
         "oracle": lambda a, k: min(v - k for v in a[0]) if a[0] else 0},
    ]


def _control():
    """Branch, recursion, and functional families: 0-4 train, 5-9 unseen."""
    return [
        {"code": "def _solve(x):\n    if x < 0:\n        return \"low\"\n    if x > $K$:\n        return \"high\"\n    return \"middle\"\n",
         "spec": "classify a number: return the text low when it is negative, high when it is greater than $K$, otherwise middle",
         "inputs": lambda k: [[-4], [0], [k + 6], [k]],
         "oracle": lambda a, k: "low" if a[0] < 0 else ("high" if a[0] > k else "middle")},
        {"code": "def _solve(n):\n    if n <= 1:\n        return $K$\n    return n * _solve(n - 1)\n",
         "spec": "recursively compute the factorial of n, returning $K$ for any n below two",
         "inputs": lambda k: [[5], [1], [7]],
         "oracle": lambda a, k: _factorial(a[0], k)},
        {"code": "def _solve(items, key):\n    return [item[key] + $K$ for item in items if item.get(key, 0) >= $K$]\n",
         "spec": "keep the dictionaries whose value at the requested key is at least $K$ and add $K$ to every kept value",
         "inputs": lambda k: [[[{"a": k + 4}, {"a": 1}, {"b": k}], "a"]],
         "oracle": lambda a, k: [item[a[1]] + k for item in a[0] if item.get(a[1], 0) >= k]},
        {"code": "def _solve(values):\n    return sorted(values, key=lambda value: abs(value - $K$))\n",
         "spec": "sort the list values by their absolute distance from $K$, placing the closest value first",
         "inputs": lambda k: [[[k + 3, k - 1, k, k + 8]], [[]]],
         "oracle": lambda a, k: sorted(a[0], key=lambda v: abs(v - k))},
        {"code": "def _solve(values, scale=$K$):\n    return sum(value * scale for value in values)\n",
         "spec": "return the sum of the list values multiplied by scale, whose default value is $K$",
         "inputs": lambda k: [[[1, 2, 3]], [[]], [[5, -2]]],
         "oracle": lambda a, k: sum(v * k for v in a[0])},
        {"code": "def _solve(a, b):\n    if a > b:\n        return a - b + $K$\n    if a < b:\n        return b - a - $K$\n    return 0\n",
         "spec": "return the positive difference of two numbers plus $K$, the negative difference minus $K$, or 0 when they are equal",
         "inputs": lambda k: [[k + 5, 2], [1, k + 6], [4, 4]],
         "oracle": lambda a, k: a[0] - a[1] + k if a[0] > a[1] else (a[1] - a[0] - k if a[0] < a[1] else 0)},
        {"code": "def _solve(values):\n    if not values:\n        return $K$\n    return values[0] + _solve(values[1:])\n",
         "spec": "recursively add the list values together and return $K$ for an empty list",
         "inputs": lambda k: [[[1, 2, 3]], [[]], [[-8, 12]]],
         "oracle": lambda a, k: sum(a[0]) + k},
        {"code": "def _solve(values):\n    return list(map(lambda value: value * $K$, filter(lambda value: value > 0, values)))\n",
         "spec": "keep the positive list values, multiply each by $K$, and return them as a list",
         "inputs": lambda k: [[[1, -2, 3]], [[]], [[0, 7]]],
         "oracle": lambda a, k: [v * k for v in a[0] if v > 0]},
        {"code": "def _solve(first, second):\n    return sorted(set(first) & set(second) | {$K$})\n",
         "spec": "return the sorted union of the intersection of two lists with the single element $K$",
         "inputs": lambda k: [[[1, k, 4], [k, 5]], [[], []]],
         "oracle": lambda a, k: sorted(set(a[0]) & set(a[1]) | {k})},
        {"code": "def _solve(values):\n    return [value if value > 0 else -value * $K$ for value in values]\n",
         "spec": "keep positive list values unchanged and multiply every remaining value by minus $K$",
         "inputs": lambda k: [[[2, -3, 0]], [[]]],
         "oracle": lambda a, k: [v if v > 0 else -v * k for v in a[0]]},
    ]


def _http():
    """Synchronous HTTP families: 0-4 train, 5-9 unseen."""
    return [
        {"code": "import requests\n\ndef _solve(url, token):\n    response = requests.get(url, headers={\"Authorization\": \"Bearer \" + token}, timeout=$K$)\n    response.raise_for_status()\n    return response.json()[\"base\"] + $K$\n",
         "spec": "send a GET request with a bearer authorization header, check the status, and return the JSON base field plus $K$",
         "inputs": lambda k: [[_url(k), "tk" + str(k)]],
         "oracle": lambda a, k: _API["base"] + k},
        {"code": "import requests\n\ndef _solve(url):\n    with requests.Session() as session:\n        response = session.get(url, params={\"offset\": $K$}, timeout=$K$)\n    return response.json()[\"offset\"] + $K$\n",
         "spec": "send a GET request through a session with the offset query parameter set to $K$, then return the returned offset plus $K$",
         "inputs": lambda k: [[_url(k)]],
         "oracle": lambda a, k: k + k},
        {"code": "import requests\n\ndef _solve(url):\n    response = requests.post(url, json={\"amount\": $K$}, timeout=$K$)\n    return response.json()[\"amount\"] * 2\n",
         "spec": "send a POST request carrying a JSON body whose amount is $K$ and return twice the amount the server reports",
         "inputs": lambda k: [[_url(k)]],
         "oracle": lambda a, k: k * 2},
        {"code": "import httpx\n\ndef _solve(url):\n    response = httpx.get(url, params={\"page\": $K$}, timeout=$K$)\n    return response.json()[\"page\"] + $K$\n",
         "spec": "send a GET request with httpx using the page query parameter set to $K$ and return the returned page plus $K$",
         "inputs": lambda k: [[_url(k)]],
         "oracle": lambda a, k: k + k},
        {"code": "import urllib3, json\n\ndef _solve(url):\n    response = urllib3.PoolManager().request(\"GET\", url)\n    return json.loads(response.data)[\"base\"] + $K$\n",
         "spec": "send a GET request with a urllib3 pool manager, decode the response bytes as JSON, and return base plus $K$",
         "inputs": lambda k: [[_url(k)]],
         "oracle": lambda a, k: _API["base"] + k},
        {"code": "import httpx\n\ndef _solve(url):\n    response = httpx.put(url, json={\"amount\": $K$}, timeout=$K$)\n    payload = response.json()\n    return {\"amount\": payload[\"amount\"], \"method\": payload[\"method\"]}\n",
         "spec": "send a PUT request with httpx whose JSON amount is $K$ and return a dictionary holding the reported amount and method",
         "inputs": lambda k: [[_url(k)]],
         "oracle": lambda a, k: {"amount": k, "method": "PUT"}},
        {"code": "import requests\n\ndef _solve(url):\n    response = requests.get(url, timeout=$K$)\n    return [item[\"value\"] for item in response.json()[\"items\"] if item[\"value\"] >= $K$]\n",
         "spec": "send a GET request and return the value field of every returned item whose value is at least $K$",
         "inputs": lambda k: [[_url(k)]],
         "oracle": lambda a, k: [item["value"] for item in _API["items"] if item["value"] >= k]},
        {"code": "import urllib3, json\n\ndef _solve(url):\n    response = urllib3.PoolManager().request(\"POST\", url, body=json.dumps({\"amount\": $K$}), headers={\"Content-Type\": \"application/json\"})\n    return json.loads(response.data)[\"amount\"] - $K$\n",
         "spec": "send a POST request with a urllib3 pool manager carrying a JSON encoded amount of $K$ and return the reported amount minus $K$",
         "inputs": lambda k: [[_url(k)]],
         "oracle": lambda a, k: 0},
        {"code": "import httpx\n\ndef _solve(url, token):\n    with httpx.Client() as client:\n        response = client.get(url, headers={\"Authorization\": token}, params={\"page\": $K$}, timeout=$K$)\n    payload = response.json()\n    return [payload[\"token\"], payload[\"page\"]]\n",
         "spec": "send a GET request through an httpx client with an authorization header and the page query parameter set to $K$, then return a list holding the token the server received and the reported page",
         "inputs": lambda k: [[_url(k), "tk" + str(k)]],
         "oracle": lambda a, k: [a[1], k]},
        {"code": "import requests\n\ndef _solve(url):\n    response = requests.post(url, data={\"amount\": $K$}, timeout=$K$)\n    if response.status_code != 200:\n        return -1\n    return response.json()[\"base\"]\n",
         "spec": "send a POST request with a URL encoded amount of $K$, return minus one unless the status code is 200, otherwise return the JSON base field",
         "inputs": lambda k: [[_url(k)]],
         "oracle": lambda a, k: _API["base"]},
    ]


def _remote():
    """Asynchronous and resilient HTTP families: 0-4 train, 5-9 unseen."""
    return [
        {"code": "import aiohttp\n\nasync def _solve(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return (await response.json())[\"base\"] + $K$\n",
         "spec": "asynchronously GET a URL with a session and return the JSON base field plus $K$", "module": "async",
         "inputs": lambda k: [[_url(k)]], "oracle": lambda a, k: _API["base"] + k},
        {"code": "import aiohttp\n\nasync def _solve(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.post(url, json={\"amount\": $K$}) as response:\n            return (await response.json())[\"amount\"] + $K$\n",
         "spec": "asynchronously POST a JSON body whose amount is $K$ and return the reported amount plus $K$", "module": "async",
         "inputs": lambda k: [[_url(k)]], "oracle": lambda a, k: k + k},
        {"code": "import aiohttp\n\nasync def _solve(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            payload = await response.json()\n    return [item[\"value\"] for item in payload[\"items\"] if item[\"value\"] > $K$ - 1]\n",
         "spec": "asynchronously GET a URL and return the value field of every item whose value is strictly greater than $K$ minus one", "module": "async",
         "inputs": lambda k: [[_url(k)]], "oracle": lambda a, k: [i["value"] for i in _API["items"] if i["value"] > k - 1]},
        {"code": "import aiohttp\n\nasync def _solve(url, token):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url, headers={\"Authorization\": token}, params={\"offset\": $K$}) as response:\n            payload = await response.json()\n    return [payload[\"token\"], payload[\"offset\"]]\n",
         "spec": "asynchronously GET a URL with an authorization header and the offset query parameter set to $K$, then return a list holding the token the server received and the reported offset", "module": "async",
         "inputs": lambda k: [[_url(k), "tk" + str(k)]], "oracle": lambda a, k: [a[1], k]},
        {"code": "import aiohttp\n\nasync def _solve(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url, params={\"offset\": $K$}) as response:\n            payload = await response.json()\n    return payload[\"offset\"] + $K$\n",
         "spec": "asynchronously GET a URL with the offset query parameter set to $K$ and return the reported offset plus $K$", "module": "async",
         "inputs": lambda k: [[_url(k)]], "oracle": lambda a, k: k + k},
        {"code": "import aiohttp\n\nasync def _solve(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.put(url, json={\"amount\": $K$}) as response:\n            payload = await response.json()\n    return [payload[\"amount\"], payload[\"method\"]]\n",
         "spec": "asynchronously PUT a JSON body whose amount is $K$ and return a list holding the reported amount and method", "module": "async",
         "inputs": lambda k: [[_url(k)]], "oracle": lambda a, k: [k, "PUT"]},
        {"code": "import aiohttp, json\n\nasync def _solve(url):\n    values = []\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            async for line in response.content:\n                values.append(json.loads(line)[\"value\"] + $K$)\n    return values\n",
         "spec": "asynchronously stream newline separated JSON objects and return each streamed value plus $K$", "module": "async",
         "inputs": lambda k: [[_url(k, True)]], "oracle": lambda a, k: [v + k for v in _API["values"]]},
        {"code": "import aiohttp\n\nasync def _solve(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return {\"status\": response.status, \"base\": (await response.json())[\"base\"] + $K$}\n",
         "spec": "asynchronously GET a URL and return a dictionary holding the response status code and the JSON base field plus $K$", "module": "async",
         "inputs": lambda k: [[_url(k)]], "oracle": lambda a, k: {"status": 200, "base": _API["base"] + k}},
        {"code": "import aiohttp\n\nasync def _solve(url):\n    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=$K$)) as session:\n        async with session.get(url) as response:\n            payload = await response.json()\n    return len(payload[\"items\"]) + $K$\n",
         "spec": "asynchronously GET a URL through a connector pool capped at $K$ connections and return the returned item count plus $K$", "module": "async",
         "inputs": lambda k: [[_url(k)]], "oracle": lambda a, k: len(_API["items"]) + k},
        {"code": "import aiohttp\n\nasync def _solve(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            payload = await response.json()\n    values = [item[\"value\"] for item in payload[\"items\"]]\n    return min(values) + max(values) + $K$\n",
         "spec": "asynchronously GET a URL and return the smallest item value plus the largest item value plus $K$", "module": "async",
         "inputs": lambda k: [[_url(k)]], "oracle": lambda a, k: min(i["value"] for i in _API["items"]) + max(i["value"] for i in _API["items"]) + k},
    ]


def _jsondomain():
    """Nested JSON and structure families: 0-4 train, 5-9 unseen."""
    return [
        {"code": "def _solve(data, key):\n    if isinstance(data, dict):\n        if key in data:\n            return data[key] + $K$\n        for value in data.values():\n            found = _solve(value, key)\n            if found is not None:\n                return found\n    if isinstance(data, list):\n        for value in data:\n            found = _solve(value, key)\n            if found is not None:\n                return found\n    return None\n",
         "spec": "recursively search nested dictionaries and lists for the first occurrence of the requested key and return its numeric value plus $K$, or None when the key is absent",
         "inputs": lambda k: [[{"meta": {"target": k}, "rows": [{"target": k + 1}, {"other": 0}], "flag": True}, "target"], [{"a": {"b": [1, 2]}}, "missing"]],
         "oracle": lambda a, k: _find(a[0], a[1], k)},
        {"code": "def _solve(data):\n    result = {}\n    separator = \".\" if $K$ % 2 == 0 else \"/\"\n\n    def _walk(item, prefix):\n        if isinstance(item, dict):\n            for name, value in item.items():\n                _walk(value, prefix + (separator if prefix else \"\") + str(name))\n        elif isinstance(item, list):\n            for index, value in enumerate(item):\n                _walk(value, prefix + separator + str(index))\n        else:\n            result[prefix] = item\n\n    _walk(data, \"\")\n    return result\n",
         "spec": "flatten nested dictionaries and lists into one dictionary whose keys join each path with a dot when $K$ is even and with a slash when $K$ is odd",
         "inputs": lambda k: [[{"a": {"b": [1, 2]}, "c": 3}], [{"x": []}]],
         "oracle": lambda a, k: _flatten(a[0], "." if k % 2 == 0 else "/")},
        {"code": "def _solve(data):\n    if isinstance(data, dict):\n        return sum(_solve(value) for value in data.values())\n    if isinstance(data, list):\n        return sum(_solve(value) for value in data)\n    if isinstance(data, int) and not isinstance(data, bool):\n        return data + $K$\n    return 0\n",
         "spec": "recursively add $K$ to every integer leaf of nested dictionaries and lists and return the total",
         "inputs": lambda k: [[{"a": 1, "b": [2, {"c": 3}]}], [[]], [{}], [{"x": True, "y": 4}]],
         "oracle": lambda a, k: _leafsum(a[0], k)},
        {"code": "def _solve(data, keys):\n    if not isinstance(data, dict):\n        return False\n    for name in keys:\n        value = data.get(name)\n        if not isinstance(value, str) or len(value.strip()) < $K$:\n            return False\n    return True\n",
         "spec": "return True only when every requested key exists and holds a nonblank string of at least $K$ characters",
         "inputs": lambda k: [[{"a": "value"}, ["a"]], [{"a": "  "}, ["a"]], [[], ["a"]], [{"a": "value"}, ["a", "b"]]],
         "oracle": lambda a, k: _valid(a[0], a[1], k)},
        {"code": "import json\n\ndef _solve(text, key):\n    result = []\n\n    def _walk(item):\n        if isinstance(item, dict):\n            for name, value in item.items():\n                if name == key:\n                    result.extend(value if isinstance(value, list) else [value])\n                else:\n                    _walk(value)\n        elif isinstance(item, list):\n            for value in item:\n                _walk(value)\n\n    _walk(json.loads(text))\n    return result[:$K$]\n",
         "spec": "parse a JSON string, collect every value stored under the requested key in traversal order, and return at most the first $K$ of them",
         "inputs": lambda k: [[json.dumps({"a": {"score": [k]}, "b": [{"score": k + 1}, {"score": 2}]}), "score"]],
         "oracle": lambda a, k: _collect(json.loads(a[0]), a[1])[:k]},
        {"code": "def _solve(data, path, score):\n    current = data\n    for name in path.split(\".\"):\n        if not isinstance(current.get(name), dict):\n            current[name] = {}\n        current = current[name]\n    current[\"score\"] = score + $K$\n    return data\n",
         "spec": "walk a dot separated path through nested dictionaries, create any missing level, store score plus $K$ under the score key, and return the original data",
         "inputs": lambda k: [[{}, "a.b", k], [{"a": {"b": {"score": 1}}}, "a.b", 2]],
         "oracle": lambda a, k: _setpath(json.loads(json.dumps(a[0])), a[1], a[2], k)},
        {"code": "def _solve(data):\n    if isinstance(data, dict):\n        return sum(_solve(value) for value in data.values())\n    if isinstance(data, list):\n        return sum(_solve(value) for value in data)\n    if isinstance(data, bool):\n        return 0\n    if isinstance(data, (int, float)):\n        return data * $K$\n    return 0\n",
         "spec": "recursively multiply every numeric leaf of nested dictionaries and lists by $K$ and return the total",
         "inputs": lambda k: [[{"a": 1, "b": [2, {"c": 3}]}], [[]], [{}], [{"x": True, "y": 2.5}]],
         "oracle": lambda a, k: _numscale(a[0], k)},
        {"code": "def _solve(records):\n    result = {}\n    for record in records:\n        result[record[\"id\"] + str($K$)] = record[\"value\"] * $K$\n    return result\n",
         "spec": "index a list of records by appending $K$ to each id and store each value multiplied by $K$",
         "inputs": lambda k: [[[{"id": "a", "value": 2}, {"id": "b", "value": 3}]], [[]]],
         "oracle": lambda a, k: {r["id"] + str(k): r["value"] * k for r in a[0]}},
        {"code": "def _solve(records):\n    return sorted([record for record in records if record.get(\"score\", 0) >= $K$], key=lambda record: record[\"name\"])\n",
         "spec": "keep the records whose score is at least $K$ and return them sorted by name",
         "inputs": lambda k: [[[{"name": "b", "score": k}, {"name": "a", "score": k + 1}, {"name": "c", "score": 0}]]],
         "oracle": lambda a, k: sorted([r for r in a[0] if r.get("score", 0) >= k], key=lambda r: r["name"])},
        {"code": "def _solve(records):\n    summary = {}\n    for record in records:\n        group = record.get(\"group\", \"none\")\n        summary[group] = summary.get(group, 0) + record[\"value\"] + $K$\n    return summary\n",
         "spec": "sum each record value plus $K$ into a dictionary keyed by the record group, using none when the group is absent",
         "inputs": lambda k: [[[{"group": "a", "value": 1}, {"group": "a", "value": 2}, {"value": 3}]], [[]]],
         "oracle": lambda a, k: _summary(a[0], k)},
    ]


def _textdomain():
    """CSV, regex, and string families: 0-4 train, 5-9 unseen."""
    return [
        {"code": "import csv, io\n\ndef _solve(text):\n    return list(csv.DictReader(io.StringIO(text)))[:$K$]\n",
         "spec": "parse comma separated text with the csv dictionary reader and return at most the first $K$ rows",
         "inputs": lambda k: [["name,amount\nAda,3\nBob,4\nCy,5\n"]],
         "oracle": lambda a, k: list(csv.DictReader(io.StringIO(a[0])))[:k]},
        {"code": "import csv, io\n\ndef _solve(text):\n    return [\"|\".join(row) for row in csv.reader(io.StringIO(text))][:$K$]\n",
         "spec": "parse comma separated text with the csv reader and return at most the first $K$ rows, each joined by a pipe",
         "inputs": lambda k: [["a,b\n1,2\n3,4\n"]],
         "oracle": lambda a, k: ["|".join(row) for row in csv.reader(io.StringIO(a[0]))][:k]},
        {"code": "import csv, io\n\ndef _solve(text, column):\n    total = 0.0\n    for row in csv.DictReader(io.StringIO(text)):\n        total += float(row[column])\n    return total * $K$\n",
         "spec": "sum one named numeric column of comma separated text and return the total multiplied by $K$",
         "inputs": lambda k: [["a,b\n1,2\n3,4\n", "b"], ["a,b\n1,2\n3,4\n", "a"]],
         "oracle": lambda a, k: sum(float(row[a[1]]) for row in csv.DictReader(io.StringIO(a[0]))) * k},
        {"code": "import re\n\ndef _solve(text):\n    return [int(value) * $K$ for value in re.findall(r\"[-+]?\\d+\", text)]\n",
         "spec": "extract every signed integer with a regular expression and multiply each one by $K$",
         "inputs": lambda k: [["a12 -3 +7 b"], ["none 0 here"]],
         "oracle": lambda a, k: [int(v) * k for v in re.findall(r"[-+]?\d+", a[0])]},
        {"code": "import re\n\ndef _solve(text):\n    return [word for word in re.findall(r\"[A-Za-z]+\", text) if len(word) >= $K$]\n",
         "spec": "return every alphabetic word of the text whose length is at least $K$",
         "inputs": lambda k: [["alpha be gamma d"], ["a b"]],
         "oracle": lambda a, k: [w for w in re.findall(r"[A-Za-z]+", a[0]) if len(w) >= k]},
        {"code": "import re\n\ndef _solve(text):\n    pattern = r\"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\"\n    return [found for found in re.findall(pattern, text.lower()) if len(found.split(\"@\")[0]) >= $K$]\n",
         "spec": "lowercase the text, extract email addresses with a regular expression, and keep those whose local part is at least $K$ characters",
         "inputs": lambda k: [["Ada.Long@Example.COM bob@t.io +bad@"]],
         "oracle": lambda a, k: [f for f in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", a[0].lower()) if len(f.split("@")[0]) >= k]},
        {"code": "import csv, io\n\ndef _solve(text):\n    rows = list(csv.reader(io.StringIO(text)))\n    output = io.StringIO(newline=\"\")\n    writer = csv.writer(output, delimiter=\";\" if $K$ % 2 == 0 else \"\\t\")\n    writer.writerows(rows)\n    return output.getvalue()\n",
         "spec": "read comma separated rows and rewrite them with a semicolon delimiter when $K$ is even and a tab delimiter when $K$ is odd",
         "inputs": lambda k: [["a,b\n1,2\n"]],
         "oracle": lambda a, k: _rewrite(a[0], k)},
        {"code": "import re\n\ndef _solve(text):\n    return re.findall(r\"[A-Za-z]+\\d+\", text)[:$K$]\n",
         "spec": "extract the tokens made of letters followed by digits and return at most the first $K$ of them",
         "inputs": lambda k: [["id9 order12 no yes42"], ["none"]],
         "oracle": lambda a, k: re.findall(r"[A-Za-z]+\d+", a[0])[:k]},
        {"code": "def _solve(text, width=$K$):\n    return [\"\".join(text[start:start + width].split()) for start in range(0, len(text), width)]\n",
         "spec": "split the text into fixed width chunks of $K$ characters and remove every whitespace character inside each chunk",
         "inputs": lambda k: [["a b cdef"], [""]],
         "oracle": lambda a, k: ["".join(a[0][s:s + k].split()) for s in range(0, len(a[0]), k)]},
        {"code": "def _solve(text, delimiter):\n    return [value.strip().upper() for value in text.split(delimiter) if value.strip() and len(value.strip()) >= $K$]\n",
         "spec": "split the text on a delimiter, keep the nonblank parts of at least $K$ characters, strip them, and return them uppercased",
         "inputs": lambda k: [[" a , bb , ccc , ", ","], ["x", "|"]],
         "oracle": lambda a, k: [v.strip().upper() for v in a[0].split(a[1]) if v.strip() and len(v.strip()) >= k]},
    ]


_TRANSFORMS = [
    ("value * $K$", lambda v, k: v * k, "multiply every value by $K$"),
    ("value + $K$", lambda v, k: v + k, "add $K$ to every value"),
    ("abs(value)", lambda v, k: abs(v), "take the absolute value of every value"),
    ("value * value", lambda v, k: v * v, "square every value"),
    ("value - $K$", lambda v, k: v - k, "subtract $K$ from every value"),
    ("-value", lambda v, k: -v, "negate every value"),
    ("value % $K$", lambda v, k: v % k, "take every value modulo $K$"),
    ("max(value, $K$)", lambda v, k: max(v, k), "raise every value below $K$ up to $K$"),
]
_AGGREGATES = [
    ("return [{E} for value in values]", lambda rows: rows, "return the results as a list"),
    ("return sum({E} for value in values)", lambda rows: sum(rows), "return the sum of the results"),
    ("return max({E} for value in values)", lambda rows: max(rows) if rows else 0, "return the largest result"),
    ("return min({E} for value in values)", lambda rows: min(rows) if rows else 0, "return the smallest result"),
    ("return sorted(({E} for value in values), reverse=True)", lambda rows: sorted(rows, reverse=True), "return the results in descending order"),
]
_FILTERS = [
    ("", lambda v, k: True, ""),
    (" if value % $K$ == 0", lambda v, k: v % k == 0, " only for values exactly divisible by $K$"),
    (" if value > $K$", lambda v, k: v > k, " only for values greater than $K$"),
    (" if value % 2 == 0", lambda v, k: v % 2 == 0, " only for even values"),
]


def _generated():
    """Compose transform, aggregate, and filter blocks into structurally distinct families."""
    out = []
    for expr, transform, verb in _TRANSFORMS:
        for aggregate, reduce, tail in _AGGREGATES:
            for condition, keep, qualifier in _FILTERS:
                body = aggregate.replace("{E}", expr)
                head = "def _solve(values):" + chr(10) + "    "
                if condition and "sum(" in aggregate or condition and "max(" in aggregate or condition and "min(" in aggregate:
                    body = body.replace("for value in values)", "for value in values" + condition + ")")
                elif condition:
                    body = body.replace("for value in values]", "for value in values" + condition + "]")
                code = head + body.replace("return ", "return ").strip() + chr(10)

                def oracle(arguments, k, transform=transform, reduce=reduce, keep=keep):
                    chosen = [transform(value, k) for value in arguments[0] if keep(value, k)]
                    return reduce(chosen)

                spec = verb + qualifier + ", then " + tail
                out.append({
                    "code": code,
                    "spec": spec,
                    "inputs": lambda k: [[[1, -2, 3, k + 2]], [[0, k, -k]], [[k * 2]], [[]]],
                    "oracle": oracle,
                })
    return out


_PREDICATES = [
    ("value > $K$", lambda v, k: v > k, "strictly greater than $K$"),
    ("value < $K$", lambda v, k: v < k, "strictly less than $K$"),
    ("value % $K$ == 0", lambda v, k: v % k == 0, "an exact multiple of $K$"),
    ("value == $K$", lambda v, k: v == k, "exactly equal to $K$"),
    ("value % 2 == 0", lambda v, k: v % 2 == 0, "an even number"),
    ("value % 3 == 0", lambda v, k: v % 3 == 0, "a multiple of three"),
    ("abs(value) > $K$", lambda v, k: abs(v) > k, "larger in absolute value than $K$"),
]


def _leaves(data):
    """Collect every integer leaf of nested dictionaries and lists in visit order."""
    out = []
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(reversed(list(item.values())))
        elif isinstance(item, list):
            stack.extend(reversed(item))
        elif isinstance(item, int) and not isinstance(item, bool):
            out.append(item)
    return out


def _generatecontrol():
    """Branch and selection families built from predicate, action, and fallback blocks."""
    out = []
    for test, holds, phrase in _PREDICATES:
        for code, spec, oracle in _controlcases(test, holds, phrase):
            out.append({"code": code, "spec": spec,
                        "inputs": lambda k: [[[k - 2, k, k + 3, k * 2]], [[k + 1, k + 5]], [[]]],
                        "oracle": oracle})
    return out


def _controlcases(test, holds, phrase):
    """Six structurally different families for one predicate."""
    nl = chr(10)
    return [
        ("def _solve(values):" + nl +
         "    for index, value in enumerate(values):" + nl +
         "        if " + test + ":" + nl +
         "            return index" + nl +
         "    return -1" + nl,
         "return the index of the first entry that is " + phrase + ", or -1 when there is none",
         lambda a, k, h=holds: next((i for i, v in enumerate(a[0]) if h(v, k)), -1)),
        ("def _solve(values):" + nl +
         "    count = 0" + nl +
         "    for value in values:" + nl +
         "        if " + test + ":" + nl +
         "            count += 1" + nl +
         "    return count" + nl,
         "count how many entries are " + phrase,
         lambda a, k, h=holds: sum(1 for v in a[0] if h(v, k))),
        ("def _solve(values):" + nl +
         "    return [value for value in values if " + test + "]" + nl,
         "keep only the entries that are " + phrase,
         lambda a, k, h=holds: [v for v in a[0] if h(v, k)]),
        ("def _solve(values):" + nl +
         "    if not values:" + nl +
         "        return $K$" + nl +
         "    for value in values:" + nl +
         "        if " + test + ":" + nl +
         "            return value" + nl +
         "    return values[0]" + nl,
         "return the first entry that is " + phrase + ", falling back to $K$ for an empty input and to the first element when nothing matches",
         lambda a, k, h=holds: k if not a[0] else next((v for v in a[0] if h(v, k)), a[0][0])),
        ("def _solve(values, flag=True):" + nl +
         "    if flag:" + nl +
         "        return [value for value in values if " + test + "]" + nl +
         "    return [value for value in values if not " + test + "]" + nl,
         "return the entries that are " + phrase + " when the flag is true and the entries that are not when it is false",
         lambda a, k, h=holds: [v for v in a[0] if h(v, k)] if (len(a) < 2 or a[1]) else [v for v in a[0] if not h(v, k)]),
        ("def _solve(values):" + nl +
         "    best = None" + nl +
         "    for value in values:" + nl +
         "        if " + test + ":" + nl +
         "            if best is None or value > best:" + nl +
         "                best = value" + nl +
         "    return best if best is not None else $K$" + nl,
         "return the largest entry that is " + phrase + ", or $K$ when nothing qualifies",
         lambda a, k, h=holds: max([v for v in a[0] if h(v, k)], default=k)),
    ]


def _generatejson():
    """Traversal and reduction families over nested dictionaries and lists."""
    out = []
    nl = chr(10)
    reductions = [
        ("sum", lambda rows: sum(rows) if rows else 0),
        ("largest", lambda rows: max(rows) if rows else 0),
        ("smallest", lambda rows: min(rows) if rows else 0),
    ]
    actions = [
        ("return data + $K$", lambda v, k: v + k, "add $K$ to every integer leaf and return their"),
        ("return abs(data)", lambda v, k: abs(v), "take the absolute value of every integer leaf and return their"),
        ("return data * $K$", lambda v, k: v * k, "multiply every integer leaf by $K$ and return their"),
    ]
    for name, reduce in reductions:
        for action, apply, verb in actions:
            if name == "sum":
                walk = ("if isinstance(data, dict):" + nl +
                        "        return sum(_solve(value) for value in data.values())" + nl +
                        "    if isinstance(data, list):" + nl +
                        "        return sum(_solve(value) for value in data)" + nl)
            elif name == "largest":
                walk = ("if isinstance(data, dict):" + nl +
                        "        return max((_solve(value) for value in data.values()), default=$K$)" + nl +
                        "    if isinstance(data, list):" + nl +
                        "        return max((_solve(value) for value in data), default=$K$)" + nl)
            else:
                walk = ("if isinstance(data, dict):" + nl +
                        "        return min((_solve(value) for value in data.values()), default=$K$)" + nl +
                        "    if isinstance(data, list):" + nl +
                        "        return min((_solve(value) for value in data), default=$K$)" + nl)
            code = ("def _solve(data):" + nl + "    " + walk +
                    "    if isinstance(data, int) and not isinstance(data, bool):" + nl +
                    "        " + action + nl +
                    "    return 0" + nl)

            def oracle(arguments, k, apply=apply, reduce=reduce):
                return reduce([apply(value, k) for value in _leaves(arguments[0])])

            out.append({"code": code,
                        "spec": "recursively " + verb + " " + name,
                        "inputs": lambda k: [[{"a": [k, k + 2], "b": {"c": k * 2}, "d": []}]],
                        "oracle": oracle})
    for wrap, phrase in (("item", "the integer leaves as a flat list in visit order"),
                         ("abs(item)", "the absolute value of every integer leaf as a flat list in visit order")):
        code = ("def _solve(data):" + nl +
                "    out = []" + nl + nl +
                "    def _walk(item):" + nl +
                "        if isinstance(item, dict):" + nl +
                "            for value in item.values():" + nl +
                "                _walk(value)" + nl +
                "        elif isinstance(item, list):" + nl +
                "            for value in item:" + nl +
                "                _walk(value)" + nl +
                "        elif isinstance(item, int) and not isinstance(item, bool):" + nl +
                "            out.append(" + wrap + ")" + nl + nl +
                "    _walk(data)" + nl +
                "    return out" + nl)
        out.append({"code": code,
                    "spec": "collect " + phrase + ", visiting dictionary values and list items in order",
                    "inputs": lambda k: [[{"a": [k, -2], "b": {"c": 3}}]],
                    "oracle": lambda a, k, w=wrap: [abs(v) if w.startswith("abs") else v for v in _leaves(a[0])]})
    return out


def _generatetext():
    """CSV and regular-expression families over fixed small tables."""
    out = []
    nl = chr(10)
    out.append({"code": "import csv, io" + nl * 2 +
                        "def _solve(text):" + nl +
                        "    rows = list(csv.DictReader(io.StringIO(text)))" + nl +
                        "    return rows[:$K$]" + nl,
                "spec": "parse comma separated text with the csv dictionary reader and return at most the first $K$ rows",
                "inputs": lambda k: [["name,amount" + nl + "Ada,3" + nl + "Bob,4" + nl + "Cy,5" + nl]],
                "oracle": lambda a, k: list(csv.DictReader(io.StringIO(a[0])))[:k]})
    for pattern, describe, sample in (
        ("[a-z]+", "every run of lowercase letters", "Ada wrote 42 lines in triple quotes"),
        ("[0-9]+", "every run of digits as text", "Ada wrote 42 lines in 7 files"),
        ("[A-Z][a-z]+", "every capitalised word", "Ada wrote 42 lines in Paris"),
        ("[aeiou]+", "every run of vowels", "Ada wrote 42 lines in triple quotes"),
        ("[^0-9\\s]+", "every run of non digit non space characters as text", "abc 123 Def 456 xy"),
    ):
        out.append({"code": "import re" + nl * 2 +
                            "def _solve(text):" + nl +
                            "    return re.findall(" + repr(pattern) + ", text)[:$K$]" + nl,
                    "spec": "find " + describe + " in the text with a regular expression and return at most the first $K$ matches",
                    "inputs": lambda k, sample=sample: [[sample]],
                    "oracle": lambda a, k, p=pattern: re.findall(p, a[0])[:k]})
    for separator, label, sample in ((",", "comma", "a,b,c" + nl + "1,2,3" + nl),
                                     (";", "semicolon", "a;b;c" + nl + "1;2;3" + nl),
                                     ("|", "pipe", "a|b|c" + nl + "1|2|3" + nl)):
        out.append({"code": "import csv, io" + nl * 2 +
                            "def _solve(text):" + nl +
                            "    rows = csv.reader(io.StringIO(text), delimiter=" + repr(separator) + ")" + nl +
                            "    return list(rows)[:$K$]" + nl,
                    "spec": "split " + label + " separated text with the csv reader and return at most the first $K$ rows as lists",
                    "inputs": lambda k, sample=sample: [[sample]],
                    "oracle": lambda a, k, d=separator: list(csv.reader(io.StringIO(a[0]), delimiter=d))[:k]})
    return out


def _generatehttp():
    """Synchronous HTTP families across requests, httpx, and urllib3 transports."""
    out = []
    nl = chr(10)
    bodies = [
        ("import requests" + nl * 2 +
         "def _solve(url, token):" + nl +
         "    response = requests.get(url, headers={\"Authorization\": \"Bearer \" + token}, timeout=$K$)" + nl +
         "    response.raise_for_status()" + nl +
         "    return response.json()[\"base\"] + $K$" + nl,
         "send a GET request with a bearer authorization header, check the status, and return the JSON base field plus $K$",
         lambda a, k: _API["base"] + k),
        ("import requests" + nl * 2 +
         "def _solve(url):" + nl +
         "    with requests.Session() as session:" + nl +
         "        response = session.get(url, params={\"offset\": $K$}, timeout=$K$)" + nl +
         "    return response.json()[\"offset\"] + $K$" + nl,
         "send a GET request through a session with the offset query parameter set to $K$, then return the returned offset plus $K$",
         lambda a, k: k + k),
        ("import requests" + nl * 2 +
         "def _solve(url):" + nl +
         "    response = requests.post(url, json={\"amount\": $K$}, timeout=$K$)" + nl +
         "    return response.json()[\"amount\"] * 2" + nl,
         "send a POST request carrying a JSON body whose amount is $K$ and return twice the amount the server reports",
         lambda a, k: k * 2),
        ("import httpx" + nl * 2 +
         "def _solve(url):" + nl +
         "    response = httpx.get(url, params={\"page\": $K$}, timeout=$K$)" + nl +
         "    return response.json()[\"page\"] + $K$" + nl,
         "send a GET request with httpx using the page query parameter set to $K$ and return the returned page plus $K$",
         lambda a, k: k + k),
        ("import urllib3, json" + nl * 2 +
         "def _solve(url):" + nl +
         "    response = urllib3.PoolManager().request(\"GET\", url)" + nl +
         "    return json.loads(response.data)[\"base\"] + $K$" + nl,
         "send a GET request with a urllib3 pool manager, decode the response bytes as JSON, and return base plus $K$",
         lambda a, k: _API["base"] + k),
        ("import requests" + nl * 2 +
         "def _solve(url):" + nl +
         "    response = requests.get(url, timeout=$K$)" + nl +
         "    payload = response.json()" + nl +
         "    return sorted(item[\"value\"] for item in payload[\"items\"])" + nl,
         "send a GET request and return the item values from the JSON payload in ascending order",
         lambda a, k: sorted(item["value"] for item in _API["items"])),
        ("import httpx" + nl * 2 +
         "def _solve(url):" + nl +
         "    with httpx.Client(timeout=$K$) as client:" + nl +
         "        response = client.get(url, params={\"page\": $K$})" + nl +
         "    return response.json()[\"method\"]" + nl,
         "open an httpx client with the timeout set to $K$, send a GET request with the page parameter, and return the method the server echoes back",
         lambda a, k: "GET"),
        ("import requests" + nl * 2 +
         "def _solve(url, token):" + nl +
         "    response = requests.get(url, headers={\"Authorization\": token}, timeout=$K$)" + nl +
         "    return response.json()[\"token\"] is not None" + nl,
         "send a GET request whose authorization header is the raw token and return whether the server echoed a header back",
         lambda a, k: True),
        ("import httpx" + nl * 2 +
         "def _solve(url):" + nl +
         "    response = httpx.post(url, json={\"amount\": $K$}, timeout=$K$)" + nl +
         "    response.raise_for_status()" + nl +
         "    return response.json()[\"amount\"] + $K$" + nl,
         "send a POST request with httpx carrying an amount of $K$, check the status, and return the reported amount plus $K$",
         lambda a, k: k + k),
        ("import urllib3, json" + nl * 2 +
         "def _solve(url):" + nl +
         "    pool = urllib3.PoolManager()" + nl +
         "    response = pool.request(\"POST\", url, body=json.dumps({\"amount\": $K$}), headers={\"Content-Type\": \"application/json\"})" + nl +
         "    return json.loads(response.data)[\"amount\"]" + nl,
         "send a POST request with a urllib3 pool manager and a JSON body whose amount is $K$, then return the amount the server reports",
         lambda a, k: k),
    ]
    for code, spec, oracle in bodies:
        # Only templates declaring a second parameter receive the token argument.
        takes = "token" in code.replace("_solve(", chr(1)).split(chr(1))[1].split(")")[0]
        out.append({"code": code, "spec": spec,
                    "inputs": (lambda k: [[_url(k), "tk" + str(k)]]) if takes else (lambda k: [[_url(k)]]),
                    "oracle": oracle})
    return out


def _generateremote():
    """Asynchronous HTTP families over aiohttp, including streamed bodies."""
    out = []
    nl = chr(10)
    bodies = [
        ("import aiohttp, asyncio" + nl * 2 +
         "async def _solve(url):" + nl +
         "    async with aiohttp.ClientSession() as session:" + nl +
         "        async with session.get(url, params={\"page\": $K$}) as response:" + nl +
         "            payload = await response.json()" + nl +
         "    return payload[\"page\"] + $K$" + nl,
         "open an aiohttp session, GET the url with the page parameter set to $K$, await the JSON body, and return the returned page plus $K$",
         lambda a, k: k + k),
        ("import aiohttp, asyncio" + nl * 2 +
         "async def _solve(url):" + nl +
         "    timeout = aiohttp.ClientTimeout(total=$K$)" + nl +
         "    async with aiohttp.ClientSession(timeout=timeout) as session:" + nl +
         "        async with session.post(url, json={\"amount\": $K$}) as response:" + nl +
         "            payload = await response.json()" + nl +
         "    return payload[\"amount\"] * 2" + nl,
         "open an aiohttp session with a total timeout of $K$, POST a JSON amount of $K$, await the JSON body, and return twice the reported amount",
         lambda a, k: k * 2),
        ("import aiohttp, asyncio" + nl * 2 +
         "async def _solve(url):" + nl +
         "    async with aiohttp.ClientSession() as session:" + nl +
         "        async with session.get(url) as response:" + nl +
         "            payload = await response.json()" + nl +
         "    return [item[\"value\"] for item in payload[\"items\"] if item[\"value\"] >= $K$]" + nl,
         "open an aiohttp session, GET the url, await the JSON body, and return the item values at least $K$",
         lambda a, k: [item["value"] for item in _API["items"] if item["value"] >= k]),
        ("import aiohttp, asyncio" + nl * 2 +
         "async def _solve(url):" + nl +
         "    values = []" + nl +
         "    async with aiohttp.ClientSession() as session:" + nl +
         "        async with session.get(url) as response:" + nl +
         "            async for line in response.content:" + nl +
         "                text = line.decode().strip()" + nl +
         "                if text:" + nl +
         "                    values.append(int(text.split(\":\")[-1].strip(\"}\").split(\":\")[-1]))" + nl +
         "    return values" + nl,
         "open an aiohttp session, GET the url, read the response content line by line, and return the value field of each JSON line as a list",
         lambda a, k: list(_API["values"])),
        ("import aiohttp, asyncio" + nl * 2 +
         "async def _solve(url):" + nl +
         "    async with aiohttp.ClientSession() as session:" + nl +
         "        async with session.get(url) as response:" + nl +
         "            payload = await response.json()" + nl +
         "    return sorted(set(item[\"value\"] for item in payload[\"items\"]))" + nl,
         "open an aiohttp session and return the distinct item values from the JSON payload in ascending order",
         lambda a, k: sorted(set(item["value"] for item in _API["items"]))),
        ("import aiohttp, asyncio" + nl * 2 +
         "async def _solve(url, token):" + nl +
         "    headers = {\"Authorization\": \"Bearer \" + token}" + nl +
         "    async with aiohttp.ClientSession(headers=headers) as session:" + nl +
         "        async with session.get(url) as response:" + nl +
         "            payload = await response.json()" + nl +
         "    return payload[\"base\"] + $K$" + nl,
         "open an aiohttp session carrying a bearer authorization header, GET the url, and return the JSON base field plus $K$",
         lambda a, k: _API["base"] + k),
        ("import aiohttp, asyncio" + nl * 2 +
         "async def _solve(url):" + nl +
         "    async with aiohttp.ClientSession() as session:" + nl +
         "        async with session.get(url, params={\"offset\": $K$}) as response:" + nl +
         "            payload = await response.json()" + nl +
         "    return payload[\"offset\"] + $K$" + nl,
         "open an aiohttp session, GET the url with the offset parameter set to $K$, and return the returned offset plus $K$",
         lambda a, k: k + k),
        ("import aiohttp, asyncio" + nl * 2 +
         "async def _solve(url):" + nl +
         "    async with aiohttp.ClientSession() as session:" + nl +
         "        async with session.get(url) as response:" + nl +
         "            return response.status" + nl,
         "open an aiohttp session and return the HTTP status code of a GET request to the url",
         lambda a, k: 200),
    ]
    for code, spec, oracle in bodies:
        takes = "token" in code.replace("_solve(", chr(1)).split(chr(1))[1].split(")")[0]
        stream = "async for line" in code
        if takes:
            supply = lambda k, s=stream: [[_url(k, s), "tk" + str(k)]]
        else:
            supply = lambda k, s=stream: [[_url(k, s)]]
        out.append({"code": code, "spec": spec, "inputs": supply,
                    "oracle": oracle, "module": "async"})
    return out


def _families():
    """Every family per domain, deduplicated by specification so no two tasks ask the same thing."""
    raw = {"loops": _loops() + _generated(), "control": _control() + _generatecontrol(), "http": _http() + _generatehttp(), "async": _remote() + _generateremote(), "json": _jsondomain() + _generatejson(), "text": _textdomain() + _generatetext()}
    clean = {}
    for domain, families in raw.items():
        kept, seen = [], set()
        for template in families:
            signature = template["spec"]
            if signature in seen:
                continue
            seen.add(signature)
            kept.append(template)
        clean[domain] = kept
    return clean


def _plain(value):
    """Convert an oracle result into strict JSON so probes travel to the sandbox unchanged."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, dict):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("expected values must be finite")
        return value
    if isinstance(value, int) or isinstance(value, str):
        return value
    raise ValueError("unsupported expected type: " + type(value).__name__)


def _rename(code, seed):
    """Rename solver parameters consistently, rewriting only NAME tokens so string literals stay intact."""
    import tokenize
    head, sep, body = code.partition("def _solve(")
    if not sep:
        return code
    arguments, close, rest = body.partition("):")
    names = _names(seed, 4)
    originals, renames = [], []
    for position, raw in enumerate(part.strip() for part in arguments.split(",")):
        if not raw:
            continue
        label, equal, default = raw.partition("=")
        originals.append(label)
        renames.append(names[position] + (equal + default if equal else ""))
    mapping = {original: fresh.partition("=")[0] for original, fresh in zip(originals, renames)}
    source = head + "def _solve(" + ", ".join(renames) + "):" + rest
    pieces, cursor = [], 0
    for token in tokenize.generate_tokens(iter(source.splitlines(True)).__next__):
        if token.type == tokenize.NAME and token.string in mapping:
            line, column = token.start
            row = source.splitlines(True)
            pieces.append(source[cursor:sum(len(item) for item in row[:line - 1]) + column])
            pieces.append(mapping[token.string])
            cursor = sum(len(item) for item in row[:line - 1]) + column + len(token.string)
    pieces.append(source[cursor:])
    return "".join(pieces)


def _challenge(domain, family, index, kind, variant, seed=0):
    """Build one unique challenge: bound constant, renamed solver parameters, verified probes."""
    template = _families()[domain][family]
    code = _rename(template["code"].replace("$K$", str(variant)), seed)
    spec = template["spec"].replace("$K$", str(variant))
    for token, word in (("values", "the list"), ("data", "the nested structure"), ("text", "the string"), ("records", "the record list"), ("items", "the dictionary list"), ("total", "the running total"), ("numbers", "the number list")):
        if token in spec:
            spec = re.sub("(?<![a-z0-9_])" + token + "(?![a-z0-9_])", word, spec)
    cases = [{"arguments": _plain(list(args)), "expected": _plain(template["oracle"](list(args), variant))} for args in template["inputs"](variant)]
    if "$K$" not in template["spec"]:
        # A family whose specification carries no constant would otherwise emit
        # the same prompt for every variant, so the seed discriminates it instead.
        spec = spec.rstrip() + " Treat the number " + str(variant) + " as the fixed constant of this task."
    lead = "Write Python code defining an async function _solve. " if template.get("module") == "async" else "Write Python code defining the function _solve. "
    tail = " Respond with the function code only." if kind.endswith("t") else " Reply with only the corrected function source."
    prompt = lead + spec + tail
    return {"id": domain + kind + str(family) + "v" + str(variant) + "s" + str(seed % 1000) + kind[0], "domain": domain, "family": family, "variant": variant, "prompt": prompt, "code": code, "cases": cases}


def _int(item):
    """Coerce one probe element to int, falling back to zero when it is not numeric."""
    try:
        return int(item)
    except (TypeError, ValueError):
        return 0


def _numlist(value):
    """Flatten a probe argument into a list of ints, or return an empty list when impossible."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        flat = []
        for item in value:
            flat.extend(_numlist(item))
        return flat
    if isinstance(value, dict):
        flat = []
        for item in value.values():
            flat.extend(_numlist(item))
        return flat
    return []


def _expand(record, domain, family, variant, roundindex, offset):
    """Grow one coding record with verified companion helpers so the answer carries real signal."""
    count = 3 + (roundindex + variant + family) % 4
    helpers, probes = [], []
    for index in range(count):
        shape = (index + family + variant) % 4
        name = "_helper" + str(index)
        degree = 1 + (variant + index) % 3
        shift = variant * (index + 1) % 5
        if shape == 0:
            source = "def " + name + "(value):" + chr(10) + "    return [item * " + str(degree) + " + " + str(shift) + " for item in value]"
            probe = lambda values, d=degree, s=shift: [_int(item) * d + s for item in values]
        elif shape == 1:
            source = "def " + name + "(value):" + chr(10) + "    return [item * " + str(degree) + " + " + str(shift) + " for item in value]"
            probe = lambda values, d=degree, s=shift: [_int(item) * d + s for item in values]
        elif shape == 2:
            source = "def " + name + "(value):" + chr(10) + "    return [item for item in value if item % " + str(degree + 1) + " == " + str(shift % (degree + 1)) + "]"
            probe = lambda values, d=degree + 1, s=shift % (degree + 1): [_int(item) for item in values if _int(item) % d == s]
        else:
            source = "def " + name + "(value):" + chr(10) + "    return [abs(item) + " + str(degree) + " for item in value]"
            probe = lambda values, d=degree: [abs(_int(item)) + d for item in values]
        helpers.append(source)
        probes.append((name, probe))
    first = record["cases"][0]["arguments"][0] if record["cases"] else []
    sample = _numlist(first)
    if not sample:
        record["helpers"] = []
        return record
    first = list(sample)
    chain = record["code"].rstrip() + chr(10) * 2 + (chr(10) * 2).join(helpers) + chr(10) * 2
    chain += "def _compose(value):" + chr(10) + "    seed = list(value)" + chr(10)
    for name, unused_probe in probes:
        chain += "    seed = " + name + "(seed)" + chr(10)
    chain += "    return seed" + chr(10)
    combined = [int(item) for item in sample]
    for unused_name, probe in probes:
        combined = probe(list(combined))
    record["code"] = chain
    record["cases"] = [{"arguments": [[int(item) for item in sample]], "expected": list(combined), "entry": "_compose"}]
    record["helpers"] = [name for name, unused in probes]
    return record
    record["code"] = chain
    record["cases"] = [{"arguments": [list(first)], "expected": list(combined), "entry": "_compose"}]
    record["helpers"] = [name for name, unused in probes]
    return record
    return record


def _coding(split, count):
    """Unique coding records for one partition; holdout draws unused variants and identifier seeds."""
    domaincount = count // len(_DOMAINS)
    records, seen = [], set()
    offset = 1000 if split == "holdout" else 0
    families = _families()
    for domain in _DOMAINS:
        produced, roundindex = 0, 0
        span = len(families[domain])
        while produced < domaincount:
            # Spread the requested count across every family before repeating any
            # of them, so a 500 record domain exercises all of its templates
            # instead of only the first few.
            perpass = max(1, (domaincount + span - 1) // max(1, span))
            order = list(range(span))
            for index in order:
                for variant in range(1, perpass + 1):
                    if produced >= domaincount:
                        break
                    family = index
                    tag = "h" if split == "holdout" else "t"
                    seed = offset + variant * 37 + index * 997 + roundindex * 7919 + len(domain) * 13
                    shift = 500 if split == "holdout" else 0
                    record = _challenge(domain, family, tag, "code" + tag, variant + roundindex * 211 + shift, seed)
                    _expand(record, domain, family, variant, roundindex, offset)
                    if record["prompt"] in seen or record["code"] in seen:
                        raise ValueError("duplicate coding record produced: " + record["id"])
                    seen.add(record["prompt"])
                    seen.add(record["code"])
                    records.append(record)
                    produced += 1
            roundindex += 1
    return records


_TOPICS = [
    ("What is the global interpreter lock in Python and how does it affect threading?", "The global interpreter lock, usually called the GIL, is a mutex owned by CPython that allows only one thread to execute Python bytecode at a time. It protects internal interpreter bookkeeping, such as reference counts, from concurrent mutation. The practical consequence is that pure-Python CPU-bound work does not speed up with threads: two threads busy in a numeric loop still run one at a time. I/O-bound work is unaffected because blocking calls release the GIL, so threads remain effective for network and disk waits. For CPU parallelism use multiprocessing or a native extension that releases the GIL."),
    ("How do decorators work in Python?", "A decorator is a callable that receives a function and returns a replacement callable. The at-sign syntax is pure sugar: writing the decorator above a def is the same as rebinding the name to decorator(function) after the definition. Most decorators close over the original function and return a wrapper that adds behaviour before or after calling it. Use functools.wraps on the wrapper so __name__, __doc__ and __wrapped__ survive. A decorator accepts arguments when you add one more layer: the outer callable takes the arguments and returns the actual decorator."),
    ("Explain generators and the yield keyword.", "A generator is a function containing yield; calling it returns a generator object without running the body. Each next resumes the frame at the last yield, so local state persists between resumes. Generators are lazy, which makes them ideal for streaming, pipelines and reading large files line by line without holding everything in memory. throw injects an exception at the paused point, close raises GeneratorExit, and send can pass a value back in as the result of the yield expression."),
    ("How do async and await work in Python?", "async def declares a coroutine function. Calling it returns a coroutine object that does nothing until it is awaited or scheduled. await suspends the current coroutine and yields control to the event loop, which can run other ready tasks meanwhile. Concurrency therefore comes from many suspended tasks, not from threads. asyncio.run starts a loop for the main coroutine. Never call blocking functions such as time.sleep or a synchronous HTTP client inside a coroutine, because they stall the whole loop. Use asyncio.sleep and an async client such as aiohttp instead."),
    ("How does Python manage memory and reference counting?", "CPython keeps a reference count on every object and destroys the object when the count reaches zero. A cyclic garbage collector handles reference cycles that counting alone cannot reclaim. References live in names, containers, closures and frames, and each one bumps the count by one. Weak references via the weakref module do not keep an object alive. sys.getrefcount reports one more than the true count because the argument itself is a temporary reference."),
    ("How do I read a traceback to find the real cause of an error?", "Read a traceback from the bottom up. The last line holds the exception type and message, which is the actual failure. The frames above show the call path, and the deepest frame is usually where the fault originates. The chaining markers matter: the message about handling the above exception means real code raised inside an except block, while the direct-cause message refers to a raise-from link. Fix the deepest frame that belongs to your code, not to a library."),
    ("What is the difference between a list and a tuple in Python?", "A list is mutable and resizable; a tuple is immutable and fixed once built. Tuples are hashable when their contents are hashable, so they can be dictionary keys or set members, while lists cannot. Tuples usually use slightly less memory and unpack naturally. Use a list for homogeneous sequences you intend to change, and a tuple for fixed records such as coordinates, rows returned from a query, or keys that must never be mutated by a callee."),
    ("Explain list comprehensions and generator expressions.", "A list comprehension builds a whole list eagerly inside brackets. A generator expression uses parentheses and produces values lazily, one at a time. Comprehensions are faster than an equivalent append loop because the loop runs in C, and they keep the transformation in one readable expression. Keep them to a single for and at most one if for readability; when you need several clauses an explicit loop is clearer. Watch memory: a comprehension over a large input materialises the entire result."),
    ("How does exception handling with try, except, finally and else work?", "The try block runs first. If it raises, matching except clauses are tested in order and the first match runs. The else block runs only when try completed without raising. The finally block always runs, even when an exception escapes or a return executes. Never use a bare except, which also swallows KeyboardInterrupt and SystemExit. Catch the narrowest exception you can handle, and prefer raise-from when you convert one error into another so the original cause stays visible."),
    ("What are context managers and the with statement?", "A context manager defines enter and exit methods, and with guarantees exit runs even if the body raises. That is how files close, locks release and transactions roll back. Write one with the contextlib.contextmanager decorator: a generator that yields the resource inside a try/finally. Returning True from exit suppresses the exception, which is rarely what you want. Multiple managers can be nested in one with statement, separated by commas."),
    ("What is the difference between a shallow copy and a deep copy?", "A shallow copy creates a new outer container but reuses the same inner objects, so mutating a nested list through the copy also changes the original. copy.copy performs a shallow copy, and slicing a list is also shallow. copy.deepcopy recursively reproduces every nested object, at the cost of time and memory and with a risk of hitting recursion limits on deeply self-referential structures. Choose a deep copy only when the caller must be fully isolated from the source."),
    ("How do I sort data in Python?", "list.sort sorts in place and returns None; sorted returns a new list and accepts any iterable. Both take key, a callable computing the sort key once per element, and reverse. For multi-field ordering return a tuple key, for example a key returning the group and the negated score to sort by group ascending and score descending. The sort is stable, so equal keys keep their original relative order, which lets you chain several passes. Sorting mixed types raises TypeError, so normalise types inside the key."),
    ("What is the difference between equality and identity comparison in Python?", "The double-equals operator tests equality by calling the equality method on the objects. The is operator tests identity: whether both names refer to the same object. Use is only for singletons such as None, True and False, and for sentinel objects you create yourself. Never compare numbers or strings with is; small-integer and short-string interning make it pass by accident for some values and fail for others, producing intermittent bugs. The one legitimate identity need is None handling."),
    ("How do dictionaries work internally and how should I use them?", "A dictionary is a hash table: the key is hashed, the hash selects a slot, and collisions are resolved by probing. Lookup averages constant time but degrades if many keys collide or if the table must resize. Keys must be hashable, meaning their hash and equality are consistent. To count occurrences use collections.Counter, to group use collections.defaultdict(list), and to preserve insertion order rely on the guarantee that dictionaries maintain it. Iterating while mutating raises RuntimeError, so iterate over a list copy instead."),
    ("What are type hints and how do I use them effectively?", "Type hints are annotations on parameters and return values, evaluated by tools rather than enforced at runtime. They document intent and let mypy or pyright catch mismatched calls before tests run. Import the future annotations feature to keep annotations unevaluated and avoid import cycles. Annotate public function boundaries and data structures, and lean on Optional, Union and Protocol where they express real contracts. Do not annotate every local variable; annotate what a caller needs to know."),
    ("What is the difference between deep recursion and iteration in Python?", "Python does not optimise tail calls, and the default recursion limit is around one thousand frames, so deep recursion raises RecursionError. Iteration uses one frame and constant stack, which is the correct choice for unbounded input. Recursion is still readable for naturally nested data such as trees and nested dictionaries, where the depth equals the structure depth and stays modest. When recursion depth is data-dependent, convert it to an explicit stack holding the pending work."),
    ("How do I work with files safely in Python?", "Open files with a with statement so the handle closes even on error. Always pass an explicit encoding, almost always utf-8, because the platform default varies and can silently corrupt text. Use an empty newline argument with the csv module. Read line by line for large files rather than calling read once. To write safely, write to a temporary file beside the target and call os.replace, so a crash never leaves a half-written destination. Open in binary mode for anything that is not text."),
    ("What is the difference between staticmethod, classmethod and instance methods?", "An instance method receives self and works on one object. A classmethod receives cls and usually builds alternative constructors or reads class-level configuration. A staticmethod receives neither and is a plain function namespaced under the class. Use a classmethod when a subclass should receive its own type back, which is why from-dict style constructors are classmethods. Use a staticmethod for helpers that genuinely belong to the class conceptually and touch no state."),
    ("How does the re module work and what are common pitfalls?", "findall returns strings for one group and tuples for several groups, which surprises many callers. match anchors at the start, search scans anywhere, and fullmatch requires the whole string. Compile patterns you reuse, and prefer raw string literals so backslashes reach the regex engine intact. Greedy quantifiers take as much as possible; add a question mark for lazy matching. Named groups make result extraction readable. Always consider input size, because a badly written nested quantifier can backtrack catastrophically."),
    ("How do I make an HTTP GET request with requests correctly?", "Use a Session when issuing more than one request so connections are reused. Pass a timeout on every call, because requests has no default timeout and will hang forever otherwise. Call raise_for_status so 4xx and 5xx responses become exceptions instead of silently parsed bodies. Send headers as a dictionary and JSON bodies with the json keyword. For retries use urllib3 Retry mounted on an HTTPAdapter with a bounded retry count, and for async code use aiohttp with a session opened in an async with block."),
    ("Why does my function return None unexpectedly?", "The most common cause is a function that computes a value but never returns it, so Python implicitly returns None. Trace every branch: a return that lives inside an if only fires on that path. Read the traceback bottom-up to find which caller received None, then check the callee for a missing return, an early return in a loop, or an assignment that shadows the result. Also confirm you are not discarding the value by calling the function for its side effects only."),
    ("Why does my list remain unchanged after I modify it inside a function?", "Function arguments are bound by object reference, not copied. Rebinding the parameter name inside the function only redirects the local name to a new object, leaving the caller's object untouched. Mutating the object in place, for example with append or item assignment, is visible to the caller. If you intended to transform the list, either mutate it explicitly and return it, or build and return a new list. A slice assignment is also in place and therefore visible to the caller."),
    ("Why do I get a KeyError when a key looks present?", "Dictionary keys are compared by equality and hash, so a key that prints the same can still differ in type or trailing whitespace. Check with a membership test before indexing, or use get with a default. Often the value is nested differently than assumed: the key exists at another level of the structure. Print the actual keys rather than the whole dictionary, and normalise keys when parsing external data so strings and integers do not mix."),
    ("Why does my async function run without awaiting?", "Calling an async function creates a coroutine object; the body only runs when that coroutine is awaited or scheduled. If you call it and discard the result, Python does nothing and may warn that the coroutine was never awaited. Await it directly inside another coroutine, or schedule it with an event loop task. Inside an ordinary synchronous function there is no loop to drive the coroutine, so the call silently has no effect."),
    ("Why does my regex fail to match across newlines?", "By default the dot does not match a newline and anchor metacharacters apply per string, not per line. Add the dot-matches-newline flag to let the dot consume newlines, or the multiline flag to make the anchors line-relative. Prefer explicit alternatives such as a negated character class when you want to stop at a delimiter. Test the pattern against the smallest failing input, and remember that greedy quantifiers must be made lazy when you want the shortest match."),
    ("Why is my deep-copied object still sharing state?", "A deep copy recursively copies containers, but immutable values are shared by design and custom classes must be copyable through the copy protocol. If a class defines its own copy method that returns self, or stores a module-level cache, the deep copy cannot break that sharing. Verify what each attribute points to after copying, and check that mutable class attributes are not being mistaken for instance attributes. A mutable default argument is a related trap, because the default is created once and reused across all calls."),
    ("Why does my loop only process the first item?", "A return statement inside the loop exits the whole function on the first iteration. To process every item, accumulate into a result list and return once after the loop. A break has the same early-exit effect on the loop, so confirm it is guarded by the condition you intend. When debugging, print the index at the top of each iteration to see how many times the body actually runs."),
    ("Why does my file handle stay open?", "A handle opened without a with statement is only closed when the object is garbage collected, which may be much later than you expect. Use a with statement so the file closes deterministically, even when an exception is raised inside the block. If you must manage it manually, close it in a finally block. On a long-running service, leaked handles eventually exhaust the process file-descriptor limit."),
    ("Why does my comparison of two floats return False?", "Floats are binary approximations, so decimal values that look equal can differ in the last bits. Never compare floats with exact equality; use math.isclose with a sensible relative and absolute tolerance. If exact decimal arithmetic is required, use the decimal module with an explicit context, or scale values to integers. Aggregating many float additions also accumulates rounding error, so sum in a stable order or use math.fsum."),
]


def _qa(pool, split, count):
    """Conversational records: explanations of Python concepts and fixes for buggy snippets."""
    records = []
    tag = "h" if split == "holdout" else "t"
    variant = 1037 if split == "holdout" else 11
    for index in range(count):
        topic, answer = pool[index % len(pool)]
        ask = topic if split == "train" else "In Python 3, answer this precisely: " + topic.replace("?", " ?")
        messages = [("user", ask), ("assistant", answer)]
        records.append({"id": "qa" + tag + str(index), "domain": "qa", "family": index % len(pool), "variant": variant + index, "messages": messages, "code": "", "cases": []})
    return records


_BUGS = [
    ("def _solve(numbers):\n    total = 0\n    for value in numbers\n        total += value\n    return total\n", "def _solve(numbers):\n    total = 0\n    for value in numbers:\n        total += value\n    return total\n", "there is a missing colon after the for clause on the header line", [{"arguments": [[1, 2, 3]], "expected": 6}]),
    ("def _solve(values):\n    result = []\n    for index in range(len(values) + 1):\n        result.append(values[index])\n    return result\n", "def _solve(values):\n    result = []\n    for index in range(len(values)):\n        result.append(values[index])\n    return result\n", "the range bound runs one past the last valid index and raises IndexError", [{"arguments": [[4, 5]], "expected": [4, 5]}]),
    ("def _solve(data):\n    result = {}\n    for key in data:\n        result[key] = result[key] + 1\n    return result\n", "def _solve(data):\n    result = {}\n    for key in data:\n        result[key] = result.get(key, 0) + 1\n    return result\n", "the accumulator is indexed before it has been created, raising KeyError", [{"arguments": [["a", "a", "b"]], "expected": {"a": 2, "b": 1}}]),
    ("def _solve(lines):\n    output = \"\"\n    for line in lines:\n        output += line + \"\\n\"\n    return output.strip\n", "def _solve(lines):\n    output = \"\"\n    for line in lines:\n        output += line + \"\\n\"\n    return output.strip()\n", "the final call to strip is never invoked because the parentheses are missing", [{"arguments": [["a", "b"]], "expected": "a\nb"}]),
    ("def _solve(values):\n    best = None\n    for value in values:\n        if best is None or value < best:\n            best = value\n    return best * 2\n", "def _solve(values):\n    if not values:\n        return 0\n    best = values[0]\n    for value in values[1:]:\n        if value < best:\n            best = value\n    return best * 2\n", "an empty input leaves best as None and the multiplication raises TypeError", [{"arguments": [[]], "expected": 0}, {"arguments": [[3, 1, 9]], "expected": 2}]),
    ("def _solve(text):\n    words = text.split()\n    return words[len(words)]\n", "def _solve(text):\n    words = text.split()\n    if not words:\n        return \"\"\n    return words[len(words) - 1]\n", "indexing with the list length points one past the final element and raises IndexError", [{"arguments": ["a b c"], "expected": "c"}, {"arguments": [""], "expected": ""}]),
    ("def _solve(values):\n    doubled = []\n    for value in values:\n        doubled.append(value * 2)\n    return doubled[:-1]\n", "def _solve(values):\n    doubled = []\n    for value in values:\n        doubled.append(value * 2)\n    return doubled\n", "the trailing slice silently discards the last result", [{"arguments": [[1, 2]], "expected": [2, 4]}]),
    ("def _solve(a, b):\n    return a / b\n", "def _solve(a, b):\n    if b == 0:\n        return 0\n    return a / b\n", "dividing by zero raises ZeroDivisionError instead of a controlled answer", [{"arguments": [4, 2], "expected": 2.0}, {"arguments": [4, 0], "expected": 0}]),
    ("def _solve(values):\n    total = 0\n    for value in values:\n        total += value\n        return total\n", "def _solve(values):\n    total = 0\n    for value in values:\n        total += value\n    return total\n", "the return inside the loop stops after the first item", [{"arguments": [[1, 2, 3]], "expected": 6}]),
    ("def _solve(data, key):\n    return data[key]\n", "def _solve(data, key):\n    return data.get(key, 0)\n", "a missing key raises KeyError instead of returning a default", [{"arguments": [{"a": 1}, "a"], "expected": 1}, {"arguments": [{"a": 1}, "b"], "expected": 0}]),
    ("def _solve(values):\n    total = 0\n    for value in values:\n        total = total + str(value)\n    return total\n", "def _solve(values):\n    total = 0\n    for value in values:\n        total = total + value\n    return total\n", "concatenating a string to an integer accumulator raises TypeError", [{"arguments": [[1, 2, 3]], "expected": 6}]),
    ("def _solve(text):\n    return text.lower.split()\n", "def _solve(text):\n    return text.lower().split()\n", "the method reference is returned instead of being called", [{"arguments": ["A B"], "expected": ["a", "b"]}]),
]


def _debug(pool, split, count):
    """Bug-fixing dialogues: user shows failing code, assistant explains and repairs it."""
    records = []
    tag = "h" if split == "holdout" else "t"
    for index in range(count):
        faulty, fixed, reason, cases = pool[index % len(pool)]
        prefix = "This Python function fails or misbehaves. Explain why and give the corrected code:" if split == "train" else "Review this failing Python snippet, name the exact defect, then return the repaired function:"
        question = prefix + chr(10) + faulty
        answer = "The defect is that " + reason + ". Here is the corrected function:\n" + fixed
        records.append({"id": "dbg" + tag + str(index), "domain": "qa", "family": index % len(pool), "variant": index, "messages": [("user", question), ("assistant", answer)], "code": fixed, "cases": cases, "faulty": faulty})
    return records


_OUT = [
    ("Write a JavaScript function that reverses an array.", "javascript"), ("How do I allocate memory with malloc in C++?", "cpp"), ("Write a Rust struct with a lifetime parameter.", "rust"),
    ("Give me a SQL query that joins three tables and groups by month.", "sql"), ("Write a bash script that counts files in a directory.", "bash"), ("How do I style a button with CSS flexbox?", "css"),
    ("Write a TypeScript interface for an API response.", "typescript"), ("How do I build a Go HTTP server with middleware?", "go"), ("Explain Java generics bounded wildcards.", "java"),
    ("Write a PHP session handler.", "php"), ("How do I compile a Kotlin coroutine with Gradle?", "kotlin"), ("Write a Swift delegate protocol.", "swift"),
    ("What is the best recipe for a vegetarian lasagna?", "cooking"), ("Which team won the last football championship?", "sports"), ("Who should I vote for in the next election?", "politics"),
    ("Write a poem about autumn leaves falling slowly.", "creative"), ("What is the capital of Mongolia?", "trivia"), ("Recommend a good financial plan for retirement.", "finance"),
    ("How do I train for a marathon in twelve weeks?", "fitness"), ("What is the plot of a famous novel about a whale?", "literature"), ("Translate this sentence into Mandarin.", "language"),
    ("What is the meaning of life?", "philosophy"), ("Explain the offside rule in soccer.", "sports"), ("Give me a plan to learn to paint portraits.", "art"),
    ("What is the weather like in Paris tomorrow?", "weather"), ("How do I fix a leaking kitchen faucet?", "home"), ("Which stock should I buy this month?", "finance"),
    ("Write a movie script about a time traveler.", "creative"), ("What are the symptoms of a common cold?", "health"), ("How do I make sourdough bread at home?", "cooking"),
    ("What is the chemical formula for table salt?", "science"), ("Write a C program that forks a child process.", "c"), ("Explain quantum entanglement in simple terms.", "science"),
    ("Write a SQL injection exploit for a login form.", "security"), ("How do I configure an nginx reverse proxy for a domain?", "devops"), ("Write a Haskell function using a fold.", "haskell"),
    ("What is the difference between a lawyer and a paralegal?", "law"), ("How do I register a car in another country?", "admin"), ("Describe the history of the Roman empire.", "history"),
    ("Write Scala code for an akka actor.", "scala"), ("How do I set up a Kubernetes cluster with kubectl?", "devops"), ("Write a Lua script for a game inventory.", "lua"),
    ("Give me a workout routine for strength training.", "fitness"), ("What is the best camera for landscape photography?", "gadgets"), ("How do I compose a symphony movement?", "music"),
    ("Write an Elixir GenServer module.", "elixir"), ("How do I reverse a linked list in C++?", "cpp"), ("Explain the rules of contract law.", "law"),
    ("What time is it in Tokyo right now?", "trivia"), ("Write a Perl one-liner to parse a log file.", "perl"), ("How do I decorate a wedding venue?", "event"),
]


def _outofdomain(split, count):
    """Guardrail records: every non-Python request pairs with the exact mandated refusal."""
    records = []
    tag = "h" if split == "holdout" else "t"
    for index in range(count):
        question, topic = _OUT[index % len(_OUT)]
        if split != "train":
            question = "Please help me with this: " + question
        records.append({"id": "ood" + tag + str(index), "domain": "guardrail", "family": index % len(_OUT), "variant": index, "topic": topic, "messages": [("user", question), ("assistant", _REFUSAL)], "code": "", "cases": [], "expected": _REFUSAL})
    return records


def _module(domain, index, rng, size):
    """Generate one filler Python module whose lines are syntactically valid and domain-flavoured."""
    name = "module" + domain + str(index)
    lines = ["\"\"\"" + domain + " support module " + str(index) + "\"\"\""]
    if domain in ("http", "async"):
        lines.append("import aiohttp" if domain == "async" else "import requests")
    if domain == "text":
        lines.append("import re")
    if domain == "json":
        lines.append("import json")
    lines.append("")
    functions = max(2, size // 14)
    for position in range(functions):
        identifier = "_helper" + str(position)
        lines.append("def " + identifier + "(value):")
        if domain == "loops":
            lines.append("    total = 0")
            lines.append("    for item in value:")
            lines.append("        total += item")
            lines.append("    return total")
        elif domain == "control":
            lines.append("    if value:")
            lines.append("        return value * 2")
            lines.append("    return 0")
        elif domain == "http":
            lines.append("    return value")
        elif domain == "async":
            lines.append("    return value")
        elif domain == "json":
            lines.append("    return {str(key): item for key, item in value.items()}")
        else:
            lines.append("    return [part.strip() for part in str(value).split()]")
        lines.append("")
    return name, lines


def _longcontext(count, split, tokenizer=None, low=2048, high=4096):
    """Multi-module retrieval records sized to the mandated 2,048-4,096 token training range."""
    tag = "h" if split == "holdout" else "t"
    records = []
    rng = random.Random(90210 if split == "train" else 19012)
    for index in range(count):
        domain = _DOMAINS[index % len(_DOMAINS)]
        target = rng.randint(low, high)
        lines, counted = [], 0
        while counted < target:
            fresh = _module(domain, index * 50 + len(lines), rng, 60)[1]
            lines.extend(fresh)
            lines.append("")
            chunk = chr(10).join(fresh) + chr(10)
            counted += len(tokenizer._encode(chunk)) if tokenizer is not None else max(1, len(chunk) // 6)
        constant = index * 3 + 7
        position = (index % 7) / 8.0 + 0.08
        context = _needle(chr(10).join(lines), constant, position)
        function = "def _lookup():" + chr(10) + "    return MAGIC" + chr(10)
        question = "The following Python module defines a constant named MAGIC at module level and a function _lookup that returns it. Report the integer value of MAGIC." + chr(10) + context + chr(10) + function
        records.append({"id": "lc" + tag + str(index), "domain": "longcontext", "family": index % len(_DOMAINS), "variant": index, "messages": [("user", question), ("assistant", str(constant))], "code": "", "cases": [], "expected": str(constant), "needle": constant, "position": position, "characters": len(context)})
    return records



def _needle(context, value, position):
    """Embed a single answer-bearing definition at a chosen depth inside a large code context."""
    lines = context.split("\n")
    insert = max(1, min(len(lines) - 1, int(len(lines) * position)))
    needle = "MAGIC = " + str(value)
    return "\n".join(lines[:insert] + [needle, ""] + lines[insert:])


def _retrieval(count, tokens, seed, tokenizer=None):
    """Long-context holdout probes built to the mandated 128k token input size, in linear time."""
    rng = random.Random(seed)
    records = []
    for index in range(count):
        domain = _DOMAINS[index % len(_DOMAINS)]
        blocks, counted = [], 0
        while counted < tokens:
            lines = _module(domain, index * 100 + len(blocks), rng, 60)[1]
            blocks.extend(lines)
            blocks.append("")
            chunk = chr(10).join(lines) + chr(10)
            counted += len(tokenizer._encode(chunk)) if tokenizer is not None else max(1, len(chunk) // 6)
        context = chr(10).join(blocks)
        value = 100000 + index * 17
        position = (index % 9) / 10.0 + 0.05
        context = _needle(context, value, position)
        question = "A Python codebase follows. Locate the module level constant named MAGIC and report its integer value. Answer with the integer only." + chr(10) + context
        records.append({"id": "ret" + str(index), "domain": "longcontext", "family": index % len(_DOMAINS), "variant": index, "messages": [("user", question), ("assistant", str(value))], "code": "", "cases": [], "expected": str(value), "needle": value, "position": position, "characters": len(context)})
    return records



_TRAINCODE, _HOLDCODE, _TRAINQA, _HOLDQA, _TRAINLONG, _HOLDLONG, _TRAINOOD, _HOLDOOD = 3000, 300, 1800, 150, 600, 50, 600, 100


def _generate(tokenizer=None, retrievaltokens=131072):
    """Build both partitions: 6,000 training records and 600 unseen holdout records."""
    train = _coding("train", _TRAINCODE)
    hold = _coding("holdout", _HOLDCODE)
    train += _qa(_TOPICS, "train", _TRAINQA - _TRAINQA // 3)
    hold += _qa(_TOPICS, "holdout", _HOLDQA - _HOLDQA // 3)
    train += _debug(_BUGS, "train", _TRAINQA // 3)
    hold += _debug(_BUGS, "holdout", _HOLDQA // 3)
    train += _outofdomain("train", _TRAINOOD)
    hold += _outofdomain("holdout", _HOLDOOD)
    train += _longcontext(_TRAINLONG, "train", tokenizer)
    hold += _retrieval(_HOLDLONG, int(retrievaltokens), 777, tokenizer)
    for record in train + hold:
        record["split"] = "train" if record in train else "holdout"
    trainids, holdids = {record["id"] for record in train}, {record["id"] for record in hold}
    if trainids & holdids:
        raise ValueError("partitions share identifiers")
    trainprompts = {_prompt(record) for record in train}
    holdprompts = {_prompt(record) for record in hold}
    if trainprompts & holdprompts:
        raise ValueError("holdout reuses a training prompt")
    if len(train) != _TRAINCODE + _TRAINQA + _TRAINOOD + _TRAINLONG:
        raise ValueError("training partition has the wrong size")
    if len(hold) != _HOLDCODE + _HOLDQA + _HOLDOOD + _HOLDLONG:
        raise ValueError("holdout partition has the wrong size")
    return train, hold


def _prompt(record):
    """Render the chat text a record presents to the model."""
    first = record["messages"][0][1] if record.get("messages") else record["prompt"]
    return first


def _verify(records):
    """Mandatory validation: AST parse every code sample and sandbox-execute every probe set."""
    checks = [record for record in records if record.get("code")]
    def __check(record):
        ast.parse(record["code"])
        if record.get("cases"):
            outcome = _execute(record["code"], record["cases"], timeout=_TIMEOUT)
            if not outcome["passed"]:
                raise ValueError(record["id"] + " failed verification: " + outcome["error"] + " | " + outcome["traceback"][-600:])
            return round(float(outcome["latency"]), 4)
        return 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        latencies = list(pool.map(__check, checks))
    for record, latency in zip(checks, latencies):
        record["verified"] = latency
    return latencies


def _storedata(train, hold):
    """Serialize both partitions and the corpus manifest into src/data."""
    paths = _paths()
    _atomicjson(train, paths["train"])
    _atomicjson(hold, paths["holdout"])
    manifest = {"name": "Chronos Alpha", "version": 2, "domains": list(_DOMAINS), "train": len(train), "holdout": len(hold), "verified": sum(1 for record in train + hold if record.get("verified") is not None), "digest": _digest([_prompt(record) for record in train + hold]), "entries": [{"id": record["id"], "domain": record["domain"], "split": record["split"], "family": record["family"], "variant": record["variant"], "executionseconds": record.get("verified")} for record in train + hold]}
    _atomicjson(manifest, paths["manifest"])
    return {"train": len(train), "holdout": len(hold), "manifest": manifest["digest"]}


def _loaddata(split):
    """Read one serialized partition, regenerating and verifying it when it is missing."""
    paths = _paths()
    key = "train" if split == "train" else "holdout"
    if not os.path.isfile(paths[key]):
        train, hold = _generate()
        _verify(train + hold)
        _storedata(train, hold)
    with open(paths[key], "r", encoding="utf-8") as stream:
        records = json.load(stream)
    return records


def _text(record, tokenizer):
    """Render a record as a chat sequence of ids, supervising only the assistant span."""
    system = _SYSTEM
    messages = record.get("messages")
    if messages is None:
        messages = [("user", record["prompt"]), ("assistant", record["code"])]
    return tokenizer._chat(messages, system)


def _tensors(tokenizer, records, cfg, width):
    """Pre-tokenize records into padded tensors, supervising assistant and code spans only."""
    rows, masks = [], []
    for record in records:
        ids = _text(record, tokenizer)
        marker = tokenizer._assistant()
        if marker in ids and len(ids) - 1 > width:
            head = ids.index(marker)
            if head + 1 >= width:
                keep = max(1, width - 192)
                ids = ids[:8] + ids[max(8, head - keep):]
            ids = ids[:width + 1]
        elif len(ids) - 1 > width:
            ids = ids[:width + 1]
        target = [0.0] * (len(ids) - 1)
        begin = None
        for index, token in enumerate(ids[:-1]):
            if token == tokenizer._assistant():
                # The token that follows the assistant marker is the first emitted
                # token of the answer, so position `index` itself must carry loss.
                # Marking from index + 1 taught the model everything except how
                # to open a reply, which left the first answer token at chance.
                begin = index
                target[index] = 1.0
            elif begin is not None:
                target[index] = 1.0
        rows.append(ids)
        masks.append(target)
    tokens = torch.zeros(len(rows), width, dtype=torch.long)
    targets = torch.zeros(len(rows), width, dtype=torch.long)
    weights = torch.zeros(len(rows), width, dtype=torch.float32)
    for index, (ids, mask) in enumerate(zip(rows, masks)):
        tokens[index, :len(ids) - 1] = torch.tensor(ids[:-1], dtype=torch.long)
        targets[index, :len(ids) - 1] = torch.tensor(ids[1:], dtype=torch.long)
        weights[index, :len(mask)] = torch.tensor(mask, dtype=torch.float32)
    return tokens.contiguous(), targets.contiguous(), weights.contiguous()


class Corpus(torch.utils.data.Dataset):
    """In-memory contiguous token matrix; training steps never touch the disk."""

    def __init__(self, tokens, targets, masks):
        self._tokens, self._targets, self._masks = tokens, targets, masks

    def __len__(self):
        return int(self._tokens.shape[0])

    def __getitem__(self, index):
        return self._tokens[index], self._targets[index], self._masks[index]


def _collate(samples):
    """Pad each batch to its own longest supervised span so padding never wastes compute."""
    tokens, targets, masks = zip(*samples)
    width = max(2, max(int(mask.nonzero().max()) + 2 if int(mask.sum()) else 2 for mask in masks))
    return torch.stack([row[:width] for row in tokens]).contiguous(), torch.stack([row[:width] for row in targets]).contiguous(), torch.stack([row[:width] for row in masks]).contiguous()


def _prepare(tokenizer, cfg, verify=True, rebuild=False):
    """Verify the corpus, serialize it, and build the in-memory training stream."""
    train, hold = _generate(tokenizer, int(cfg["retrievaltokens"]))
    if verify:
        _verify(train + hold)
    _storedata(train, hold)
    tokens, targets, masks = _tensors(tokenizer, train, cfg, int(cfg["sequence"]))
    lengths = masks.sum(dim=1)
    order = torch.argsort(lengths, stable=True).tolist()
    buckets = [order[begin:begin + int(cfg["bucket"])] for begin in range(0, len(order), int(cfg["bucket"]))]
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    loader = BucketLoader(tokens, targets, masks, buckets, int(cfg["batch"]), generator)
    return {"records": train, "holdout": hold, "loader": loader, "tokens": tokens, "targets": targets, "masks": masks, "width": int(tokens.shape[1]), "batches": len(loader), "totaltokens": int(masks.sum())}


class BucketLoader:
    """Serve batches of near-equal length so padding never dominates the arithmetic."""

    def __init__(self, tokens, targets, masks, buckets, batch, generator):
        self._tokens, self._targets, self._masks = tokens, targets, masks
        self._buckets, self._batch, self._generator = buckets, batch, generator
        self._order, self._position = [], 0
        self._shuffle()

    def _shuffle(self):
        """Reshuffle bucket order and drop buckets that cannot fill a batch."""
        order = torch.randperm(len(self._buckets), generator=self._generator).tolist()
        self._order = [self._buckets[index] for index in order if len(self._buckets[index]) >= self._batch]
        self._position = 0

    def __len__(self):
        return sum(len(bucket) // self._batch for bucket in self._order)

    def __iter__(self):
        self._shuffle()
        return self

    def __next__(self):
        """Return one tightly padded batch, reshuffling when the epoch is exhausted."""
        while self._position < len(self._order):
            bucket = self._order[self._position]
            if len(bucket) < self._batch:
                self._position += 1
                continue
            picks = torch.randperm(len(bucket), generator=self._generator)[:self._batch].tolist()
            rows = [bucket[index] for index in picks]
            cutoff = 1
            for row in rows:
                occupied = torch.nonzero(self._masks[row] > 0).flatten()
                if occupied.numel():
                    cutoff = max(cutoff, int(occupied[-1]) + 2)
            cutoff = max(1, min(cutoff, self._tokens.shape[1]))
            index = torch.tensor(rows, dtype=torch.long)
            return self._tokens.index_select(0, index)[:, :cutoff], self._targets.index_select(0, index)[:, :cutoff], self._masks.index_select(0, index)[:, :cutoff]
        self._shuffle()
        if not self._order:
            raise StopIteration
        return self.__next__()



def _batch(data, cfg, step):
    """Deterministic cycling batch sampler over the pre-tokenized corpus."""
    cursor = data.get("cursor")
    batch = next(cursor, None) if cursor is not None else None
    if batch is None:
        cursor = iter(data["loader"])
        data["cursor"] = cursor
        batch = next(cursor)
    return batch
