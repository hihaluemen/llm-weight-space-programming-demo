"""Recognition tasks for the tiny LSTM. The active task comes from the WS_TASK env var
(default "dyck1"); every module calls the functions below, which dispatch on it.

dyck1  Is a string of '(' and ')' balanced?  Even lengths only. Hard negatives:
         "flip"   one bracket flipped                     -> final count is +-2
         "rotate" cyclic rotation cut at nonzero depth     -> count 0, depth goes negative
       A count-only rule gets ~75%.

a2b    Strings over 'a','b'. Accept iff #a == 2*#b AND every prefix has #a >= #b.
       Lengths are multiples of 3 only. Hard negatives:
         "flip"   one letter flipped                       -> ratio broken
         "prefix" a permutation with the right counts whose prefix condition fails
       No textbook LSTM construction: it needs two counters with different increments
       (+1/-1 for the prefix check, +0.5/-1 for the ratio) plus a latch.

Data for dyck1 is generated exactly as before the multi-task refactor (same RNG stream), so
earlier results stay comparable.
"""

from __future__ import annotations

import functools
import os

import numpy as np

# Fixed seeds per split so the editor's practice data never overlaps evaluation data.
SPLITS = {
    # name: (min_len, max_len, n_examples, seed)
    "practice": (2, 20, 512, 1001),   # what the LLM editor may evaluate on
    "val": (2, 20, 1000, 3001),       # checkpoint selection for gradient baselines
    "test_id": (2, 20, 2000, 2001),   # in-distribution test
    "test_long": (40, 100, 2000, 2002),
    "test_xlong": (200, 500, 1000, 2003),
}


def name() -> str:
    return os.environ.get("WS_TASK", "dyck1")


def set_task(task: str):
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {sorted(TASKS)}")
    os.environ["WS_TASK"] = task  # inherited by multiprocessing workers


def runs_dir() -> str:
    """dyck1 keeps the original runs/ layout; other tasks get runs/<task>/."""
    return "runs" if name() == "dyck1" else f"runs/{name()}"


# ----------------------------------------------------------------------------- dyck1

def _balanced(n_pairs: int, rng: np.random.Generator) -> str:
    out, depth = [], 0
    remaining = 2 * n_pairs
    opens_left = n_pairs
    while remaining:
        if depth == 0:
            c = "("
        elif opens_left == 0:
            c = ")"
        else:
            c = "(" if rng.random() < 0.5 else ")"
        out.append(c)
        if c == "(":
            depth += 1
            opens_left -= 1
        else:
            depth -= 1
        remaining -= 1
    return "".join(out)


def is_balanced(s: str) -> bool:
    d = 0
    for c in s:
        d += 1 if c == "(" else -1
        if d < 0:
            return False
    return d == 0


def _flip(s: str, rng: np.random.Generator) -> str:
    i = int(rng.integers(len(s)))
    return s[:i] + (")" if s[i] == "(" else "(") + s[i + 1 :]


def _rotate(s: str, rng: np.random.Generator) -> str:
    depths = np.cumsum([1 if c == "(" else -1 for c in s])
    cuts = [k + 1 for k in range(len(s) - 1) if depths[k] != 0]
    k = int(rng.choice(cuts))
    return s[k:] + s[:k]


def _dyck_sample(n, min_len, max_len, seed):
    rng = np.random.default_rng(seed)
    lo, hi = max(1, min_len // 2), max_len // 2
    xs, ys = [], []
    for _ in range(n):
        s = _balanced(int(rng.integers(lo, hi + 1)), rng)
        r = rng.random()
        if r < 0.5:
            x = s
        elif r < 0.75 or len(s) < 4:
            x = _flip(s, rng)
        else:
            x = _rotate(s, rng)
        xs.append(x)
        ys.append(int(is_balanced(x)))
    return xs, np.array(ys, dtype=np.int64)


def _dyck_kind(s: str) -> str:
    if is_balanced(s):
        return "balanced (should accept)"
    if s.count("(") != s.count(")"):
        return "unequal counts (should reject)"
    return "equal counts but depth goes negative (should reject)"


# ----------------------------------------------------------------------------- a2b

def is_a2b(s: str) -> bool:
    na = nb = 0
    for c in s:
        if c == "a":
            na += 1
        else:
            nb += 1
        if na < nb:
            return False
    return na == 2 * nb


def _prefix_ok(s: str) -> bool:
    d = 0
    for c in s:
        d += 1 if c == "a" else -1
        if d < 0:
            return False
    return True


def _a2b_perm(m: int, rng, want_prefix_ok: bool) -> str | None:
    """Random arrangement of 2m a's and m b's whose prefix condition == want_prefix_ok."""
    base = np.array(["a"] * (2 * m) + ["b"] * m)
    for _ in range(200):
        s = "".join(rng.permutation(base))
        if _prefix_ok(s) == want_prefix_ok:
            return s
    return None


def _a2b_sample(n, min_len, max_len, seed):
    rng = np.random.default_rng(seed)
    lo, hi = max(1, -(-min_len // 3)), max_len // 3  # m with 3m in [min_len, max_len]
    xs, ys = [], []
    for _ in range(n):
        m = int(rng.integers(lo, hi + 1))
        r = rng.random()
        if r < 0.5:
            x = _a2b_perm(m, rng, True) or "aab" * m
        elif r < 0.75:
            s = _a2b_perm(m, rng, True) or "aab" * m
            i = int(rng.integers(len(s)))
            x = s[:i] + ("b" if s[i] == "a" else "a") + s[i + 1 :]
        else:
            x = _a2b_perm(m, rng, False) or "b" + "a" * (2 * m) + "b" * (m - 1)
        xs.append(x)
        ys.append(int(is_a2b(x)))
    return xs, np.array(ys, dtype=np.int64)


def _a2b_kind(s: str) -> str:
    if is_a2b(s):
        return "valid (should accept)"
    if s.count("a") != 2 * s.count("b"):
        return "wrong a:b ratio (should reject)"
    return "ratio 2:1 but some prefix has more b than a (should reject)"


# ----------------------------------------------------------------------------- registry

TASKS = {
    "dyck1": {
        "alphabet": "()",
        "sample": _dyck_sample,
        "label": is_balanced,
        "kind": _dyck_kind,
        "describe": (
            "Input: a string over the two characters '(' and ')'. Output: accept iff the string is a\n"
            "balanced (properly nested) bracket sequence, e.g. \"(())()\" accept, \"())(()\" reject, "
            "\"(()\" reject."),
    },
    "a2b": {
        "alphabet": "ab",
        "sample": _a2b_sample,
        "label": is_a2b,
        "kind": _a2b_kind,
        "describe": (
            "Input: a string over the two characters 'a' and 'b'. Output: accept iff BOTH hold:\n"
            "  (1) the number of a's is exactly twice the number of b's, and\n"
            "  (2) in every prefix of the string, the number of a's is >= the number of b's.\n"
            "Examples: \"aab\" accept, \"aba\" accept, \"aabaab\" accept, \"baa\" reject (prefix \"b\"),\n"
            "\"aaab\" reject (ratio), \"abbaaa\" reject (prefix \"abb\"), \"aabb\" reject (ratio)."),
    },
}


def current() -> dict:
    return TASKS[name()]


def alphabet() -> str:
    return current()["alphabet"]


def label(s: str) -> bool:
    return current()["label"](s)


def kind(s: str) -> str:
    return current()["kind"](s)


def sample(n: int, min_len: int, max_len: int, seed: int) -> tuple[list[str], np.ndarray]:
    """n strings and 0/1 labels (1 = accept), ~50% positives, for the active task."""
    return current()["sample"](n, min_len, max_len, seed)


@functools.lru_cache(maxsize=None)
def _split_cached(task: str, split_name: str):
    lo, hi, n, seed = SPLITS[split_name]
    return TASKS[task]["sample"](n, lo, hi, seed)


def split(split_name: str) -> tuple[list[str], np.ndarray]:
    return _split_cached(name(), split_name)


def one_hot(s: str) -> np.ndarray:
    alpha = alphabet()
    x = np.zeros((len(s), len(alpha)))
    x[np.arange(len(s)), [alpha.index(c) for c in s]] = 1.0
    return x
