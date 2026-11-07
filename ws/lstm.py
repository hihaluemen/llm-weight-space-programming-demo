"""Tiny LSTM classifier as a plain dict of weights + an exact float64 numpy forward pass.

The numpy forward is the single source of truth for evaluation, for both gradient-trained
and LLM-written weights. Gate order follows PyTorch: i, f, g, o.

    z_t = W_ih @ x_t + W_hh @ h_{t-1} + b            (shape 4H)
    i, f, o = sigmoid(z[0:H]), sigmoid(z[H:2H]), sigmoid(z[3H:4H])
    g       = tanh(z[2H:3H])
    c_t = f * c_{t-1} + i * g
    h_t = o * tanh(c_t)                                (h_0 = c_0 = 0)
    logit = W2 @ relu(W1 @ h_T + b1) + b2              accept iff logit > 0
"""

from __future__ import annotations

import json

import numpy as np

from . import tasks

IN = 2  # one-hot '(' , ')'


def shapes(hidden: int, head: int) -> dict[str, tuple[int, ...]]:
    H, M = hidden, head
    return {
        "W_ih": (4 * H, IN),
        "W_hh": (4 * H, H),
        "b": (4 * H,),
        "W1": (M, H),
        "b1": (M,),
        "W2": (1, M),
        "b2": (1,),
    }


def zeros(hidden: int, head: int) -> dict[str, np.ndarray]:
    return {k: np.zeros(s) for k, s in shapes(hidden, head).items()}


def dims(w: dict[str, np.ndarray]) -> tuple[int, int]:
    return w["W_hh"].shape[1], w["W1"].shape[0]


def n_params(w: dict[str, np.ndarray]) -> int:
    return int(sum(v.size for v in w.values()))


def validate(w: dict, hidden: int, head: int) -> dict[str, np.ndarray]:
    """Coerce to float arrays and check shapes/finiteness; raise ValueError with a clear message."""
    want = shapes(hidden, head)
    out = {}
    for k, s in want.items():
        if k not in w:
            raise ValueError(f"missing tensor {k!r}")
        a = np.asarray(w[k], dtype=np.float64)
        if a.shape != s:
            raise ValueError(f"{k} must have shape {list(s)}, got {list(a.shape)}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{k} contains non-finite values")
        out[k] = a
    extra = set(w) - set(want)
    if extra:
        raise ValueError(f"unknown tensors {sorted(extra)}")
    return out


def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # numerically stable sigmoid


def forward(w: dict[str, np.ndarray], s: str, trace: bool = False):
    H, _ = dims(w)
    h, c = np.zeros(H), np.zeros(H)
    steps = []
    for x in tasks.one_hot(s):
        z = w["W_ih"] @ x + w["W_hh"] @ h + w["b"]
        i, f, g, o = _sig(z[:H]), _sig(z[H : 2 * H]), np.tanh(z[2 * H : 3 * H]), _sig(z[3 * H :])
        c = f * c + i * g
        h = o * np.tanh(c)
        if trace:
            steps.append({"i": i, "f": f, "g": g, "o": o, "c": c.copy(), "h": h.copy()})
    a = np.maximum(0.0, w["W1"] @ h + w["b1"])
    logit = float((w["W2"] @ a + w["b2"])[0])
    return (logit, steps, a) if trace else logit


def forward_batch(w: dict[str, np.ndarray], xs: list[str]) -> np.ndarray:
    """Vectorised over a batch by grouping equal lengths. Returns logits."""
    H, _ = dims(w)
    logits = np.empty(len(xs))
    by_len: dict[int, list[int]] = {}
    for idx, s in enumerate(xs):
        by_len.setdefault(len(s), []).append(idx)
    for L, idxs in by_len.items():
        X = np.stack([tasks.one_hot(xs[i]) for i in idxs])  # (B, L, 2)
        h = np.zeros((len(idxs), H))
        c = np.zeros((len(idxs), H))
        for t in range(L):
            z = X[:, t] @ w["W_ih"].T + h @ w["W_hh"].T + w["b"]
            i, f = _sig(z[:, :H]), _sig(z[:, H : 2 * H])
            g, o = np.tanh(z[:, 2 * H : 3 * H]), _sig(z[:, 3 * H :])
            c = f * c + i * g
            h = o * np.tanh(c)
        a = np.maximum(0.0, h @ w["W1"].T + w["b1"])
        logits[idxs] = (a @ w["W2"].T + w["b2"])[:, 0]
    return logits


def accuracy(w, xs, ys) -> float:
    return float(np.mean((forward_batch(w, xs) > 0).astype(int) == ys))


def evaluate(w, split_names=("test_id", "test_long", "test_xlong")) -> dict[str, float]:
    return {name: accuracy(w, *tasks.split(name)) for name in split_names}


def to_json(w: dict[str, np.ndarray], ndigits: int | None = None) -> str:
    def r(a):
        return np.round(a, ndigits).tolist() if ndigits is not None else a.tolist()

    return json.dumps({k: r(v) for k, v in w.items()})


def save(w, path):
    with open(path, "w") as f:
        f.write(to_json(w))


def load(path) -> dict[str, np.ndarray]:
    with open(path) as f:
        d = json.load(f)
    return {k: np.asarray(v, dtype=np.float64) for k, v in d.items()}
