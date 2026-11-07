"""Hand-built reference solutions (never shown to the editor LLM).

Proves the architecture (hidden=4, head=4) can solve each task exactly at any length, and
sanity-checks the data: simple shortcut rules should score well below 100%.

Run:  uv run python -m tests.oracle
"""

import numpy as np

from ws import lstm, tasks

BIG = 20.0


def _gates(H):
    return 0, H, 2 * H, 3 * H  # I, F, G, O row offsets


def _counter(w, u, inc0, inc1):
    """Unit u integrates g: +inc0 on symbol 0, +inc1 on symbol 1 (|inc| <= 1 via tanh)."""
    H = w["W_hh"].shape[1]
    I, F, G, O = _gates(H)
    to_z = lambda v: BIG * np.sign(v) if abs(v) == 1 else np.arctanh(v)  # noqa: E731
    w["b"][I + u] = BIG
    w["b"][F + u] = BIG
    w["W_ih"][G + u] = [to_z(inc0), to_z(inc1)]
    w["b"][O + u] = BIG


def _latch(w, u, src):
    """Unit u latches to ~1 when symbol 1 is read while unit `src` (a counter) reads zero."""
    H = w["W_hh"].shape[1]
    I, F, G, O = _gates(H)
    w["W_ih"][I + u] = [0.0, BIG]
    w["W_hh"][I + u, src] = -2 * BIG
    w["b"][I + u] = -BIG / 2
    w["b"][F + u] = BIG
    w["b"][G + u] = BIG
    w["b"][O + u] = BIG


def _head_zero_and_clear(w, zero_unit, latch_unit):
    """accept iff h[zero_unit] == 0 and h[latch_unit] == 0"""
    w["W1"][0, zero_unit] = 10.0
    w["W1"][1, zero_unit] = -10.0
    w["W1"][2, latch_unit] = 10.0
    w["W2"][0, :3] = -10.0
    w["b2"][0] = 5.0


def reference(task="dyck1", hidden=4, head=4):
    w = lstm.zeros(hidden, head)
    if task == "dyck1":
        _counter(w, 0, +1, -1)       # depth
        _latch(w, 1, src=0)          # depth went negative
        _head_zero_and_clear(w, zero_unit=0, latch_unit=1)
    elif task == "a2b":
        _counter(w, 0, +1, -1)       # #a - #b       (prefix condition)
        _latch(w, 1, src=0)          # some prefix had #b > #a
        _counter(w, 2, +0.5, -1)     # #a/2 - #b     (ratio condition)
        _head_zero_and_clear(w, zero_unit=2, latch_unit=1)
    else:
        raise ValueError(task)
    return w


def shortcut_acc(task, split_name):
    """Best single-condition shortcut: final count only."""
    xs, ys = tasks.split(split_name)
    if task == "dyck1":
        pred = [int(s.count("(") == s.count(")")) for s in xs]
    else:
        pred = [int(s.count("a") == 2 * s.count("b")) for s in xs]
    return float(np.mean(np.array(pred) == ys))


if __name__ == "__main__":
    for task in tasks.TASKS:
        tasks.set_task(task)
        print(f"== {task}")
        for name in tasks.SPLITS:
            xs, ys = tasks.split(name)
            lens = [len(s) for s in xs]
            kinds = {}
            for s in xs:
                k = tasks.kind(s)
                kinds[k] = kinds.get(k, 0) + 1
            print(f"  {name:11s} n={len(xs):5d} len={min(lens)}-{max(lens)} pos={ys.mean():.2f} "
                  f"count-only acc={shortcut_acc(task, name):.3f}  "
                  + " | ".join(f"{k.split(' (')[0]}: {v}" for k, v in kinds.items()))
        w = reference(task)
        res = lstm.evaluate(w, tuple(tasks.SPLITS))
        print("  reference LSTM:", res, "params:", lstm.n_params(w))
        assert all(v == 1.0 for v in res.values()), res
        xs, _ = tasks.split("practice")
        a = lstm.forward_batch(w, xs[:50])
        b = np.array([lstm.forward(w, s) for s in xs[:50]])
        assert np.allclose(a, b)
    print("OK")
