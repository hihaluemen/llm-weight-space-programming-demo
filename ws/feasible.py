"""Does a k-parameter repair EXIST? Search with gradient descent on mixed-length data (2-100).

This is an existence check, not a fair baseline: the search sees long strings and many
candidate masks. If any candidate reaches long>=0.99 and xlong>=0.97, a k-edit fix exists.

Candidate masks (exactly <= k entries) for each (model, k):
  flash-warm / flash-orig   positions DeepSeek changed in its k-limited runs; start from its
                            values or from the original values
  topk                      largest |dL/dw| on the mixed-length data
  unit{u}-a                 counter recipe on unit u: g_u('('), g_u(')'), f_u bias, + (k-3) head entries
  unit{u}-b                 same + i_u bias, + (k-4) head entries
  (unit recipes start saturated: g = +-20 with the original sign, f/i bias = 20; head entries are
   the top-|grad| head parameters after saturating)

Run:  uv run python -m ws.feasible
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import lstm, tasks
from .finetune import finetune, grad_magnitude
from .train import Net

MODELS = ["runs/baselines/h4_n8192_lr0.003_st10000_s1.json",   # positive control (flash fixed it)
          "runs/baselines/h4_n8192_lr0.003_st10000_s0.json",
          "runs/baselines/h4_n8192_lr0.01_st10000_s1.json"]
KS = [4, 6]
HEAD = ("W1", "b1", "W2", "b2")
SAT = 20.0


def empty_mask(w):
    return {k: np.zeros_like(v, dtype=bool) for k, v in w.items()}


def grads(w, xs, ys):
    H, M = lstm.dims(w)
    net = Net(H, M)
    net.load_weights(w)
    return grad_magnitude(net, xs[:2048], ys[:2048])


def top_entries(g, keys, n, exclude=None):
    """The n largest-|grad| entries among tensors `keys`, as (key, index) pairs."""
    cands = []
    for k in keys:
        for idx in np.ndindex(g[k].shape):
            if exclude is None or not exclude[k][idx]:
                cands.append((g[k][idx], k, idx))
    cands.sort(key=lambda t: -t[0])
    return [(k, idx) for _, k, idx in cands[:n]]


def unit_candidate(w0, u, k, with_i, xs, ys):
    H = lstm.dims(w0)[0]
    I, F, G = 0, H, 2 * H
    w = {kk: v.copy() for kk, v in w0.items()}
    m = empty_mask(w0)
    sign = 1.0 if w0["W_ih"][G + u, 0] >= w0["W_ih"][G + u, 1] else -1.0
    w["W_ih"][G + u] = [sign * SAT, -sign * SAT]
    m["W_ih"][G + u, :] = True
    w["b"][F + u] = SAT
    m["b"][F + u] = True
    if with_i:
        w["b"][I + u] = SAT
        m["b"][I + u] = True
    used = 4 if with_i else 3
    for key, idx in top_entries(grads(w, xs, ys), HEAD, k - used):
        m[key][idx] = True
    return w, m


def flash_runs(init, k):
    out = []
    for d in sorted(glob.glob("runs/agent/*_fix_*")):
        p = Path(d) / "result.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        if r["init"].replace("\\", "/") == init and r.get("edit_limit") == k:
            out.append((Path(d).name, lstm.load(Path(d) / "weights.json")))
    return out


def run_one(job):
    torch.set_num_threads(1)
    init, k, name, start, mask = job
    t0 = time.time()
    xs, ys = tasks.sample(8192, 2, 100, 88_001)
    w0 = lstm.load(init)
    if mask is None:  # computed here so it parallelises
        if name == "topk":
            start = w0
            mask = empty_mask(w0)
            for key, idx in top_entries(grads(w0, xs, ys), list(w0), k):
                mask[key][idx] = True
        else:  # unit{u}-a / unit{u}-b
            u, variant = int(name[4]), name[-1]
            start, mask = unit_candidate(w0, u, k, variant == "b", xs, ys)
    w = finetune(start, xs, ys, mask, steps=2000, lr=3e-2)
    changed = int(sum(np.count_nonzero(w[kk] != w0[kk]) for kk in w))
    assert changed <= k, (name, changed, k)
    return {"init": init, "k": k, "candidate": name, "changed": changed,
            **lstm.evaluate(w), "secs": round(time.time() - t0, 1)}


def main():
    from multiprocessing import Pool

    jobs = []
    for init in MODELS:
        w0 = lstm.load(init)
        H = lstm.dims(w0)[0]
        for k in KS:
            for run, wf in flash_runs(init, k):
                m = {kk: wf[kk] != w0[kk] for kk in w0}
                if sum(v.sum() for v in m.values()) == 0:
                    continue
                jobs.append((init, k, f"flash-warm:{run[-12:]}", wf, m))
                jobs.append((init, k, f"flash-orig:{run[-12:]}", w0, m))
            jobs.append((init, k, "topk", None, None))
            for u in range(H):
                for v in ("a", "b"):  # at k=4, "b" is the 4 recipe entries with no head entry
                    jobs.append((init, k, f"unit{u}-{v}", None, None))

    out = Path("runs/feasible")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs", flush=True)
    with Pool(8) as pool:
        for res in pool.imap_unordered(run_one, jobs):
            print(json.dumps(res), flush=True)
            with open(out / "results.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
