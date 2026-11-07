"""Gradient repair baselines: fine-tune a trained LSTM while changing only a few parameters.

Masks (which entries may change):
  full        all parameters
  topk{k}     the k entries with the largest |dL/dw| on the fine-tuning data (sparse fine-tuning)
  llm         exactly the entries DeepSeek changed in a given fix run (gives gradient the LLM's answer
              to "where to edit"; it still has to find the values)
Data:
  short       lengths 2-20 -- the same information the LLM editor gets   (fair comparison)
  long        lengths 40-100 -- MORE information than the LLM gets        (reference only)

Run:  uv run python -m ws.finetune
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import lstm, tasks
from .train import Net, _batch

# torch parameter name -> weights-dict key
PMAP = {"rnn.weight_ih_l0": "W_ih", "rnn.weight_hh_l0": "W_hh", "rnn.bias_ih_l0": "b",
        "rnn.bias_hh_l0": "b", "fc1.weight": "W1", "fc1.bias": "b1", "fc2.weight": "W2", "fc2.bias": "b2"}

DATA = {"short": (2, 20, 77_001), "long": (40, 100, 77_002)}  # seeds disjoint from all splits


def grad_magnitude(net: Net, xs, ys) -> dict[str, np.ndarray]:
    net.zero_grad()
    Y = torch.tensor(ys, dtype=torch.float32)
    for i in range(0, len(xs), 512):
        X, lens = _batch(xs[i : i + 512])
        loss = nn.functional.binary_cross_entropy_with_logits(net(X, lens), Y[i : i + 512], reduction="sum")
        loss.backward()
    g = {}
    for name, p in net.named_parameters():
        k = PMAP[name]
        a = p.grad.detach().double().numpy()
        g[k] = g.get(k, 0) + a  # b gets bias_ih + bias_hh grads (they are equal anyway)
    return {k: np.abs(v) for k, v in g.items()}


def topk_mask(g: dict[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    flat = np.concatenate([g[key].ravel() for key in g])
    thresh = np.sort(flat)[-k]
    mask, taken = {}, 0
    for key in g:  # exactly k entries even with ties
        m = (g[key] >= thresh) & (taken < k)
        idx = np.flatnonzero(m.ravel())[: max(0, k - taken)]
        mm = np.zeros(g[key].size, dtype=bool)
        mm[idx] = True
        mask[key] = mm.reshape(g[key].shape)
        taken += len(idx)
    return mask


def finetune(w0, xs, ys, mask, steps=3000, lr=1e-2, batch=64, seed=0):
    """Adam on masked entries only; returns w0 with just the masked entries replaced."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    H, M = lstm.dims(w0)
    net = Net(H, M)
    net.load_weights(w0)
    tmask = {name: torch.tensor(mask[PMAP[name]], dtype=torch.float32) for name, _ in net.named_parameters()}
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Y = torch.tensor(ys, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        idx = rng.choice(len(xs), size=batch, replace=False)
        X, lens = _batch([xs[i] for i in idx])
        loss = loss_fn(net(X, lens), Y[idx])
        opt.zero_grad()
        loss.backward()
        for name, p in net.named_parameters():
            p.grad.mul_(tmask[name])  # zero grad => Adam moments stay 0 => entry never moves
        opt.step()
    trained = net.to_weights()
    out = {k: v.copy() for k, v in w0.items()}
    for k in out:
        out[k][mask[k]] = trained[k][mask[k]]
    return out


def llm_masks():
    """(init path, run name, edit limit, mask) for every DeepSeek fix run."""
    out = []
    for d in sorted(glob.glob(f"{tasks.runs_dir()}/agent/*_fix_*")):
        p = Path(d) / "result.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        w0, w1 = lstm.load(r["init"]), lstm.load(Path(d) / "weights.json")
        out.append((r["init"], Path(d).name, r["edit_limit"], {k: w0[k] != w1[k] for k in w0}))
    return out


def run_one(job):
    torch.set_num_threads(1)
    init, mask_name, mask, data, seed = job
    t0 = time.time()
    lo, hi, dseed = DATA[data]
    xs, ys = tasks.sample(8192, lo, hi, dseed)
    w0 = lstm.load(init)
    if mask is None:  # computed here so it uses this job's data
        H, M = lstm.dims(w0)
        net = Net(H, M)
        net.load_weights(w0)
        if mask_name == "full":
            mask = {k: np.ones_like(v, dtype=bool) for k, v in w0.items()}
        else:
            mask = topk_mask(grad_magnitude(net, xs, ys), int(mask_name[4:]))
    w = finetune(w0, xs, ys, mask, seed=seed)
    return {"method": "gradient-repair", "init": init, "mask": mask_name, "data": data, "seed": seed,
            "changed": int(sum(np.count_nonzero(w[k] != w0[k]) for k in w)),
            "allowed": int(sum(m.sum() for m in mask.values())),
            "init_scores": lstm.evaluate(w0), **lstm.evaluate(w), "secs": round(time.time() - t0, 1)}


def main():
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=tasks.name(), choices=sorted(tasks.TASKS))
    ap.add_argument("--inits", nargs="+", help="trained models to repair (default: the dyck1 three)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", help="default: <runs_dir>/finetune")
    a = ap.parse_args()
    tasks.set_task(a.task)
    a.out = a.out or f"{tasks.runs_dir()}/finetune"
    if a.inits is None:
        if a.task != "dyck1":
            ap.error("--inits is required for tasks other than dyck1")
        a.inits = ["runs/baselines/h4_n8192_lr0.003_st10000_s1.json",
                   "runs/baselines/h4_n8192_lr0.003_st10000_s0.json",
                   "runs/baselines/h4_n8192_lr0.01_st10000_s1.json"]

    jobs = []
    for init in a.inits:
        for s in a.seeds:
            for mname in ["full", "topk13", "topk6", "topk4"]:
                jobs.append((init, mname, None, "short", s))
            jobs.append((init, "full", None, "long", s))
    # gradient gets the exact entries the LLM chose (edit limit 13), both data regimes
    for init, run, limit, mask in llm_masks():
        if init.replace("\\", "/") in a.inits and limit == 13:
            for data in ("short", "long"):
                jobs.append((init, f"llm{limit}", mask, data, 0))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs on {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(run_one, jobs):
            print(json.dumps(res), flush=True)
            with open(out / "results.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
