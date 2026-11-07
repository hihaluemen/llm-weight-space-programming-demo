"""Gradient-trained baselines: same architecture, trained with Adam on n examples (len <= 20).

Run:  uv run python -m ws.train --n-train 32 128 512 2048 8192 --seeds 0 1 2
Writes runs/baselines/h{H}_n{n}_s{seed}.json (weights) and runs/baselines/results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from . import lstm, tasks


class Net(nn.Module):
    def __init__(self, hidden: int, head: int):
        super().__init__()
        self.rnn = nn.LSTM(lstm.IN, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden, head)
        self.fc2 = nn.Linear(head, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(packed)
        return self.fc2(torch.relu(self.fc1(h[-1])))[:, 0]

    def to_weights(self) -> dict[str, np.ndarray]:
        r = self.rnn
        g = lambda t: t.detach().double().numpy().copy()  # noqa: E731
        return {
            "W_ih": g(r.weight_ih_l0),
            "W_hh": g(r.weight_hh_l0),
            "b": g(r.bias_ih_l0 + r.bias_hh_l0),
            "W1": g(self.fc1.weight),
            "b1": g(self.fc1.bias),
            "W2": g(self.fc2.weight),
            "b2": g(self.fc2.bias),
        }

    def load_weights(self, w: dict[str, np.ndarray]):
        r = self.rnn
        t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
        with torch.no_grad():
            r.weight_ih_l0.copy_(t(w["W_ih"]))
            r.weight_hh_l0.copy_(t(w["W_hh"]))
            r.bias_ih_l0.copy_(t(w["b"]))
            r.bias_hh_l0.zero_()
            self.fc1.weight.copy_(t(w["W1"]))
            self.fc1.bias.copy_(t(w["b1"]))
            self.fc2.weight.copy_(t(w["W2"]))
            self.fc2.bias.copy_(t(w["b2"]))


def _batch(xs: list[str]):
    L = max(len(s) for s in xs)
    X = np.zeros((len(xs), L, lstm.IN), dtype=np.float32)
    for i, s in enumerate(xs):
        X[i, : len(s)] = tasks.one_hot(s)
    return torch.from_numpy(X), torch.tensor([len(s) for s in xs])


def train(xs, ys, hidden=4, head=4, steps=4000, lr=1e-2, batch=64, seed=0, init=None,
          val=None, eval_every=500):
    """Adam + cosine LR. If `val` (xs, ys) is given, return the checkpoint with best val accuracy
    (val uses lengths <= 20 only, so selection never peeks at long strings)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(hidden, head)
    if init is not None:
        net.load_weights(init)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Y = torch.tensor(ys, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_acc = None, -1.0
    for step in range(1, steps + 1):
        idx = rng.choice(len(xs), size=min(batch, len(xs)), replace=False)
        X, lens = _batch([xs[i] for i in idx])
        loss = loss_fn(net(X, lens), Y[idx])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if val is not None and (step % eval_every == 0 or step == steps):
            w = net.to_weights()
            acc = lstm.accuracy(w, *val)
            if acc > best_acc:  # strict: earliest checkpoint wins ties
                best, best_acc = w, acc
    return (best, best_acc) if val is not None else net.to_weights()


def run_one(job):
    torch.set_num_threads(1)
    hidden, head, n, lr, steps, seed, out = job
    t0 = time.time()
    # training data: lengths 2-20, seeds disjoint from practice/val/test splits
    xs, ys = tasks.sample(n, 2, 20, seed=10_000 * (seed + 1) + n)
    w, val_acc = train(xs, ys, hidden, head, steps, lr, seed=seed, val=tasks.split("val"))
    name = f"h{hidden}_n{n}_lr{lr:g}_st{steps}_s{seed}"
    lstm.save(w, Path(out) / f"{name}.json")
    return {"name": name, "task": tasks.name(), "method": "gradient", "hidden": hidden, "n_train": n, "lr": lr,
            "steps": steps, "seed": seed, "params": lstm.n_params(w),
            "train_acc": lstm.accuracy(w, xs, ys), "val": val_acc, **lstm.evaluate(w),
            "secs": round(time.time() - t0, 1)}


def main():
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, nargs="+", default=[512, 8192])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--head", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, nargs="+", default=[3e-3, 1e-2])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--task", default=tasks.name(), choices=sorted(tasks.TASKS))
    ap.add_argument("--out", help="default: <runs_dir>/baselines")
    a = ap.parse_args()
    tasks.set_task(a.task)  # env var -> inherited by the worker processes
    a.out = a.out or f"{tasks.runs_dir()}/baselines"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(h, a.head, n, lr, a.steps, s, str(out))
            for h in a.hidden for n in a.n_train for lr in a.lr for s in a.seeds]
    print(f"{len(jobs)} jobs on {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(run_one, jobs):
            print(json.dumps(res), flush=True)
            with open(out / "results.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
