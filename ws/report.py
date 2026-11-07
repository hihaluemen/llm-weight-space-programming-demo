"""Aggregate baselines and LLM runs into one table (mean ± std over seeds/repeats).

Run:  uv run python -m ws.report [--task a2b]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import tasks

COLS = ["test_id", "test_long", "test_xlong"]


def load(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=tasks.name(), choices=sorted(tasks.TASKS))
    a = ap.parse_args()
    tasks.set_task(a.task)
    R = tasks.runs_dir()

    groups = defaultdict(list)
    for r in load(f"{R}/baselines/results.jsonl"):
        key = f"gradient h{r['hidden']} n={r['n_train']} lr={r['lr']:g} steps={r['steps']}"
        groups[(0, r["hidden"], r["n_train"], key)].append(r)
    for r in load(f"{R}/agent/results.jsonl"):
        if r["mode"] == "write":
            key = f"{r['model']} write(zeros) budget={r['budget']}"
        else:  # pooled over the repaired models
            lim = r.get("edit_limit") or 999
            key = f"{r['model']} fix budget={r['budget']} edit<={lim if lim != 999 else 'all'}"
            groups[(2, -lim, 0, key)].append(r)
            continue
        groups[(1, 0, 0, key)].append(r)
    for r in load(f"{R}/agent/keepbest.jsonl"):  # same runs, version picked by practice accuracy
        key = f"{r['model']} fix budget={r['budget']} edit<={r['edit_limit']} keep-best"
        groups[(2, -r["edit_limit"] - 0.5, 0, key)].append(r)
    for r in load(f"{R}/finetune/results.jsonl"):  # pooled over the repaired models
        order = {"full": 999, "llm13": 13.5}.get(r["mask"], int(r["mask"][4:]) if r["mask"].startswith("topk") else 0)
        ref = "  [REFERENCE: sees long data]" if r["data"] == "long" else ""
        key = f"gradient-repair mask={r['mask']} data={r['data']}{ref}"
        groups[(3 + (r["data"] == "long"), -order, 0, key)].append(r)

    seen = {}
    for r in load(f"{R}/agent/results.jsonl"):
        if r["mode"] == "fix":
            seen[r["init"]] = r["init_scores"]
    if seen:
        groups[(2, -10_000, 0, f"before repair ({len(seen)} trained h4 models)")] = list(seen.values())

    print(f"task: {a.task}")
    w = max(len(k[3]) for k in groups) if groups else 10
    print(f"{'method':{w}s} {'k':>2s} " + " ".join(f"{c:>15s}" for c in COLS))
    for (*_, name), rs in sorted(groups.items()):
        cells = []
        for c in COLS:
            v = np.array([r[c] for r in rs])
            cells.append(f"{v.mean():.3f}±{v.std():.3f}")
        print(f"{name:{w}s} {len(rs):2d} " + " ".join(f"{c:>15s}" for c in cells))


if __name__ == "__main__":
    main()
