"""Keep-best selection for LLM fix runs, applied post hoc by replaying the transcript.

Every set_weights / edit_entries call in transcript.jsonl is replayed through the same Env
(same validation, same edit limit) to reconstruct each weight version the editor produced.
The selected version is the one with the highest PRACTICE accuracy (lengths 2-20 only, the same
information the editor and the gradient baselines' checkpoint selection use); ties go to the
latest version. The initial weights count as version 0, so selection can never be worse than
the starting model on short strings.

Run:  uv run python -m ws.keepbest           -> runs/agent/keepbest.jsonl
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from . import lstm, tasks
from .agent import Env

MUTATING = {"set_weights", "edit_entries"}


def replay(run_dir: Path):
    r = json.loads((run_dir / "result.json").read_text())
    w0 = lstm.load(r["init"])
    H, M = lstm.dims(w0)
    env = Env(lstm.validate(w0, H, M), H, M, r["budget"], r.get("edit_limit"))
    versions = [{k: v.copy() for k, v in env.w.items()}]
    for line in (run_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        for tc in rec.get("assistant", {}).get("tool_calls", []) or []:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                continue
            if not isinstance(args, dict):
                continue
            res = env.call(name, args)
            if name in MUTATING and res.get("ok"):
                versions.append({k: v.copy() for k, v in env.w.items()})
    final = lstm.load(run_dir / "weights.json")
    if not all(np.array_equal(final[k], env.w[k]) for k in final):
        raise RuntimeError(f"replay mismatch in {run_dir.name}")
    return r, versions


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=tasks.name(), choices=sorted(tasks.TASKS))
    tasks.set_task(ap.parse_args().task)
    R = tasks.runs_dir()
    xs, ys = tasks.split("practice")
    out = []
    for d in sorted(glob.glob(f"{R}/agent/*_fix_*")):
        d = Path(d)
        if not (d / "result.json").exists():
            continue
        r, versions = replay(d)
        accs = [lstm.accuracy(w, xs, ys) for w in versions]
        best = max(range(len(versions)), key=lambda i: (accs[i], i))  # ties -> latest
        w = versions[best]
        rec = {"run": r["run"], "init": r["init"], "edit_limit": r.get("edit_limit"),
               "budget": r["budget"], "model": r["model"], "n_versions": len(versions),
               "picked": best, "picked_is_final": best == len(versions) - 1,
               "picked_is_initial": best == 0, "practice": accs[best],
               "changed": int(sum(np.count_nonzero(w[k] != versions[0][k]) for k in w)),
               **lstm.evaluate(w), "submitted": {k: r[k] for k in ("test_id", "test_long", "test_xlong")}}
        out.append(rec)
        print(f"{d.name[:70]:70s} versions={len(versions):2d} picked={best:2d} "
              f"{'(final)' if rec['picked_is_final'] else '(initial)' if best == 0 else ''} "
              f"long {r['test_long']:.3f} -> {rec['test_long']:.3f}  id {r['test_id']:.3f} -> {rec['test_id']:.3f}")
    Path(f"{R}/agent/keepbest.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))


if __name__ == "__main__":
    main()
