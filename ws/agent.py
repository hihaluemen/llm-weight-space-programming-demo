"""LLM weight-space programming loop: DeepSeek reads/writes the LSTM's weights via tools.

The editor never trains: it has no code execution, no gradients, and no data beyond what the
budgeted tools return. Everything it can observe uses strings of length <= 20 (the same range
the gradient baselines train on), so any length generalisation must come from understanding.

Run:
  $env:DEEPSEEK_API_KEY = "sk-..."
  uv run python -m ws.agent --mode write --budget 10
  uv run python -m ws.agent --mode fix --init runs/baselines/h4_n512_s0.json --budget 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

from . import lstm, tasks

MAX_OBS_LEN = 20
MAX_TRACE_STRINGS = 4

SPEC = """\
You are programming the weights of a tiny LSTM classifier BY HAND. There is no training:
you cannot run code, compute gradients, or see a dataset. You decide the numbers.

## Task
{TASK}
Evaluation uses held-out strings of MANY lengths, including lengths far longer than anything
you can observe (up to 500 characters). Your network must stay correct at any length.

## Architecture (exact float64 math; hidden size H={H}, head size M={M}; {P} parameters)
Each character is one-hot: '{C0}' -> x=[1,0], '{C1}' -> x=[0,1]. h_0 = c_0 = 0. For each step t:
    z_t = W_ih @ x_t + W_hh @ h_(t-1) + b                     # shape [4H]
    i = sigmoid(z[0:H])   f = sigmoid(z[H:2H])   g = tanh(z[2H:3H])   o = sigmoid(z[3H:4H])
    c_t = f * c_(t-1) + i * g
    h_t = o * tanh(c_t)
After the last character (h_T = final hidden state):
    a = relu(W1 @ h_T + b1)                                   # shape [M]
    logit = W2 @ a + b2                                       # scalar; accept iff logit > 0
Tensor shapes: W_ih [4H,2], W_hh [4H,H], b [4H], W1 [M,H], b1 [M], W2 [1,M], b2 [1].
Row blocks of W_ih / W_hh / b are ordered [input gate i | forget gate f | cell candidate g | output gate o].
Column 0 of W_ih is '{C0}' and column 1 is '{C1}'.

## Tools
- get_weights / set_weights / edit_entries: free. set_weights takes any subset of tensors as
  nested lists (full tensors, exact shapes) and replaces them. edit_entries changes individual
  entries, e.g. {{"tensor": "b", "index": [5], "value": 20}} or {{"tensor": "W_hh", "index": [9, 0], "value": -3.5}}.
- evaluate: COSTS 1 forward-pass budget. Accuracy of the current weights on a practice set
  (512 strings, lengths 2-20) plus some misclassified examples.
- trace: COSTS 1 forward-pass budget. Runs up to {NT} strings of your choice (each at most
  {L} characters) and returns every gate / cell / hidden value per step and the final logit.
- submit: finish. Your current weights are then scored on the held-out test sets.
You have a forward-pass budget of {B}. When it is used up, evaluate and trace stop working.
{MODE}
Think carefully about what computation solves the task for ALL lengths and how to express it in
these weights. Large pre-activations saturate sigmoid/tanh to exactly 0/1/+-1, which is useful.
Call submit when you are done.
"""

MODE_WRITE = "\n## Starting point\nAll weights are currently zero. Write the network from scratch.\n"
MODE_FIX = """
## Starting point
The current weights were produced by gradient descent on strings of length <= 20. The network
is imperfect, especially on long strings. Repair or rewrite it so it is correct at all lengths.
Current weights:
{W}
"""
MODE_FIX_LIMITED = """
## Starting point
The current weights were produced by gradient descent on strings of length <= 20. The network
is imperfect, especially on long strings. Repair it so it is correct at all lengths.
EDIT LIMIT: you may change at most {K} of the {P} parameters. A parameter counts as changed if
it differs from its initial value below (copying back a printed 4-decimal value unchanged does
not count). Any edit that would exceed the limit is rejected and nothing is applied. So you
cannot rewrite the network: first understand what the trained weights already compute, then
make targeted edits. edit_entries is the convenient tool for this.
Initial weights:
{W}
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weights", "description": "Return the current weights (free).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "set_weights",
        "description": "Replace one or more tensors (free). Each value must be a full tensor "
                       "as nested lists with the exact shape.",
        "parameters": {"type": "object", "properties": {
            "weights": {"type": "object", "description":
                        "Map tensor name (W_ih, W_hh, b, W1, b1, W2, b2) to nested list of numbers."}},
            "required": ["weights"]}}},
    {"type": "function", "function": {
        "name": "edit_entries",
        "description": "Set individual weight entries (free). Applied atomically.",
        "parameters": {"type": "object", "properties": {
            "edits": {"type": "array", "items": {"type": "object", "properties": {
                "tensor": {"type": "string", "description": "W_ih, W_hh, b, W1, b1, W2 or b2"},
                "index": {"type": "array", "items": {"type": "integer"},
                          "description": "[row, col] for matrices, [i] for vectors"},
                "value": {"type": "number"}},
                "required": ["tensor", "index", "value"]}}},
            "required": ["edits"]}}},
    {"type": "function", "function": {
        "name": "evaluate",
        "description": "Practice-set accuracy of the current weights. Costs 1 forward pass.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "trace",
        "description": f"Per-step gates/cell/hidden values for up to {MAX_TRACE_STRINGS} strings "
                       f"(each <= {MAX_OBS_LEN} chars of '{{C0}}' and '{{C1}}'). Costs 1 forward pass.",
        "parameters": {"type": "object", "properties": {
            "strings": {"type": "array", "items": {"type": "string"}}},
            "required": ["strings"]}}},
    {"type": "function", "function": {
        "name": "submit", "description": "Finish and submit the current weights.",
        "parameters": {"type": "object", "properties": {}}}},
]


def tools_for_task() -> list:
    """TOOLS with the active task's alphabet filled into the trace description."""
    a = tasks.alphabet()
    t = json.loads(json.dumps(TOOLS))
    for tool in t:
        f = tool["function"]
        f["description"] = f["description"].replace("{C0}", a[0]).replace("{C1}", a[1])
    return t


def _r(a, n=3):
    return np.round(np.asarray(a, dtype=float), n).tolist()


SNAP_TOL = 1e-4  # values within this of the initial value are snapped back (4-decimal printing)


class Env:
    def __init__(self, w, hidden, head, budget, max_changed=None):
        self.w, self.H, self.M = w, hidden, head
        self.w0 = {k: v.copy() for k, v in w.items()}
        self.max_changed = max_changed
        self.budget = budget
        self.used = 0
        self.done = False
        self.practice = tasks.split("practice")

    def n_changed(self, w=None) -> int:
        w = self.w if w is None else w
        return int(sum(np.count_nonzero(w[k] != self.w0[k]) for k in w))

    def _apply(self, new: dict):
        new = lstm.validate(new, self.H, self.M)
        for k in new:  # undo rounding noise from copying printed values back
            near = np.abs(new[k] - self.w0[k]) <= SNAP_TOL
            new[k][near] = self.w0[k][near]
        k = self.n_changed(new)
        if self.max_changed is not None and k > self.max_changed:
            raise ValueError(f"rejected: this would change {k} parameters but the limit is "
                             f"{self.max_changed}. Nothing was applied.")
        self.w = new
        return {"ok": True, "changed_vs_initial": k,
                **({"edit_limit": self.max_changed} if self.max_changed is not None else {})}

    def _spend(self):
        if self.used >= self.budget:
            raise ValueError(f"forward-pass budget exhausted ({self.budget} used). "
                             "Use get_weights/set_weights and submit.")
        self.used += 1

    def call(self, name: str, args: dict) -> dict:
        try:
            return getattr(self, "t_" + name)(**args)
        except (ValueError, TypeError) as e:
            return {"error": str(e)}

    def t_get_weights(self):
        out = {k: _r(v, 4) for k, v in self.w.items()}
        if self.max_changed is not None:
            out["changed_vs_initial"] = self.n_changed()
            out["edit_limit"] = self.max_changed
        return out

    def t_set_weights(self, weights: dict):
        if not isinstance(weights, dict) or not weights:
            raise ValueError("weights must be a non-empty object")
        return self._apply({**self.w, **weights})

    def t_edit_entries(self, edits: list):
        if not isinstance(edits, list) or not edits:
            raise ValueError("edits must be a non-empty list")
        new = {k: v.copy() for k, v in self.w.items()}
        for e in edits:
            t, idx, val = e.get("tensor"), e.get("index"), e.get("value")
            if t not in new:
                raise ValueError(f"unknown tensor {t!r}; nothing applied")
            if not isinstance(idx, list) or len(idx) != new[t].ndim:
                raise ValueError(f"{t} needs an index of length {new[t].ndim}; nothing applied")
            if any(not isinstance(i, int) or not 0 <= i < n for i, n in zip(idx, new[t].shape)):
                raise ValueError(f"index {idx} out of range for {t} with shape "
                                 f"{list(new[t].shape)}; nothing applied")
            new[t][tuple(idx)] = float(val)
        return self._apply(new)

    def t_evaluate(self):
        self._spend()
        xs, ys = self.practice
        logits = lstm.forward_batch(self.w, xs)
        pred = (logits > 0).astype(int)
        lens = np.array([len(s) for s in xs])
        kinds = np.array([tasks.kind(s) for s in xs])
        order = sorted(map(str, dict.fromkeys(kinds)), key=lambda k: "should accept" not in k)  # accept first
        groups = {k: kinds == k for k in order}
        wrong = np.flatnonzero(pred != ys)[:8]
        return {
            "budget_left": self.budget - self.used,
            "accuracy": round(float(np.mean(pred == ys)), 4),
            # 9-14 / 15-20 (not 10-14 / 16-20) so a2b's lengths 9 and 15 are covered; same for dyck1
            "by_length": {f"{lo}-{hi}": round(float(np.mean((pred == ys)[(lens >= lo) & (lens <= hi)])), 4)
                          for lo, hi in [(2, 8), (9, 14), (15, 20)]},
            "by_type": {k: round(float(np.mean((pred == ys)[m])), 4) for k, m in groups.items() if m.any()},
            "misclassified_examples": [{"string": xs[i], "label": "accept" if ys[i] else "reject",
                                        "logit": round(float(logits[i]), 3)} for i in wrong],
        }

    def t_trace(self, strings: list):
        if not isinstance(strings, list) or not 1 <= len(strings) <= MAX_TRACE_STRINGS:
            raise ValueError(f"give 1-{MAX_TRACE_STRINGS} strings")
        alpha = tasks.alphabet()
        for s in strings:
            if not isinstance(s, str) or not s or len(s) > MAX_OBS_LEN or set(s) - set(alpha):
                raise ValueError(f"each string must be 1-{MAX_OBS_LEN} chars of '{alpha[0]}' and "
                                 f"'{alpha[1]}': {s!r}")
        self._spend()
        out = []
        for s in strings:
            logit, steps, a = lstm.forward(self.w, s, trace=True)
            out.append({
                "string": s, "true_label": "accept" if tasks.label(s) else "reject",
                "steps": [{"char": ch, **{k: _r(v) for k, v in st.items()}} for ch, st in zip(s, steps)],
                "head_relu": _r(a), "logit": round(logit, 4),
                "prediction": "accept" if logit > 0 else "reject"})
        return {"budget_left": self.budget - self.used, "traces": out}

    def t_submit(self):
        self.done = True
        return {"ok": True}


def load_dotenv(path=Path(__file__).resolve().parent.parent / ".env"):
    """Minimal .env loader (KEY=VALUE lines); real environment variables take precedence."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()


def chat_with_retry(client, retries=6, **kw):
    """Retry transient network/server errors with exponential backoff (2, 4, 8, ... s)."""
    import openai

    for attempt in range(retries + 1):
        try:
            return client.chat.completions.create(**kw)
        except (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError,
                openai.InternalServerError) as e:
            if attempt == retries:
                raise
            wait = 2 ** (attempt + 1)
            print(f"[retry {attempt + 1}/{retries} in {wait}s] {type(e).__name__}: {e}", flush=True)
            time.sleep(wait)


def run(mode, budget, hidden, head, init, model, max_turns, out_root, tag="", max_changed_frac=None):
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    w0 = lstm.load(init) if init else lstm.zeros(hidden, head)
    if init:
        hidden, head = lstm.dims(w0)
    P = lstm.n_params(w0)
    max_changed = None
    if mode == "fix" and max_changed_frac is not None:
        max_changed = max(1, int(np.floor(max_changed_frac * P)))
    env = Env(lstm.validate(w0, hidden, head), hidden, head, budget, max_changed)
    if mode == "write":
        mode_txt = MODE_WRITE
    elif max_changed is None:
        mode_txt = MODE_FIX.format(W=lstm.to_json(w0, 4))
    else:
        mode_txt = MODE_FIX_LIMITED.format(K=max_changed, P=P, W=lstm.to_json(w0, 4))
    alpha = tasks.alphabet()
    system = SPEC.format(H=hidden, M=head, P=P, NT=MAX_TRACE_STRINGS, L=MAX_OBS_LEN, B=budget,
                         MODE=mode_txt, TASK=tasks.current()["describe"], C0=alpha[0], C1=alpha[1])
    tools = tools_for_task()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Begin. Use the tools; call submit when finished."}]

    lim = f"_k{max_changed}" if max_changed is not None else ""
    src = f"_{Path(init).stem}" if init else ""
    # pid + random suffix: parallel processes starting in the same second must not share a dir
    run_id = (time.strftime("%Y%m%d-%H%M%S") + f"_{mode}{src}_b{budget}{lim}"
              + (f"_{tag}" if tag else "") + f"_{os.getpid()}{os.urandom(2).hex()}")
    out = Path(out_root) / run_id
    out.mkdir(parents=True, exist_ok=False)
    log = open(out / "transcript.jsonl", "w", encoding="utf-8")
    log.write(json.dumps({"system": system}) + "\n")
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    nudges = 0

    for turn in range(max_turns):
        resp = chat_with_retry(client, model=model, messages=messages, tools=tools)
        if resp.usage:
            usage["prompt_tokens"] += resp.usage.prompt_tokens
            usage["completion_tokens"] += resp.usage.completion_tokens
        msg = resp.choices[0].message
        am = {"role": "assistant", "content": msg.content or ""}
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:  # thinking mode: pass reasoning back within the same task
            am["reasoning_content"] = reasoning
        if msg.tool_calls:
            am["tool_calls"] = [{"id": tc.id, "type": "function",
                                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                for tc in msg.tool_calls]
        messages.append(am)
        log.write(json.dumps({"turn": turn, "assistant": am}, ensure_ascii=False) + "\n")
        print(f"[turn {turn}] {(msg.content or '')[:200]!r} tools={[tc.function.name for tc in msg.tool_calls or []]}")

        if not msg.tool_calls:
            nudges += 1
            if nudges > 3:
                break
            messages.append({"role": "user", "content": "Please act through the tools (set_weights, "
                             "evaluate, trace) and call submit when finished."})
            continue
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = env.call(tc.function.name, args) if isinstance(args, dict) else {"error": "bad args"}
            except json.JSONDecodeError as e:
                result = {"error": f"arguments are not valid JSON: {e}"}
            content = json.dumps(result)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
            log.write(json.dumps({"turn": turn, "tool": tc.function.name, "result": result}) + "\n")
        if env.done:
            break

    lstm.save(env.w, out / "weights.json")
    res = {"run": run_id, "task": tasks.name(), "method": "llm", "mode": mode, "model": model, "budget": budget,
           "passes_used": env.used, "submitted": env.done, "turns": turn + 1, "init": init,
           "params": P, "edit_limit": max_changed, "changed": env.n_changed(),
           "practice": lstm.accuracy(env.w, *env.practice), **lstm.evaluate(env.w), **usage}
    if init:
        res["init_scores"] = lstm.evaluate(w0)
    (out / "result.json").write_text(json.dumps(res, indent=2))
    with open(Path(out_root) / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    log.close()
    print(json.dumps(res, indent=2))
    return res


def main():
    import sys

    # Windows consoles default to GBK; model output can contain e.g. subscripts (c₁)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["write", "fix"], default="write")
    ap.add_argument("--budget", type=int, default=10, help="forward passes (evaluate/trace calls)")
    ap.add_argument("--hidden", type=int, default=4)
    ap.add_argument("--head", type=int, default=4)
    ap.add_argument("--init", help="weights json to start from (fix mode)")
    ap.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-changed-frac", type=float,
                    help="fix mode: max fraction of parameters that may differ from --init (e.g. 0.1)")
    ap.add_argument("--task", default=tasks.name(), choices=sorted(tasks.TASKS))
    ap.add_argument("--out", help="default: <runs_dir>/agent")
    a = ap.parse_args()
    tasks.set_task(a.task)
    a.out = a.out or f"{tasks.runs_dir()}/agent"
    if a.mode == "fix" and not a.init:
        ap.error("--mode fix needs --init")
    for r in range(a.repeats):
        run(a.mode, a.budget, a.hidden, a.head, a.init, a.model, a.max_turns, a.out,
            tag=f"r{r}", max_changed_frac=a.max_changed_frac)


if __name__ == "__main__":
    main()
