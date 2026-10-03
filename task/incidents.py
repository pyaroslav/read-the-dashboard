# %%
"""Read the Dashboard, Part 2 — multi-panel incidents. Three stacked panels (latency, error rate, CPU) for one service on a
shared clock. Which signal moved first, in what order did the others follow, when did it start, which deploy lines up?"""
import os, glob, json, re, time
from dataclasses import dataclass, asdict
from typing import List
import pandas as pd
import kaggle_benchmarks as kbench
from kaggle_benchmarks.content_types import images

MAX_OUTPUT_TOKENS = 16384
NAMES = {"latency": "latency", "p99latency": "latency", "p99": "latency", "errors": "errors", "error": "errors", "errorrate": "errors",
         "saturation": "saturation", "cpu": "saturation", "cpuusage": "saturation"}

def find_data_dir() -> str:
    cands = [os.environ.get("INCIDENTS_DIR", ""), "data/incidents", "../data/incidents"] + [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/manifest.csv", recursive=True)]
    for c in cands:
        f = os.path.join(c, "manifest.csv")
        if c and os.path.exists(f) and "first_mover" in open(f).readline(): return c
    raise FileNotFoundError("incidents manifest not found; inputs: " + str(glob.glob("/kaggle/input/**", recursive=True)[:20]))

@dataclass
class IncidentReading:
    first_mover: str          # "latency", "errors" or "saturation" (CPU)
    order: List[str]          # every panel that changed, in the order it changed; leave out a panel that stayed flat
    start_time: str           # "HH:MM" when the first change begins
    implicated_deploy: str    # "A", "B", "C" or "none"
    dashed_marker_count: int  # canary

PROMPT = (
    "You are the on-call engineer. This image shows three stacked monitoring panels for one service on a shared clock: "
    "p99 latency (top), error rate (middle) and CPU (bottom). Dashed vertical lines are deploy markers.\n"
    "Report:\n- first_mover: which signal changed first: latency, errors or saturation (CPU).\n"
    "- order: every signal that changed, in the order it changed (2 or 3 items). Leave out a panel that stayed flat.\n"
    "- start_time: the 24-hour clock time (HH:MM, read off the x-axis) when the first change begins.\n"
    "- implicated_deploy: the letter of the deploy marker that coincides with the first change (within a minute or two), or 'none'.\n"
    "- dashed_marker_count: how many dashed vertical deploy markers there are.\nReport only what the panels show.")

def _img(file):
    p = os.path.join(find_data_dir(), file)
    if os.path.exists(p): return p
    hits = glob.glob(f"/kaggle/input/**/{file}", recursive=True); return hits[0] if hits else p

def _n(s):
    x = str(s).lower()                                   # tolerant: "saturation (CPU)", "CPU utilization", "error rate", "p99 latency"
    if "cpu" in x or "satur" in x: return "saturation"
    if "err" in x: return "errors"
    if "lat" in x or "p99" in x: return "latency"
    return re.sub(r"[^a-z0-9]", "", x)
def _hhmm(s):
    m = re.search(r"(\d{1,2}):(\d{2})", str(s)); return None if not m else (int(m.group(1)) % 24) * 60 + int(m.group(2))
def _dep(s):
    m = re.search(r"\b([ABC])\b", str(s).upper()); return m.group(1) if m else "NONE"

# %%
@kbench.task(name="read-the-dashboard-incidents",
             description="Part 2 of Read the Dashboard: three stacked panels (latency, errors, CPU) per incident. Which signal moved first, in what order did the others follow, when, and which deploy lines up? 120 synthetic incidents, deterministic grading.")
def incidents(llm, limit: int = 0) -> float:
    df = pd.read_csv(os.path.join(find_data_dir(), "manifest.csv"))
    if limit: df = df.head(limit)
    cols = ["id", "file", "first_mover", "order", "start_hhmm", "implicated_deploy", "dashed_marker_count"]
    os.environ["RENDER_SUBRUNS"] = "False"
    with kbench.client.enable_cache():
        runs = read_incident.evaluate(llm=[llm], evaluation_data=df[cols], n_jobs=4, timeout=240, on_failure="continue", max_attempts=1, remove_run_files=True)
    res = pd.DataFrame([r.result for r in runs.completed_runs]); n, done = len(df), len(res)
    m = lambda k: float(res[k].mean()) if done else 0.0
    summary = {"incidents": n, "completed": done, "accepted": int(res["accepted"].sum()) if done else 0, "composite": m("composite"),
               "first_acc": m("first_ok"), "order_acc": m("order_ok"), "start_acc": m("start_ok"), "deploy_acc": m("deploy_ok"), "canary_acc": m("canary_ok"),
               "cost_usd": float(res["cost_usd"].sum()) if done else 0.0}
    print("PANEL_ROWS_JSONL_BEGIN"); print(res.to_json(orient="records", lines=True)); print("PANEL_ROWS_JSONL_END"); print(json.dumps(summary, indent=1))
    kbench.assertions.assert_true(summary["accepted"] == n, expectation=f"Model accepted every image ({summary['accepted']}/{n})")
    kbench.assertions.assert_true(summary["first_acc"] >= 0.8, expectation=f"Names the first signal to move on ≥80% of incidents ({summary['first_acc']:.0%})")
    kbench.assertions.assert_true(summary["order_acc"] >= 0.8, expectation=f"Gets the full order of signals on ≥80% ({summary['order_acc']:.0%})")
    kbench.assertions.assert_true(summary["start_acc"] >= 0.8, expectation=f"Start time within ±3 min on ≥80% ({summary['start_acc']:.0%})")
    kbench.assertions.assert_true(summary["deploy_acc"] >= 0.8, expectation=f"Implicates the right deploy on ≥80% ({summary['deploy_acc']:.0%})")
    return summary["composite"]

# %%
@kbench.task(name="read-incident", store_task=False)
def read_incident(llm, id, file, first_mover, order, start_hhmm, implicated_deploy, dashed_marker_count) -> dict:
    out = {"id": id, "truth_first": first_mover, "truth_order": order, "truth_start": start_hhmm, "truth_deploy": implicated_deploy}
    t0 = time.time(); pred = None; err = None
    for attempt in range(4):
        try:
            with kbench.chats.new(f"inc-{id}-{attempt}") as chat:
                r = llm.prompt(PROMPT, image=images.from_path(_img(file)), schema=IncidentReading,
                               extra_api_params={"max_tokens": MAX_OUTPUT_TOKENS})
            pred = asdict(r); out.update(pred=json.dumps(pred), accepted=True, input_tokens=chat.usage.input_tokens,
                                         output_tokens=chat.usage.output_tokens, cost_usd=(chat.usage.total_cost_nanodollars or 0) / 1e9)
            break
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            if "exceeds your available quota" in err: out["quota_blocked"] = True
            if not any(k in err for k in ("429", "RateLimit", "500", "502", "503", "504", "imeout", "overloaded", "heavy load", "Connection")) or attempt == 3: break
            time.sleep(8 * (attempt + 1))
    out["latency_s"] = round(time.time() - t0, 2)
    if pred is None:
        out.update(pred=None, accepted=False, error=err, input_tokens=0, output_tokens=0, cost_usd=0.0)
        for k in ("first_ok", "order_ok", "start_ok", "deploy_ok", "canary_ok"): out[k] = False
        out["composite"] = 0.0; return out
    truth_order = json.loads(order)
    out["first_ok"] = _n(pred["first_mover"]) == _n(first_mover)
    out["order_ok"] = [_n(x) for x in (pred["order"] or [])] == [_n(x) for x in truth_order]
    ts, ps = _hhmm(start_hhmm), _hhmm(pred["start_time"]); out["start_ok"] = ps is not None and min(abs(ts - ps), 1440 - abs(ts - ps)) <= 3
    out["deploy_ok"] = _dep(pred["implicated_deploy"]) == _dep(implicated_deploy)
    try: out["canary_ok"] = int(pred["dashed_marker_count"]) == int(dashed_marker_count)
    except Exception: out["canary_ok"] = False
    out["composite"] = sum(out[k] for k in ("first_ok", "order_ok", "start_ok", "deploy_ok")) / 4.0
    return out

# %%
incidents.run(kbench.llm)
