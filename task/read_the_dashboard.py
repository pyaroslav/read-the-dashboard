# %%
"""Read the Dashboard — can a model read a monitoring panel the way an on-call engineer does?

Each panel is a synthetic Grafana-style chart with exact ground truth (event type, start time,
implicated deploy marker, recovery, peak value) plus two canary fields that reveal whether the
model can see the image at all. Panels + manifest.csv live in the attached dataset."""
import os, glob, json, re, time
from dataclasses import dataclass, asdict
import pandas as pd
import kaggle_benchmarks as kbench
from kaggle_benchmarks.content_types import images

# %%
MAX_OUTPUT_TOKENS = 16384   # per-panel output budget incl. hidden reasoning; above every model's observed p99, keeps the proxy's cost reservation small
EVENTS = ["step_change", "ramp", "spike", "flapping", "plateau", "gap", "none"]

def find_data_dir() -> str:
    cands = [os.environ.get("PANELS_DIR", "")] + sorted(glob.glob("/kaggle/input/*/")) + ["data/pilot", "../data/pilot", "../../data/pilot"]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "manifest.csv")):
            return c
    raise FileNotFoundError("manifest.csv not found; attach the panels dataset or set PANELS_DIR")

@dataclass
class Reading:
    event_type: str           # one of: step_change, ramp, spike, flapping, plateau, gap, none
    start_time: str           # "HH:MM" when the event begins, or "none"
    implicated_deploy: str    # "A", "B", "C" or "none" — the dashed deploy marker that coincides with the start
    recovering: bool          # has the metric returned to its earlier level by the right edge of the panel?
    peak_value: float         # the highest value of the affected series, in the units printed on the y-axis
    y_axis_label: str         # the exact y-axis label text
    dashed_marker_count: int  # how many dashed vertical deploy markers are drawn

PROMPT = (
    "You are the on-call engineer. This image is one monitoring panel from our dashboards.\n"
    "Look only at the FIRST series in the legend (the service named in the panel title) and report:\n"
    "- event_type, one of: step_change (an abrupt jump within a minute or two to a new level that holds for 15+ minutes; it may drop back later), "
    "ramp (a steady climb over 30+ minutes that never returns), spike (a burst shorter than 10 minutes that returns to the earlier level), "
    "flapping (repeated alternation between two levels from the start time onward), plateau (the series climbs for 10+ minutes and then "
    "flat-lines at a ceiling with no further movement), gap (a stretch of missing data where the line breaks), or none.\n"
    "- start_time: the 24-hour clock time (HH:MM exactly as printed on the x-axis) when the event begins, or 'none'.\n"
    "- implicated_deploy: the letter (A, B or C) of the dashed deploy marker that coincides with the start (within a minute or two), "
    "or 'none' if no marker coincides.\n"
    "- recovering: true only if the series is back at its earlier level by the right edge of the panel.\n"
    "- peak_value: the highest value reached by that series, as a number in the units printed on the y-axis label.\n"
    "- y_axis_label: the exact y-axis label text.\n"
    "- dashed_marker_count: how many dashed vertical marker lines are drawn.\n"
    "Report only what the panel shows."
)

def _norm(s): return re.sub(r"[^a-z0-9/%]", "", str(s).lower())
_UNIT_CLASS = {"ms": "ms", "millisecond": "ms", "milliseconds": "ms", "msec": "ms",
               "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
               "%": "%", "percent": "%", "pct": "%", "percentage": "%",
               "msgs": "msgs", "msg": "msgs", "messages": "msgs", "message": "msgs",
               "req/s": "req/s", "reqs": "req/s", "rps": "req/s", "requests/s": "req/s", "req/sec": "req/s", "requestspersecond": "req/s",
               "requests/sec": "req/s", "reqpersec": "req/s", "requests": "req/s", "req": "req/s", "r/s": "req/s"}
def _unit(s):
    k = _norm(s); return _UNIT_CLASS.get(k, k)   # canary compares unit *classes*, so "rps" == "req/s" (it detects blindness, not spelling)
def _hhmm(s):
    s = str(s); m = re.search(r"(\d{1,2}):(\d{2})", s)
    if not m: return None
    h, mi = int(m.group(1)), int(m.group(2))
    if re.search(r"p\.?m", s, re.I) and h < 12: h += 12
    if re.search(r"a\.?m", s, re.I) and h == 12: h = 0
    return (h % 24) * 60 + mi
def _deploy(s):
    s = str(s).strip().upper(); m = re.search(r"\b([ABC])\b", s)
    return m.group(1) if m else "NONE"
def _mins_apart(a, b):
    d = abs(a - b); return min(d, 1440 - d)

# %%
@kbench.task(name="read-the-dashboard",
             description="Can a model read a monitoring panel like an on-call engineer? 336 synthetic Grafana-style charts with exact ground truth: what happened, when, which deploy, is it recovering, how bad. Canary fields expose models that cannot see the image. Output budget: 16,384 tokens per panel. Answer key v3: every label is read from the rendered data (visible onset, visible recovery) and self-checked.")
def read_the_dashboard(llm, limit: int = 0) -> float:
    df = pd.read_csv(os.path.join(find_data_dir(), "manifest.csv"))
    if limit: df = df.head(limit)
    cols = ["id", "file", "event_type", "start_hhmm", "implicated_deploy", "recovering", "peak_value", "y_axis_label", "dashed_marker_count", "unit"]
    os.environ["RENDER_SUBRUNS"] = "False"
    with kbench.client.enable_cache():
        runs = read_panel.evaluate(llm=[llm], evaluation_data=df[cols], n_jobs=4, timeout=180,
                                   on_failure="continue", max_attempts=1, remove_run_files=True)
    rows = [r.result for r in runs.completed_runs]
    res = pd.DataFrame(rows)
    res.to_csv("read_the_dashboard_results.csv", index=False)
    n = len(df); done = len(res)
    m = lambda k: float(res[k].mean()) if done else 0.0
    summary = {"panels": n, "completed": done, "accepted": int(res["accepted"].sum()) if done else 0,
               "composite": m("composite"), "event_acc": m("event_ok"), "start_acc": m("start_ok"), "deploy_acc": m("deploy_ok"),
               "recovering_acc": m("recovering_ok"), "peak_acc": m("peak_ok"), "canary_acc": m("canary_ok"),
               "cost_usd": float(res["cost_usd"].sum()) if done else 0.0, "median_latency_s": float(res["latency_s"].median()) if done else 0.0}
    print("PANEL_ROWS_JSONL_BEGIN"); print(res.to_json(orient="records", lines=True)); print("PANEL_ROWS_JSONL_END")
    print(json.dumps(summary, indent=1))
    kbench.assertions.assert_true(summary["accepted"] == n, expectation=f"Model accepted the image for every panel ({summary['accepted']}/{n})")
    kbench.assertions.assert_true(summary["canary_acc"] >= 0.9, expectation=f"Can see the image: canary fields correct on ≥90% of panels ({summary['canary_acc']:.0%})")
    kbench.assertions.assert_true(summary["event_acc"] >= 0.8, expectation=f"Names the event correctly on ≥80% of panels ({summary['event_acc']:.0%})")
    kbench.assertions.assert_true(summary["start_acc"] >= 0.8, expectation=f"Start time within ±2 min on ≥80% of panels ({summary['start_acc']:.0%})")
    kbench.assertions.assert_true(summary["deploy_acc"] >= 0.8, expectation=f"Implicates the right deploy marker on ≥80% of panels ({summary['deploy_acc']:.0%})")
    kbench.assertions.assert_true(summary["peak_acc"] >= 0.8, expectation=f"Peak value within 10% on ≥80% of panels ({summary['peak_acc']:.0%})")
    return summary["composite"]

# %%
@kbench.task(name="read-panel", store_task=False)
def read_panel(llm, id, file, event_type, start_hhmm, implicated_deploy, recovering, peak_value, y_axis_label, dashed_marker_count, unit) -> dict:
    data_dir = find_data_dir()
    out = {"id": id, "truth_event": event_type, "truth_start": start_hhmm, "truth_deploy": implicated_deploy,
           "truth_recovering": bool(recovering), "truth_peak": float(peak_value)}
    t0 = time.time(); pred = None; last_err = None
    for attempt in range(4):                      # evaluate() retries are coerced to 1 on Kaggle -> retry transient errors here
        try:
            with kbench.chats.new(f"panel-{id}-{attempt}") as chat:
                r = llm.prompt(PROMPT, image=images.from_path(os.path.join(data_dir, file)), schema=Reading,
                               extra_api_params={"max_tokens": MAX_OUTPUT_TOKENS})
            pred = asdict(r); out["pred"] = json.dumps(pred); out["accepted"] = True; out["attempts"] = attempt + 1
            out["input_tokens"] = chat.usage.input_tokens; out["output_tokens"] = chat.usage.output_tokens
            out["cost_usd"] = (chat.usage.total_cost_nanodollars or 0) / 1e9
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            quota_block = "exceeds your available quota" in last_err
            if quota_block: out["quota_blocked"] = True
            transient = any(k in last_err for k in ("429", "RateLimit", "500", "502", "503", "504", "Timeout", "timeout", "overloaded", "heavy load", "Connection"))
            if not transient or attempt == 3: break
            time.sleep(8 * (attempt + 1))
    if pred is None:
        out.update(pred=None, accepted=False, error=last_err, attempts=4 if last_err and any(k in last_err for k in ("429", "RateLimit")) else 1, input_tokens=0, output_tokens=0, cost_usd=0.0)
    out["latency_s"] = round(time.time() - t0, 2)
    if pred is None:
        for k in ("event_ok", "start_ok", "deploy_ok", "recovering_ok", "peak_ok", "canary_ok"): out[k] = False
        out["peak_rel_err"] = None; out["composite"] = 0.0
        return out
    pe = _norm(pred["event_type"]); out["event_ok"] = (pe == _norm(event_type))
    ts, ps = _hhmm(start_hhmm), _hhmm(pred["start_time"])
    tol = 6 if _norm(event_type) == "ramp" else 2
    out["start_ok"] = (ts is None and ps is None) if ts is None else (ps is not None and _mins_apart(ts, ps) <= tol)
    out["deploy_ok"] = _deploy(pred["implicated_deploy"]) == _deploy(implicated_deploy)
    out["recovering_ok"] = bool(pred["recovering"]) == bool(recovering)
    try:
        rel = abs(float(pred["peak_value"]) - float(peak_value)) / max(abs(float(peak_value)), 1e-9)
    except Exception:
        rel = None
    out["peak_rel_err"] = rel; out["peak_ok"] = (rel is not None and rel <= 0.10)
    out["canary_ok"] = (_unit(pred["y_axis_label"]) == _unit(y_axis_label)) and (int(pred["dashed_marker_count"]) == int(dashed_marker_count))
    out["composite"] = sum(out[k] for k in ("event_ok", "start_ok", "deploy_ok", "recovering_ok", "peak_ok")) / 5.0
    return out

# %%
read_the_dashboard.run(kbench.llm)
