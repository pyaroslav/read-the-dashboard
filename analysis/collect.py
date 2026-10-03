"""Download every run of the task and rebuild per-panel rows from the notebooks' stream output.

Usage: .venv/bin/python analysis/collect.py [--no-download] [--version N]
Writes results/all_panels.csv (one row per panel × model × run) and results/summary.csv (one row per model × run).
"""
import argparse, glob, json, os, re, subprocess, sys
import pandas as pd

TASK = os.environ.get("KB_TASK", "read-the-dashboard"); OUT = "results/runs"; PFX = os.environ.get("KB_PREFIX", "")
BEGIN, END = "PANEL_ROWS_JSONL_BEGIN", "PANEL_ROWS_JSONL_END"

def download():
    subprocess.run([".venv/bin/kaggle", "b", "t", "download", TASK, "-o", OUT, "-s"], check=False)

def rows_from_notebook(nb_path):
    nb = json.load(open(nb_path)); text = []
    for c in nb.get("cells", []):
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                t = o.get("text", ""); text.append("".join(t) if isinstance(t, list) else t)
    blob = "\n".join(text)
    if BEGIN not in blob or END not in blob: return [], blob
    body = blob.split(BEGIN, 1)[1].split(END, 1)[0]
    rows = []
    for line in body.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
    return rows, blob

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-download", action="store_true"); ap.add_argument("--version", type=int, default=0)
    a = ap.parse_args()
    if not a.no_download: download()
    panels, summary = [], []
    for run_dir in sorted(glob.glob(f"{OUT}/{TASK}/*/*/*/")):
        _, _, version, model, run_id = run_dir.rstrip("/").split("/")[-5:]
        if a.version and int(version) != a.version: continue
        rj = glob.glob(os.path.join(run_dir, "*.run.json")); nbp = os.path.join(run_dir, "__notebook__.ipynb")
        state, score = None, None
        if rj:
            d = json.load(open(rj[0])); state = d.get("state"); res = d.get("results") or []
            for r in res:
                for k, v in r.items():
                    if k.lower().endswith("result"):
                        if isinstance(v, (int, float)): score = v
                        elif isinstance(v, dict) and isinstance(v.get("value"), (int, float)): score = v["value"]
        rows, blob = rows_from_notebook(nbp) if os.path.exists(nbp) else ([], "")
        for r in rows: r.update(model=model, version=int(version), run_id=run_id); panels.append(r)
        summary.append(dict(model=model, version=int(version), run_id=run_id, state=state, leaderboard_score=score,
                            panels=len(rows), composite=float(pd.DataFrame(rows)["composite"].mean()) if rows else None,
                            **({k: float(pd.DataFrame(rows)[k].mean()) for k in ("event_ok","first_ok","order_ok","start_ok","deploy_ok","recovering_ok","peak_ok","canary_ok","accepted") if k in pd.DataFrame(rows)} if rows else {}),
                            cost_usd=float(pd.DataFrame(rows)["cost_usd"].sum()) if rows else None,
                            median_latency_s=float(pd.DataFrame(rows)["latency_s"].median()) if rows else None))
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(panels).to_csv(f"results/{PFX}all_panels.csv", index=False)
    sm = pd.DataFrame(summary).sort_values(["version", "composite"], ascending=[True, False])
    sm.to_csv(f"results/{PFX}summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(sm.to_string(index=False))

if __name__ == "__main__":
    main()
