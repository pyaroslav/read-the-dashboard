"""Quota-aware orchestrator for task v6.

Loop: probe quota -> (push v6 once) -> launch a small batch of models -> wait -> collect ->
mark each model valid (>=98% panels accepted), not_served (proxy 404), or retry -> repeat until the queue is empty.
Usage: nohup .venv/bin/python analysis/orchestrate.py > results/orchestrate.log 2>&1 &
State: results/orchestrate_state.json
"""
import json, os, subprocess, sys, time, datetime as dt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
K = ".venv/bin/kaggle"; TASK = os.environ.get("KB_TASK", "read-the-dashboard"); VERSION = int(os.environ.get("KB_VERSION", "7"))
STATE = os.environ.get("KB_STATE", "results/orchestrate_state_v7.json"); TASK_FILE = os.environ.get("KB_FILE", "task/read_the_dashboard.py"); DATASET = os.environ.get("KB_DATASET", "yaroslavperventsev/read-the-dashboard-panels"); PFX = os.environ.get("KB_PREFIX", "")
CHEAP = ["gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite-preview",
         "gemini-3-flash-preview", "gemini-2.5-flash", "gemma-4-26b-a4b-it", "gemma-4-31b-it", "claude-haiku-4-5-20251001",
         "gpt-5.4-nano-2026-03-17", "gpt-5.4-mini-2026-03-17", "grok-4.20-0309-non-reasoning", "grok-4.20-0309-reasoning"]
MID = ["claude-sonnet-4-5-20250929", "claude-sonnet-4-6-default", "claude-sonnet-5-default", "gpt-5.4-2026-03-05", "gpt-5.5-2026-04-23", "gpt-5.6-luna", "gpt-5.6-terra"]
PREMIUM = ["claude-opus-5-default", "gemini-3.1-pro-preview", "gemini-2.5-pro", "gpt-6-astra", "claude-opus-4-8-default", "claude-opus-4-7-default",
           "claude-opus-4-6-default", "claude-opus-4-5-20251101", "claude-opus-4-1-20250805"]
BLIND = ["qwen3-235b-a22b-instruct-2507", "gpt-oss-120b", "glm-5", "deepseek-r1-0528"]
BATCH = {**{m: 2 for m in CHEAP + BLIND}, **{m: 2 for m in MID}, **{m: 1 for m in PREMIUM}}
MAX_TRIES = 3

def log(*a): print(dt.datetime.now(dt.UTC).strftime("%m-%d %H:%M:%S"), *a, flush=True)
def sh(args, timeout=600):
    try: return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired: return ""
def env():
    sh([K, "b", "auth", "-y", "--env-file", "./.env"], 120)
    for line in open(".env"):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1); os.environ[k] = v

def quota_ok():
    """A real (tiny) call with a 16k cap: succeeds only if the proxy will reserve ~$0.2 for us."""
    env()
    code = ("import os,kaggle_benchmarks as kb\n"
            "names=[n for n in os.environ['LLMS_AVAILABLE'].split(',') if 'claude' in n or 'gemini-3-flash' in n]\n"
            "llm=kb.llms[names[0]]\n"
            "with kb.chats.new('probe'):\n  llm.prompt('Reply with OK.', extra_api_params={'max_tokens':65536})\n"
            "print('QUOTA_OK')")
    out = subprocess.run([".venv/bin/python", "-c", code], capture_output=True, text=True, timeout=180)
    ok = "QUOTA_OK" in out.stdout
    if not ok: log("quota probe blocked:", (out.stdout + out.stderr).strip().splitlines()[-1][:160] if (out.stdout + out.stderr).strip() else "?")
    return ok

def status_busy():
    s = sh([K, "b", "t", "status", TASK], 90)
    return any(w in s for w in ("Running", "Queued", "Scheduled", "Pending"))

def current_version():
    for line in sh([K, "b", "t", "status", TASK], 90).splitlines():
        if line.startswith("Version:"): return int(line.split()[1])
    return 0

def collect():
    sh([K, "b", "t", "download", TASK, "-o", "results/runs", "-s"], 900)
    subprocess.run([".venv/bin/python", "analysis/collect.py", "--no-download", "--version", str(VERSION)], capture_output=True, text=True, env={**os.environ})
    try: d = pd.read_csv(f"results/{PFX}all_panels.csv", low_memory=False); d = d[d.version == VERSION]
    except Exception: return {}
    res = {}
    for m, g in d.groupby("model"):
        latest = g[g.run_id == g.run_id.max()]
        acc = latest.accepted.astype(bool).mean(); err = latest["error"].dropna().astype(str) if "error" in latest.columns else pd.Series([], dtype=str)
        if len(latest) == int(os.environ.get("KB_N", "336")) and acc >= 0.98: res[m] = "valid"
        elif len(err) and err.str.contains("404").mean() > 0.9: res[m] = "not_served"
        elif len(err) and err.str.contains("MIME type|not a multimodal", regex=True).mean() > 0.9: res[m] = "no_vision"
        elif len(err) and err.str.contains("exceeds your available quota").sum() > 0.3 * len(latest): res[m] = "quota"
        else: res[m] = f"retry(acc={acc:.2f},n={len(latest)})"
    return res

def main():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"pushed": False, "tries": {}, "final": {}}
    if st["pushed"]:                                   # resume: bank any valid runs already on this version
        while status_busy(): time.sleep(45)
        for m, r in collect().items():
            if r in ("valid", "not_served", "no_vision") and m not in st["final"]: st["final"][m] = r; log("resume:", m, "->", r)
        json.dump(st, open(STATE, "w"), indent=1)
    queue = [m for m in CHEAP + MID + PREMIUM + BLIND if m not in st["final"]]
    log("start; queue", len(queue))
    while queue:
        while not quota_ok(): time.sleep(900)
        if not st["pushed"]:
            log(f"push v{VERSION}"); out = sh([K, "b", "t", "push", TASK, "-f", TASK_FILE, "-d", DATASET, "--wait"], 3600)
            for _ in range(120):                                   # wait for the push's validation run to finish
                stt = sh([K, "b", "t", "status", TASK], 90)
                if not any(w in stt.split("Created")[0] for w in ("Running", "Queued", "RUNNING", "QUEUED")): break
                time.sleep(30)
            v = current_version(); head = sh([K, "b", "t", "status", TASK], 90).split("Created")[0]
            log("version now", v, "|", " ".join(head.split())[:160])
            if v != VERSION or "Completed" not in head: log("unexpected version or validation not completed; stopping"); return
            st["pushed"] = True; json.dump(st, open(STATE, "w"), indent=1); continue
        quota_hit = False
        bs = BATCH[queue[0]]; batch = [m for m in queue if BATCH[m] == bs][:bs] or queue[:1]
        log("launch", batch)
        args = [K, "b", "t", "run", TASK]; [args.extend(["-m", m]) for m in batch]; log(sh(args, 180).strip().replace("\n", " | ")[:300])
        time.sleep(60)
        while status_busy(): time.sleep(45)
        res = collect()
        for m in batch:
            r = res.get(m, "retry(no rows)"); log(" ", m, "->", r)
            if r == "quota": quota_hit = True; continue   # quota refusals don't count as a try
            st["tries"][m] = st["tries"].get(m, 0) + 1
            if not r.startswith("retry") or st["tries"][m] >= MAX_TRIES: st["final"][m] = r
        json.dump(st, open(STATE, "w"), indent=1)
        queue = [m for m in CHEAP + MID + PREMIUM + BLIND if m not in st["final"]]
        log("queue left", len(queue))
        if quota_hit: log("quota block in last batch; sleeping 60 min before probing"); time.sleep(3600)
    log("DONE", json.dumps(st["final"]))
    if not PFX: subprocess.run([".venv/bin/python", "analysis/report.py", "--version", str(VERSION)])

if __name__ == "__main__":
    main()
