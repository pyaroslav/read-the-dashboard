"""Summary tables + charts from results/all_panels.csv.  Usage: .venv/bin/python analysis/report.py [--version N]"""
import argparse, json, math, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FIELDS = ["event_ok", "start_ok", "deploy_ok", "recovering_ok", "peak_ok"]
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--version", type=int, default=0); ap.add_argument("--manifest", default="data/full/manifest.csv"); a = ap.parse_args()
    df = pd.read_csv("results/all_panels.csv", low_memory=False)
    if a.version: df = df[df.version == a.version]
    df = df[df.run_id == df.groupby("model").run_id.transform("max")]            # latest run per model
    accm = df.groupby("model").accepted.apply(lambda s: s.astype(bool).mean())
    canm = df.groupby("model").canary_ok.mean()
    excluded = sorted(set(accm[accm < 0.5].index) | set(canm[canm < 0.2].index))   # not served / no vision / blind
    pd.DataFrame({"accepted": accm, "canary": canm}).loc[excluded].to_csv("results/excluded_models.csv")
    print("excluded (blind / not served):", excluded)
    df = df[~df.model.isin(excluded)]
    man = pd.read_csv(a.manifest)
    df = df.merge(man[["id", "theme", "dpi", "n_series", "metric", "unit", "log_y", "unit_trap", "truncated_y", "decoy", "clutter", "difficulty"]], on="id", how="left")
    df["pred_d"] = df["pred"].apply(lambda s: json.loads(s) if isinstance(s, str) else {})
    df["pred_event"] = df["pred_d"].apply(lambda d: d.get("event_type"))
    os.makedirs("results/charts", exist_ok=True)
    # per-model summary with Wilson CI on composite-as-proportion (mean of 5 binary fields)
    rows = []
    for mdl, g in df.groupby("model"):
        k = g[FIELDS].values.sum(); n = g[FIELDS].size; p, lo, hi = wilson(k, n)
        rows.append(dict(model=mdl, panels=len(g), runs=g.run_id.nunique(), composite=p, ci_lo=lo, ci_hi=hi,
                         **{f: g[f].mean() for f in FIELDS}, blind_rate=1 - g["canary_ok"].mean(), accepted=g["accepted"].mean(),
                         cost_usd=g["cost_usd"].sum() / g.run_id.nunique(), cost_per_correct_panel=(g["cost_usd"].sum() / max((g["composite"] == 1).sum(), 1)),
                         median_latency_s=g["latency_s"].median(), peak_mae_rel=g["peak_rel_err"].median()))
    sm = pd.DataFrame(rows).sort_values("composite", ascending=False); sm.to_csv("results/summary_by_model.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format): print(sm.to_string(index=False))
    # start-time behaviour (event panels only): precision, "none" rate, tick snapping, bias
    def _hhmm(v):
        import re as _re; m = _re.search(r"(\d{1,2}):(\d{2})", str(v)); return None if not m else int(m.group(1)) * 60 + int(m.group(2))
    ev = df[df.truth_event != "none"].copy(); ev["ps"] = ev["pred_d"].apply(lambda d: _hhmm(d.get("start_time"))); ev["ts"] = ev["truth_start"].apply(_hhmm)
    ev["delta"] = ev["ps"] - ev["ts"]; ev["pred_none"] = ev["ps"].isna(); ev["pred_on_tick"] = ev["ps"].apply(lambda v: v is not None and v % 30 == 0)
    st = ev.groupby("model").agg(start_ok=("start_ok", "mean"), pred_none=("pred_none", "mean"), pred_on_tick=("pred_on_tick", "mean"),
                                 median_abs_delta=("delta", lambda x: x.abs().median()), p90_abs_delta=("delta", lambda x: x.abs().quantile(.9)), bias_min=("delta", "median"))
    st.sort_values("start_ok", ascending=False).to_csv("results/start_time_behaviour.csv")
    ev.pivot_table(index="model", columns="truth_event", values="start_ok", aggfunc="mean").to_csv("results/slice_start_by_event.csv")
    print("\nstart-time behaviour:\n", st.sort_values("start_ok", ascending=False).round(2).to_string())
    # slices
    for col in ["dpi", "theme", "n_series", "difficulty", "truth_event", "unit_trap", "log_y", "truncated_y", "decoy", "clutter"]:
        piv = df.groupby(["model", col])["composite"].mean().unstack(col); piv.to_csv(f"results/slice_{col}.csv")
    # confusion (all models pooled + per model)
    conf = pd.crosstab(df["truth_event"], df["pred_event"]); conf.to_csv("results/confusion_pooled.csv"); print("\npooled confusion:\n", conf)
    # charts
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(sm) + 1.5)); y = np.arange(len(sm))
    ax.barh(y, sm.composite, xerr=[sm.composite - sm.ci_lo, sm.ci_hi - sm.composite], color="#4c78a8", capsize=2)
    ax.set_yticks(y); ax.set_yticklabels(sm.model); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("composite reading accuracy (5 fields, Wilson 95% CI)")
    fig.tight_layout(); fig.savefig("results/charts/leaderboard.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(sm.cost_usd, sm.composite, s=30)
    for _, r in sm.iterrows(): ax.annotate(r.model, (r.cost_usd, r.composite), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("cost per full run (USD, log)"); ax.set_ylabel("composite accuracy"); fig.tight_layout(); fig.savefig("results/charts/cost_vs_accuracy.png", dpi=150); plt.close(fig)
    for col, fname in [("dpi", "accuracy_vs_resolution.png"), ("difficulty", "accuracy_vs_difficulty.png"), ("n_series", "accuracy_vs_series.png")]:
        piv = df.groupby(["model", col])["composite"].mean().unstack(col)
        fig, ax = plt.subplots(figsize=(8, 5)); piv.T.plot(ax=ax, marker="o", legend=False, alpha=0.7); ax.set_ylabel("composite accuracy"); ax.set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(f"results/charts/{fname}", dpi=150); plt.close(fig)
    print("\ncharts →", os.listdir("results/charts"))

if __name__ == "__main__":
    main()
