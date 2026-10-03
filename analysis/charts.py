"""Post-quality charts from results/summary_by_model.csv + results/all_panels.csv (v6, latest run per model).
Vendor = colour (validated 3-slot categorical + neutral gray for xAI); every mark is direct-labelled or in the post's table."""
import json, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
VENDOR = {"Google": "#2a78d6", "OpenAI": "#eb6834", "Anthropic": "#1baf7a", "xAI": "#8a8984"}
NICE = {"gemini-3.7-flash": "Gemini 3.7 Flash", "gemini-3.8-flash": "Gemini 3.8 Flash", "gemini-3.5-flash": "Gemini 3.5 Flash", "gemini-3.6-flash": "Gemini 3.6 Flash",
        "gemini-3-flash-preview": "Gemini 3 Flash (preview)", "gemini-3.1-pro-preview": "Gemini 3.1 Pro (preview)", "gemini-2.5-pro": "Gemini 2.5 Pro", "gemini-2.5-flash": "Gemini 2.5 Flash",
        "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite", "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite", "gemma-4-31b-it": "Gemma 4 31B", "gemma-4-26b-a4b-it": "Gemma 4 26B",
        "gpt-6-astra": "GPT-6 Astra", "gpt-5.5-2026-04-23": "GPT-5.5", "gpt-5.6-terra": "GPT-5.6 Terra", "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.4-2026-03-05": "GPT-5.4",
        "gpt-5.4-mini-2026-03-17": "GPT-5.4 mini", "gpt-5.4-nano-2026-03-17": "GPT-5.4 nano", "claude-opus-5-default": "Claude Opus 5", "claude-opus-4-8-default": "Claude Opus 4.8",
        "claude-opus-4-7-default": "Claude Opus 4.7", "claude-opus-4-6-default": "Claude Opus 4.6", "claude-opus-4-5-20251101": "Claude Opus 4.5", "claude-sonnet-5-default": "Claude Sonnet 5",
        "claude-sonnet-4-6-default": "Claude Sonnet 4.6", "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5", "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
        "grok-4.20-0309-reasoning": "Grok 4.20 (reasoning)", "grok-4.20-0309-non-reasoning": "Grok 4.20 (non-reasoning)"}
def vendor(m): return "Google" if m.startswith(("gemini", "gemma")) else "OpenAI" if m.startswith("gpt") else "Anthropic" if m.startswith("claude") else "xAI"

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})

sm = pd.read_csv("results/summary_by_model.csv"); sm["name"] = sm.model.map(NICE); sm["vendor"] = sm.model.map(vendor)
sm = sm.sort_values("composite", ascending=True).reset_index(drop=True)
def legend(ax, loc="lower right"):
    h = [Line2D([0], [0], marker="o", ls="", ms=8, mfc=c, mec=SURF, label=v) for v, c in VENDOR.items()]
    ax.legend(handles=h, loc=loc, frameon=False, labelcolor=INK2)

# 1. leaderboard
fig, ax = plt.subplots(figsize=(8.5, 9.5)); y = np.arange(len(sm))
ax.barh(y, sm.composite, height=0.72, color=sm.vendor.map(VENDOR), edgecolor=SURF, linewidth=2)
ax.errorbar(sm.composite, y, xerr=[sm.composite - sm.ci_lo, sm.ci_hi - sm.composite], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
for yi, v, hi in zip(y, sm.composite, sm.ci_hi): ax.text(hi + 0.008, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y, sm.name); ax.set_xlim(0.4, 1.0); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Composite reading accuracy — 5 fields × 336 panels (bars: 95% Wilson CI)")
ax.set_title("Read the Dashboard — 30 vision models", loc="left", fontsize=13, fontweight="bold"); legend(ax)
fig.tight_layout(); fig.savefig("post/img/leaderboard.png", dpi=200); plt.close(fig)

# 2. cost vs accuracy
fig, ax = plt.subplots(figsize=(9, 5.8))
for v, g in sm.groupby("vendor"): ax.scatter(g.cost_usd, g.composite, s=70, color=VENDOR[v], edgecolor=SURF, linewidth=2, zorder=3)
label = {"Gemini 3.7 Flash": (-30, 14), "GPT-6 Astra": (10, 14), "GPT-5.5": (14, -2), "Gemini 3.1 Pro (preview)": (-20, -26), "Gemini 2.5 Pro": (-20, -24), "Claude Opus 5": (-95, -20),
         "GPT-5.6 Luna": (8, 8), "Gemini 3.1 Flash-Lite": (8, -14), "Gemma 4 31B": (8, 6), "Claude Haiku 4.5": (8, -4), "GPT-5.4 nano": (8, 4),
         "Grok 4.20 (reasoning)": (10, -4), "Grok 4.20 (non-reasoning)": (8, -4), "Claude Sonnet 5": (-20, 16)}
for _, r in sm.iterrows():
    if r["name"] in label:
        dx, dy = label[r["name"]]
        ax.annotate(r["name"], (r.cost_usd, r.composite), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK2, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=0, shrinkB=4))
ax.set_xscale("log"); ax.set_xticks([0.1, 0.3, 1, 3, 10], ["$0.10", "$0.30", "$1", "$3", "$10"]); ax.minorticks_off()
ax.set_xlabel("Cost of one full run (336 panels), log scale"); ax.set_ylabel("Composite reading accuracy")
ax.set_title("Price does not buy chart reading", loc="left", fontsize=13, fontweight="bold"); legend(ax)
fig.tight_layout(); fig.savefig("post/img/cost_vs_accuracy.png", dpi=200); plt.close(fig)

# 3. per-field heatmap (sequential blue), sorted by composite
fields = [("event_ok", "What\nhappened"), ("start_ok", "When\n(±2 min)"), ("deploy_ok", "Which\ndeploy"), ("recovering_ok", "Recovering?"), ("peak_ok", "Peak\n(±10%)")]
h = sm.sort_values("composite", ascending=False)
M = h[[f for f, _ in fields]].values
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
fig, ax = plt.subplots(figsize=(7.2, 10)); ax.grid(False)
ax.imshow(M, cmap=cmap, vmin=0.2, vmax=1.0, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.72 else INK)
ax.set_xticks(range(len(fields)), [l for _, l in fields]); ax.xaxis.tick_top(); ax.set_yticks(range(len(h)), h.name); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
fig.suptitle("Everyone reads the peak.\nFew models read the clock.", x=0.02, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig("post/img/field_heatmap.png", dpi=200); plt.close(fig)

# 4. ladders
lad = [("GPT", ["gpt-5.4-nano-2026-03-17", "gpt-5.4-mini-2026-03-17", "gpt-5.4-2026-03-05", "gpt-5.5-2026-04-23", "gpt-6-astra"], "OpenAI"),
       ("Claude Opus", ["claude-opus-4-5-20251101", "claude-opus-4-6-default", "claude-opus-4-7-default", "claude-opus-4-8-default", "claude-opus-5-default"], "Anthropic"),
       ("Gemini", ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite-preview", "gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.7-flash"], "Google")]
comp = sm.set_index("model").composite
fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
for ax, (fam, ms, v) in zip(axes, lad):
    ys = [comp[m] for m in ms]; ax.plot(range(5), ys, color=VENDOR[v], lw=2, marker="o", ms=8, mec=SURF, mew=2, zorder=3)
    for i, m in enumerate(ms):
        ax.annotate(f"{ys[i]:.2f}", (i, ys[i]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    short = [NICE[m].replace("Claude ", "").replace("Gemini ", "").replace(" (preview)", "").replace("GPT-", "").replace("Opus ", "") for m in ms]
    ax.set_xticks(range(5), short, rotation=30, ha="right", fontsize=8.5); ax.set_xlim(-0.4, 4.4)
    ax.set_title(fam, loc="left", fontsize=11, fontweight="bold", color=VENDOR[v])
axes[0].set_ylim(0.5, 1.0); axes[0].set_ylabel("Composite reading accuracy")
fig.suptitle("Climbing each vendor's ladder", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig("post/img/ladders.png", dpi=200); plt.close(fig)

# 5. where the answers go: pooled confusion, bottom-10 vs top-5 models
d = pd.read_csv("results/all_panels.csv", low_memory=False); d = d[(d.version == 7) & d.model.isin(sm.model)]
d = d[d.run_id == d.groupby("model").run_id.transform("max")]
d["pe"] = d.pred.apply(lambda s: (json.loads(s).get("event_type") or "").strip().lower().replace(" ", "_") if isinstance(s, str) else "no answer")
ev = ["step_change", "ramp", "spike", "flapping", "plateau", "gap", "none"]; d.loc[~d.pe.isin(ev), "pe"] = "other"
order = sm.sort_values("composite").model.tolist(); bottom, top = order[:10], order[-5:]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, grp, title in [(axes[0], top, "Top 5 models"), (axes[1], bottom, "Bottom 10 models")]:
    c = pd.crosstab(d[d.model.isin(grp)].truth_event, d[d.model.isin(grp)].pe, normalize="index").reindex(index=ev, columns=ev + ["other"], fill_value=0)
    ax.imshow(c.values, cmap=cmap, vmin=0, vmax=1, aspect="auto"); ax.grid(False)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            v = c.values[i, j]
            if v >= 0.03: ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(c.shape[1]), [x.replace("_", " ") for x in c.columns], rotation=35, ha="right"); ax.set_yticks(range(len(ev)), [x.replace("_", " ") for x in ev])
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold"); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
axes[0].set_ylabel("What actually happened"); fig.supxlabel("What the model said", color=INK2, fontsize=10)
fig.suptitle("When in doubt, weak models say “step change”", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig("post/img/confusion.png", dpi=200); plt.close(fig)
print("wrote", sorted(__import__("os").listdir("post/img")))

# 6. resolution dumbbell (75 vs 200 dpi), sorted by gain
rs = pd.read_csv("results/resolution_split.csv"); rs["name"] = rs.model.map(NICE); rs["vendor"] = rs.model.map(vendor)
rs["gain"] = rs.high_200dpi - rs.low_75dpi; rs = rs.sort_values("gain").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8.5, 9.5)); y = np.arange(len(rs))
for yi, r in rs.iterrows():
    c = VENDOR[r.vendor]; ax.plot([r.low_75dpi, r.high_200dpi], [yi, yi], color=c, lw=2, alpha=0.55, zorder=2)
    ax.scatter([r.low_75dpi], [yi], s=46, facecolor=SURF, edgecolor=c, linewidth=2, zorder=3)
    ax.scatter([r.high_200dpi], [yi], s=46, color=c, edgecolor=SURF, linewidth=1.5, zorder=4)
    g = 0.0 if abs(r.gain) < 0.005 else r.gain; ax.text(max(r.low_75dpi, r.high_200dpi) + 0.012, yi, f"{g:+.2f}" if g else "0.00", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y, rs.name); ax.tick_params(axis="y", length=0); ax.grid(axis="y", visible=False); ax.set_xlim(0.45, 1.0)
ax.set_xlabel("Composite accuracy:  ○ 75 dpi panels   ● 200 dpi panels")
ax.set_title("Low resolution hurts everyone except Google", loc="left", fontsize=13, fontweight="bold"); legend(ax, loc="lower left")
fig.tight_layout(); fig.savefig("post/img/resolution.png", dpi=200); plt.close(fig)
print("resolution chart written")
