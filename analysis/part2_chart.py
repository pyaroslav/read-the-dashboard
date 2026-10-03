"""Part 2 chart: Part 1 score vs Part 2 score per model (slope chart), vendor colours. Reads latest valid v3 run per model."""
import pandas as pd, numpy as np, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "analysis"); import charts as C   # reuse palette, names, rcParams
d = pd.read_csv("results/inc_all_panels.csv", low_memory=False); d = d[d.version == 3]
d = d[d.run_id == d.groupby("model").run_id.transform("max")]
ok = d.groupby("model").accepted.apply(lambda s: s.astype(bool).mean()); keep = ok[ok >= 0.98].index
p2 = d[d.model.isin(keep)].groupby("model").composite.mean()
p1 = pd.read_csv("results/v7_final_leaderboard.csv").set_index("model").comp
t = pd.DataFrame({"p1": p1, "p2": p2}).dropna(); t["name"] = t.index.map(C.NICE); t["vendor"] = t.index.map(C.vendor)
t.to_csv("results/part1_vs_part2.csv")
fig, ax = plt.subplots(figsize=(8.5, 8))
for m, r in t.iterrows():
    c = C.VENDOR[r.vendor]; ax.plot([0, 1], [r.p1, r.p2], color=c, lw=1.6, alpha=0.8, marker="o", ms=6, mec=C.SURF, mew=1.5)
import textwrap
top = t[t.p2 >= 0.975].sort_values("p2", ascending=False); rest = t[t.p2 < 0.975].sort_values("p2")
names = ", ".join(n.replace(" (preview)", "") for n in top["name"])
ax.annotate(f"{len(top)} models at {top.p2.min():.2f}–{top.p2.max():.2f}:\n" + "\n".join(textwrap.wrap(names, 46)), (1, top.p2.mean()),
            xytext=(1.04, 1.0), textcoords="data", fontsize=7.5, color=C.INK2, va="center")
ys = []
for m, r in rest.iterrows():
    y = r.p2; y = max(y, ys[-1] + 0.014) if ys else y; y = min(y, top.p2.min() - 0.05) if y > top.p2.min() - 0.05 and len(ys) and False else y; ys.append(y)
    ax.annotate(f"{r['name']}  {r.p2:.2f}", (1, r.p2), xytext=(1.04, y), textcoords="data", fontsize=7.5, color=C.INK2, va="center",
                arrowprops=dict(arrowstyle="-", color=C.GRID, lw=0.6))
ax.set_xticks([0, 1], ["Part 1\nsingle panel", "Part 2\nthree panels, one clock"]); ax.set_xlim(-0.15, 1.75); ax.set_ylim(0.3, 1.06)
ax.set_ylabel("Composite accuracy"); ax.grid(axis="x", visible=False); C.legend(ax, loc="lower left")
ax.set_title("Reading across panels splits the field", loc="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig("post/img/part2_slope.png", dpi=200); plt.close(fig)
print(t.sort_values("p2", ascending=False).round(3).to_string())
