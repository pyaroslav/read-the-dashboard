"""Part 2: multi-panel incidents. One image = three stacked panels (p99 latency, error rate, CPU) for one service
on a shared clock. 2 or 3 signals move, in a known order, with lags of 8-20 min. Labels come from the data.

Usage: python gen/incidents.py --out data/incidents --n 120 --seed 7
"""
import argparse, json, os, math
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates

SIGNALS = {"latency": ("p99 latency", "ms", 120.0, 7.0, None), "errors": ("error rate", "%", 0.4, 0.08, 100.0), "saturation": ("cpu", "%", 38.0, 2.5, 100.0)}
ORDER = ["latency", "errors", "saturation"]
SERVICES = ["checkout-api", "cart-svc", "auth-gw", "search-api", "payments", "inventory"]
MARK = {"A": "#8ab8ff", "B": "#f2cc0c", "C": "#ff7383"}

@dataclass
class Truth:
    id: str; seed: int; theme: str; dpi: int; service: str; moving: str; first_mover: str; order: str
    start_hhmm: str; onsets: str; implicated_deploy: str; markers: str; dashed_marker_count: int; n_samples: int; file: str

def noise_sd(y): return float(np.median(np.abs(np.diff(y))) / 0.954)

def onset(y, mu, sd, from_i):
    for i in range(from_i, len(y) - 5):
        if all(np.mean(y[j:j + 5]) > mu + 3 * sd for j in range(i, min(i + 10, len(y) - 5))): return i
    return -1

def make(i, seed, out_dir):
    r = np.random.default_rng(seed); n = int(r.choice([150, 180, 210]))
    theme = "dark" if r.random() < 0.5 else "light"; dpi = int(r.choice([75, 200])); svc = str(r.choice(SERVICES))
    k = int(r.choice([2, 3], p=[0.4, 0.6])); moving = list(r.permutation(ORDER)[:k])
    t0 = int(r.integers(int(n * 0.25), int(n * 0.5))); starts = [t0]
    for _ in moving[1:]: starts.append(starts[-1] + int(r.integers(8, 21)))
    if starts[-1] > n - 25: return None
    series, mus, sds = {}, {}, {}
    for s in ORDER:
        _, unit, base, sd, ceil = SIGNALS[s]
        y = base + r.normal(0, sd, n) + base * 0.03 * np.sin(np.linspace(0, 2 * math.pi * r.uniform(0.5, 2), n))
        mus[s], sds[s] = base, sd
        if s in moving:
            st = starts[moving.index(s)]
            if s == "saturation":                                    # climbs over ~8-15 min, then pins near the ceiling
                rise = int(r.integers(8, 16)); top = 96.0
                y[st:st + rise] = np.linspace(base, top, rise) + r.normal(0, sd, rise); y[st + rise:] = top - np.abs(r.normal(0, 0.6, n - st - rise))
            else:                                                    # abrupt step to 3-5x
                y[st:] *= r.uniform(3.0, 5.0)
        series[s] = np.clip(y, 0, ceil) if ceil else np.clip(y, 0, None)
    # labels from the data: visible onset of each moving signal, must keep >= 5 min separation and order
    vis = {s: onset(series[s], float(np.mean(series[s][:t0 - 2])), noise_sd(series[s][:t0 - 2]), max(0, starts[moving.index(s)] - 3)) for s in moving}
    if any(v < 0 for v in vis.values()): return None
    for s in ORDER:
        if s not in moving and onset(series[s], float(np.mean(series[s][:t0 - 2])), noise_sd(series[s][:t0 - 2]), 0) >= 0: return None   # a "flat" panel must stay flat
    if any(abs(vis[s] - starts[moving.index(s)]) > 4 for s in moving): return None   # the bend must be visible within 4 min
    vis = {s: starts[moving.index(s)] for s in moving}                                 # label = true start (steep changes)
    seq = list(moving); first = seq[0]; v0 = vis[first]
    start = datetime(2026, 9, 21, int(r.integers(0, 23)), int(r.choice([0, 15, 30, 45]))); t = [start + timedelta(minutes=m) for m in range(n)]
    n_marks = int(r.choice([1, 2, 3], p=[0.3, 0.4, 0.3])); idx = []
    if r.random() < 0.7: idx.append(max(2, v0 + int(r.integers(-1, 2))))
    tries = 0
    while len(idx) < n_marks and tries < 500:
        tries += 1; c = int(r.integers(5, n - 5))
        if all(abs(c - x) >= 18 for x in idx) and all(abs(c - vis[s]) >= 10 for s in moving): idx.append(c)
    idx = sorted(idx); letters = sorted(r.choice(["A", "B", "C"], size=len(idx), replace=False)); markers = dict(zip(letters, idx))
    implicated = next((L for L, x in markers.items() if abs(x - v0) <= 1), "none")
    plt.style.use("dark_background" if theme == "dark" else "default"); bg = "#181b1f" if theme == "dark" else "white"
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.6), dpi=dpi, sharex=True); fig.patch.set_facecolor(bg)
    col = "#73bf69" if theme == "dark" else "#1f77b4"
    for ax, s in zip(axes, ORDER):
        name, unit, *_ = SIGNALS[s]; ax.set_facecolor(bg); ax.plot(t, series[s], color=col, lw=1.4)
        ax.set_ylabel(unit); ax.set_title(f"{name} — {svc}", fontsize=9, loc="left"); ax.grid(alpha=0.25); ax.set_ylim(0, float(np.max(series[s])) * 1.18)
        for L, x in markers.items():
            ax.axvline(t[x], color=MARK[L], ls="--", lw=1)
            if s == "latency": ax.annotate(f"deploy {L}", (t[x], ax.get_ylim()[1]), xytext=(3, -10), textcoords="offset points", color=MARK[L], fontsize=8, va="top")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    fig.tight_layout(); pid = f"incident_{i:04d}"; fig.savefig(os.path.join(out_dir, pid + ".png"), facecolor=bg); plt.close(fig)
    return Truth(pid, seed, theme, dpi, svc, json.dumps(sorted(moving)), first, json.dumps(seq), t[v0].strftime("%H:%M"),
                 json.dumps({s: t[vis[s]].strftime("%H:%M") for s in seq}), implicated, json.dumps({L: t[x].strftime("%H:%M") for L, x in markers.items()}),
                 len(markers), n, pid + ".png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=120); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed); rows = []
    i = 0
    while len(rows) < a.n:
        tr = make(len(rows), int(rng.integers(0, 2**31 - 1)), a.out)
        if tr: rows.append(asdict(tr))
    import pandas as pd; pd.DataFrame(rows).to_csv(os.path.join(a.out, "manifest.csv"), index=False); print("wrote", len(rows))
