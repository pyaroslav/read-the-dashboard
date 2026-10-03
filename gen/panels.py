"""Synthetic Grafana-style monitoring panels with exact ground truth (v3: labels = what is visible; self-checked).

Usage:  python gen/panels.py --out data/pilot --n 28 --seed 1 --balanced
Writes PNGs + manifest.csv (one row per panel: ground truth, canaries, trap flags).

Event definitions (must match the prompt in task/read_the_dashboard.py):
  step_change  sustained jump to a new level lasting >= 15 min; may roll back later (recovering=True)
  ramp         steady climb over 30+ min, never returns
  spike        burst shorter than 10 min, returns to the earlier level (recovering=True)
  flapping     repeated alternation between two levels from the start time onward
  plateau      climbs over 15-30 min then pins dead-flat at a ceiling (saturation) and stays there
  gap          a stretch of missing data (10–25 min), line resumes afterwards (recovering=True)
  none         nothing beyond normal noise/drift
Start-time tolerance used by the grader: ±2 min, except ramp ±6 min (onset is gradual).
"""
import argparse, json, os, math
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

EVENTS = ["step_change", "ramp", "spike", "flapping", "plateau", "gap", "none"]
METRICS = {  # name: (unit, baseline, noise_sd, ceiling)
    "p99 latency": ("ms", 120.0, 7.0, None),
    "error rate": ("%", 0.4, 0.08, 100.0),
    "cpu": ("%", 38.0, 2.5, 100.0),
    "queue depth": ("msgs", 850.0, 60.0, None),
    "requests": ("req/s", 2400.0, 90.0, None),
}
SERVICES = ["checkout-api", "cart-svc", "auth-gw", "search-api", "payments", "inventory", "notif-worker"]
PALETTE_DARK = ["#73bf69", "#8ab8ff", "#f2cc0c", "#ff7383", "#b877d9", "#ffa64d"]
PALETTE_LIGHT = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#8c564b"]
MARK_COLORS = {"A": "#8ab8ff", "B": "#f2cc0c", "C": "#ff7383"}

@dataclass
class Truth:
    id: str; seed: int; theme: str; dpi: int; n_series: int; metric: str; unit: str; service: str
    event_type: str; start_idx: int; start_hhmm: str; implicated_deploy: str; markers: str
    recovering: bool; peak_value: float; y_axis_label: str; dashed_marker_count: int
    log_y: bool; unit_trap: bool; truncated_y: bool; decoy: bool; clutter: bool; n_samples: int; difficulty: int; file: str; raw_start_idx: int = -1

def make_series(rng, n, base, sd, ceiling):
    y = base + rng.normal(0, sd, n) + base * 0.03 * np.sin(np.linspace(0, 2 * math.pi * rng.uniform(0.5, 2), n))
    return np.clip(y, 0, ceiling) if ceiling else np.clip(y, 0, None)

def apply_event(rng, y, ev, base, ceiling):
    """Returns (series, start_idx, recovering). start_idx=-1 for 'none'."""
    n = len(y); y = y.copy(); recovering = False
    if ev == "none":
        return y, -1, False
    t0 = int(rng.integers(int(n * 0.2), int(n * 0.7)))
    head = 0.85 * ceiling / max(np.nanmax(y), 1e-9) if ceiling else None   # max multiplier that stays clear of the ceiling
    if ev == "step_change":
        f = rng.uniform(2.0, 5.0) if head is None else min(rng.uniform(1.8, 2.4), head); y[t0:] *= f
        if rng.random() < 0.4 and n - t0 > 30:           # rollback after >= 15 min at the new level
            t1 = int(rng.integers(t0 + 15, n - 8)); y[t1:] /= f; recovering = True
    elif ev == "ramp":
        top = rng.uniform(2.0, 4.0) if head is None else min(rng.uniform(1.0, 1.3), head - 1.0)
        sd = float(np.median(np.abs(np.diff(y[:t0]))) / 0.954)
        rise = max(base * max(top, 0.6), 15 * sd)                      # v3: a ramp must rise >= 15 noise-widths to be readable
        if ceiling: rise = min(rise, 0.95 * ceiling - float(np.max(y[:t0])))
        y[t0:] += np.linspace(0, rise, n - t0)
    elif ev == "spike":
        w = int(rng.integers(3, 9)); y[t0:t0 + w] *= (rng.uniform(4.0, 8.0) if head is None else min(rng.uniform(1.8, 2.4), head)); recovering = True
    elif ev == "flapping":
        k = int(rng.integers(4, 9)); f = rng.uniform(3.0, 4.0) if head is None else min(rng.uniform(1.8, 2.4), head)
        for i in range(t0, n):
            if ((i - t0) // k) % 2 == 0: y[i] *= f
        last_phase = ((n - 1 - t0) // k) % 2
        if last_phase == 1:                                   # v3: never end on a low phase — keep "recovering" unambiguous
            tail = t0 + ((n - 1 - t0) // k) * k; y[tail:] *= f
    elif ev == "plateau":
        cap = ceiling if ceiling else float(np.round(base * rng.uniform(2.2, 3.0), -1))
        rise = int(rng.integers(15, 31)); rise = min(rise, n - t0 - 10)   # climb over 15-30 min, then pinned flat at the ceiling
        y[t0:t0 + rise] = np.linspace(y[t0 - 1], cap, rise) + rng.normal(0, base * 0.02, rise); y[t0 + rise:] = cap - np.abs(rng.normal(0, cap * 0.002, max(0, n - t0 - rise)))
    elif ev == "gap":
        w = int(rng.integers(10, 26)); y[t0:t0 + w] = np.nan; recovering = True
    if ceiling: y = np.clip(y, 0, ceiling)
    return y, t0, recovering


def _noise(y):
    y = y[~np.isnan(y)]; return float(np.median(np.abs(np.diff(y))) / 0.954 + 1e-12) if len(y) > 3 else 1e-9   # robust sd from first differences

def visible_onset(y, t0, ev):
    """v3: the answer key uses the first minute the event is VISIBLE, not the generator's mathematical start."""
    if ev in ("none",): return -1
    if ev in ("step_change", "spike", "flapping", "gap"): return t0
    pre = y[:t0]; mu, sd = float(np.nanmean(pre)), _noise(pre)
    for i in range(t0, len(y) - 5):
        w = y[i:i + 5]
        if np.nanmean(w) > mu + 3 * sd and all(np.nanmean(y[j:j + 5]) > mu + 3 * sd for j in range(i, min(i + 15, len(y) - 5))):
            return i
    return -2                                                # never becomes visible -> reject

def derived_recovering(y, t0):
    pre = y[:t0]; mu, sd = float(np.nanmean(pre)), _noise(pre)
    tail = y[~np.isnan(y)][-5:]
    return bool(abs(float(np.mean(tail)) - mu) <= 3 * sd)

def self_check(ev, y0, t0, vis, recovering, others):
    """Reject panels whose labels cannot be read off the rendered data."""
    errs = []
    if ev != "none":
        if vis == -2: errs.append("onset never visible")
        if derived_recovering(y0, t0) != recovering: errs.append("recovering label disagrees with the data")
        if ev == "gap":
            mu = float(np.nanmean(y0[:t0])); sd = _noise(y0[:t0])
            for o in others:
                if abs(float(np.nanmean(o)) - mu) < 6 * max(sd, _noise(o)): errs.append("another series overlaps the gap level")
    return errs

def render(t, series, names, colors, markers, theme, dpi, log_y, unit_label, truncated_y, clutter, path, service):
    plt.style.use("dark_background" if theme == "dark" else "default")
    bg = "#181b1f" if theme == "dark" else "white"
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=dpi); fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    handles = []
    for z, (y, nm, c) in reversed(list(enumerate(zip(series, names, colors)))):   # v3: primary drawn last, on top
        (h,) = ax.plot(t, y, color=c, lw=1.8 if z == 0 else 1.3, label=nm, zorder=5 if z == 0 else 3); handles.append((z, h))
    handles = [h for _, h in sorted(handles)]
    ymax = np.nanmax(np.concatenate(series)); ymin = np.nanmin(np.concatenate(series))
    if log_y: ax.set_yscale("log")
    elif truncated_y: ax.set_ylim(ymin * 0.97, ymax * 1.08)
    else: ax.set_ylim(0, ymax * 1.15)
    for letter, idx in markers.items():
        ax.axvline(t[idx], color=MARK_COLORS[letter], ls="--", lw=1.1)
        ax.annotate(f"deploy {letter}", (t[idx], ax.get_ylim()[1]), xytext=(3, -10), textcoords="offset points",
                    color=MARK_COLORS[letter], fontsize=8, va="top")
    if clutter:
        thr = ymax * 0.9; ax.axhline(thr, color="#ff7383", ls=":", lw=1); ax.text(t[2], thr, "alert threshold", fontsize=7, color="#ff7383", va="bottom")
        ax.text(0.99, 0.02, f"avg {np.nanmean(series[0]):.1f}  max {ymax:.1f}  last {series[0][-1]:.1f}", transform=ax.transAxes,
                fontsize=7, ha="right", color="#9aa0a6")
    ax.set_title(f"{names[0].split(' · ')[0]} — {service}", fontsize=10, loc="left")
    ax.set_ylabel(unit_label); ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax.legend(handles=handles, loc="upper left", fontsize=7, ncol=min(len(names), 2), frameon=False)
    fig.tight_layout(); fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)

def make_panel(i, rng, out_dir, event=None, theme=None, dpi=None):
    for attempt in range(40):
        seed = int(rng.integers(0, 2**31 - 1))
        out = _make_panel(i, seed, out_dir, event, theme, dpi)
        if out is not None: return out
    raise RuntimeError(f"panel {i}: no seed passed the self-check")

def _make_panel(i, seed, out_dir, event=None, theme=None, dpi=None):
    r = np.random.default_rng(seed)
    ev = event or EVENTS[int(r.integers(len(EVENTS)))]
    theme = theme or ("dark" if r.random() < 0.6 else "light"); dpi = dpi or int(r.choice([75, 200]))
    metric = list(METRICS)[int(r.integers(len(METRICS)))]; unit, base, sd, ceiling = METRICS[metric]
    n = int(r.choice([120, 180, 240])); n_series = int(r.choice([1, 1, 2, 3, 4]))
    log_y = bool(r.random() < 0.15 and unit != "%"); unit_trap = bool(r.random() < 0.5 and unit == "ms")
    truncated_y = bool(r.random() < 0.2 and not log_y); decoy = bool(n_series > 1 and r.random() < 0.5); clutter = bool(r.random() < 0.3)
    start = datetime(2026, 9, 14, int(r.integers(0, 23)), int(r.choice([0, 15, 30, 45])))
    t = [start + timedelta(minutes=k) for k in range(n)]
    services = list(r.choice(SERVICES, size=n_series, replace=False)); primary = services[0]
    y0 = make_series(r, n, base, sd, ceiling); y0, t0, recovering = apply_event(r, y0, ev, base, ceiling)
    series, names = [y0], [f"{metric} · {primary}"]
    for s in services[1:]:
        lvl = r.choice([0.45, 1.8]) if ev == "gap" else r.uniform(0.6, 1.4)          # v3: keep neighbours off the gap's level
        yd = make_series(r, n, base * lvl, sd, ceiling)
        if decoy and s == services[1]:
            yd = yd * (1 + 0.6 * np.sin(np.linspace(0, math.pi, n)))   # benign slow swell on a neighbour service
            if ceiling: yd = np.clip(yd, 0, ceiling)
        series.append(yd); names.append(f"{metric} · {s}")
    vis = visible_onset(y0, t0, ev) if t0 >= 0 else -1
    if self_check(ev, y0, t0, vis, recovering, series[1:]): return None
    raw_t0 = t0
    t0 = vis if ev != "none" else t0
    unit_label = unit
    if unit_trap:                                                       # values drawn in ms, axis labelled in s
        series = [y / 1000.0 for y in series]; unit_label = "s"
    # deploy markers: at most one coincides with the start (±1 min); the others sit >= 8 min away from it
    markers, implicated = {}, "none"
    n_marks = int(r.choice([0, 1, 2, 3], p=[0.15, 0.3, 0.3, 0.25]))
    if n_marks:
        coincide = ev != "none" and r.random() < 0.7
        idxs = [max(2, min(n - 3, t0 + int(r.integers(-1, 2))))] if coincide else []
        pool = [k for k in range(5, n - 5) if t0 < 0 or abs(k - t0) >= 8]
        while len(idxs) < n_marks:                                   # keep markers >= 14 min apart so labels never overlap
            k = int(r.choice(pool))
            if all(abs(k - j) >= 14 for j in idxs): idxs.append(k)
        idxs = sorted(idxs)
        letters = sorted(r.choice(["A", "B", "C"], size=n_marks, replace=False))   # any subset of labels, so a lone marker is not always "A"
        markers = dict(zip(letters, idxs))
        for L, k in markers.items():
            if t0 >= 0 and abs(k - t0) <= 1: implicated = L
    peak = float(np.nanmax(series[0]))
    pid = f"panel_{i:04d}"; path = os.path.join(out_dir, pid + ".png")
    colors = (PALETTE_DARK if theme == "dark" else PALETTE_LIGHT)[:n_series]
    render(t, series, names, colors, markers, theme, dpi, log_y, unit_label, truncated_y, clutter, path, primary)
    difficulty = int(log_y) + int(unit_trap) + int(truncated_y) + int(decoy) + int(clutter) + (n_series - 1) + int(dpi == 75)
    return Truth(id=pid, seed=seed, theme=theme, dpi=dpi, n_series=n_series, metric=metric, unit=unit_label, service=primary,
                 event_type=ev, start_idx=t0, start_hhmm=(t[t0].strftime("%H:%M") if t0 >= 0 else "none"), implicated_deploy=implicated,
                 markers=json.dumps({k: t[v].strftime("%H:%M") for k, v in markers.items()}), recovering=recovering,
                 peak_value=round(peak, 4), y_axis_label=unit_label, dashed_marker_count=len(markers), log_y=log_y, unit_trap=unit_trap,
                 truncated_y=truncated_y, decoy=decoy, clutter=clutter, n_samples=n, difficulty=difficulty, file=pid + ".png", raw_start_idx=raw_t0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--balanced", action="store_true", help="cycle events×themes×dpi")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed); rows = []
    combos = [(e, th, d) for e in EVENTS for th in ("dark", "light") for d in (75, 200)]
    for i in range(a.n):
        e, th, d = combos[i % len(combos)] if a.balanced else (None, None, None)
        rows.append(asdict(make_panel(i, rng, a.out, e, th, d)))
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "manifest.csv"), index=False)
    print(f"wrote {len(rows)} panels to {a.out}")
