---
title: "I showed 30 AI models 336 monitoring dashboards. Most of them can't tell time."
published: true
tags: devchallenge, kagglechallenge, machinelearning, ai
cover_image: https://raw.githubusercontent.com/pyaroslav/read-the-dashboard/main/img/cover.png
---

*This is a submission for the [Kaggle Benchmarking Challenge](https://dev.to/challenges/kaggle-2026-09-23)*

Most incidents start with a graph.

Not a log line, but a panel: a latency line that jumped, a CPU series pinned at the ceiling, a gap where a metric stopped. The first thirty seconds of on-call are spent *reading a picture*.

AI copilots for on-call usually get the evidence as text. Real incidents hand you a screenshot. So I built a benchmark for the step the demos skip: **can a model read a monitoring panel the way an on-call engineer does?**

I ran it against 30 vision models on Kaggle. The best read these charts almost perfectly. Most of the rest find the peak but cannot read the clock, and when unsure they make the same mistake. Along the way, the best models also found three bugs in my answer key. *Updated Oct 3 with Part 2: three panels on one clock.*

## What I Benchmarked

**The capability:** turning one monitoring panel into five facts an engineer would act on.

| Field | Question | Graded as |
|---|---|---|
| What happened | step change, ramp, spike, flapping, plateau, gap, or none | exact match |
| When | the clock time it started, read off the x-axis | within ±2 min (±6 for slow ramps) |
| Which deploy | the dashed deploy marker (A/B/C) that lines up with the start, or none | exact match |
| Recovering? | is the series back to its earlier level by the right edge | exact match |
| How bad | the peak value, in the axis units | within ±10% |

A model's score is the mean of those five fields over every panel.

**The panels are synthetic, and that is the point.** A generator renders Grafana-style panels and records the ground truth, so nothing is hand-labelled and no model has seen them before. **336 panels** cover 7 event shapes × dark/light × 75/200 dpi × 12 variants, randomising the metric, 1–4 series, 0–3 deploy markers, and everyday traps:

- a **log-scale** y-axis
- latency drawn in milliseconds but **labelled in seconds**
- a **truncated** y-axis that exaggerates small moves
- a **decoy** neighbour series with a bigger but harmless swing
- **clutter**: alert-threshold lines and summary text on the panel

Here is one panel. Latency on `auth-gw` climbs from 15:26, right at deploy A, and pins flat at 0.29 s. That is a **plateau** (saturation), not a step change.

![Latency panel: climbs at deploy A around 15:26 and flat-lines at 0.29 s.](https://raw.githubusercontent.com/pyaroslav/read-the-dashboard/main/img/example_plateau.png)

| Model | What happened | When | Right? |
|---|---|---|---|
| Gemini 3.7 Flash | plateau | 15:25 | ✅ |
| GPT-6 Astra | plateau | 15:25 | ✅ |
| Claude Opus 5 | plateau | 15:25 | ✅ |
| Claude Sonnet 4.5 | step change | 15:30 | ❌ |
| Claude Haiku 4.5 | step change | 15:20 | ❌ |
| GPT-5.4 nano | step change | 15:30 | ❌ |
| Grok 4.20 (non-reasoning) | step change | 15:20 | ❌ |

**Canary fields catch models that cannot see.** Every answer also reports the y-axis label and the number of dashed markers, which are trivial if you can see the image. That turned out to matter (see "Blind but confident").

**How it runs on Kaggle.** It is one task built with the [`kaggle-benchmarks`](https://github.com/Kaggle/kaggle-benchmarks) SDK. The PNGs and their answer key live in a Kaggle dataset attached to the task. A per-panel sub-task sends one image and asks for a typed answer, in its own isolated chat.

{% details The task code, in short %}

```python
@dataclass
class Reading:
    event_type: str           # step_change, ramp, spike, flapping, plateau, gap, none
    start_time: str           # "HH:MM" read off the x-axis, or "none"
    implicated_deploy: str    # "A", "B", "C" or "none"
    recovering: bool
    peak_value: float
    y_axis_label: str         # canary
    dashed_marker_count: int  # canary

with kbench.chats.new(f"panel-{id}"):
    r = llm.prompt(PROMPT, image=images.from_path(path), schema=Reading,
                   extra_api_params={"max_tokens": 16384})
```

{% enddetails %}

The parent task runs all 336 panels with `.evaluate()` and returns the composite score, with six assertions for the per-field breakdown. Each panel gets a 16,384-token output budget, above every model's 99th percentile.

## Models Tested

**30 vision models from four vendors,** chosen to walk each vendor's price ladder rather than just crown a winner:

- **Google:** Gemini 3.5 / 3.6 / 3.7 / 3.8 Flash, 3 Flash preview, 3.1 and 3.5 Flash-Lite, 2.5 Flash, 2.5 Pro, 3.1 Pro preview, plus open-weight Gemma 4 26B and 31B
- **OpenAI:** GPT-5.4 nano, mini and full, GPT-5.5, GPT-5.6 Luna and Terra, GPT-6 Astra
- **Anthropic:** Claude Haiku 4.5, Sonnet 4.5 / 4.6 / 5, Opus 4.5 / 4.6 / 4.7 / 4.8 / 5
- **xAI:** Grok 4.20 with and without reasoning, the same model with the thinking dial flipped

A few catalog models could not be scored: Claude Opus 4.1, Claude Sonnet 4 and two Grok versions returned "model not found", DeepSeek-R1 rejects images, and three models accept the image but cannot see it (more below).

Every model ran all 336 panels, for 10,080 readings in the final version. Twenty-one models also ran two or three times on earlier versions with the same prompt; between runs a score moved by a median of **0.005** and never more than 0.015.

## Findings

![Bar chart of 30 models: Gemini 3.7 Flash leads at 0.978; GPT-5.4 nano is last at 0.561.](https://raw.githubusercontent.com/pyaroslav/read-the-dashboard/main/img/leaderboard.png)

### 1. The best chart reader costs $1.51

**Gemini 3.7 Flash leads at 0.978.** Gemini 3.8 Flash (0.972), Gemini 3.5 Flash (0.963) and GPT-6 Astra (0.961) sit within 0.02 behind it, with overlapping confidence intervals. The bills do not overlap: one full 336-panel run cost **$1.51** on Gemini 3.7 Flash and **$7.23** on GPT-6 Astra.

![Accuracy against cost: Gemini 3.7 Flash scores 0.978 for $1.51; Gemini 2.5 Pro scores 0.790 for $9.80.](https://raw.githubusercontent.com/pyaroslav/read-the-dashboard/main/img/cost_vs_accuracy.png)

Price does not buy chart reading. The most expensive run, **Gemini 2.5 Pro at $9.80, scored 0.790**, and Gemini 3.1 Pro ($7.75, 0.921) also trails every current Gemini Flash. Claude Opus 5 (0.871, $5.41) sits just below GPT-5.6 Luna (0.877, **$0.31**). On a budget, Luna and Gemma 4 31B (0.855, $0.46) are the bargains.

### 2. Everyone reads the peak. Few models read the clock.

![Per-field heatmap: peak 0.83–1.00 for every model; start time from 0.91 down to 0.20.](https://raw.githubusercontent.com/pyaroslav/read-the-dashboard/main/img/field_heatmap.png)

Split the score into its five fields and one column does almost all the work. **Peak value** is easy for every model (0.83–1.00). The millisecond-labelled-as-seconds trap barely registered: the worst drop on its 37 panels was 0.08, for Claude Opus 4.5. **Start time** is where models separate. The top five land within two minutes 87–91% of the time; the rest range from 20% to 80%, and 17 of the 30 models get it right less than half the time.

I expected weak models to snap to printed tick labels. They don't: they land on a tick about 4% of the time, the same as the true starts. They are simply imprecise. For the eight weakest, the median miss is 5–9 minutes, and one answer in ten is off by 15 to 45. At 3 a.m., "the deploy at 15:20 or the one at 15:30" is the whole question.

### 3. When in doubt, weak models say "step change"

![Confusion matrices: top models on the diagonal; the bottom ten call plateaus, flapping and gaps a step change.](https://raw.githubusercontent.com/pyaroslav/read-the-dashboard/main/img/confusion.png)

The top five name the event correctly almost every time (97–100% for every shape). The bottom ten have a default answer: **step change**. They give it for 56% of plateaus, 34% of flapping series and 25% of gaps. And 44% of real gaps get called "none". A metric that *stopped reporting* is often the actual incident.

This is the gap panel. `payments` stops reporting at about 04:33 and resumes about ten minutes later, right as deploy C lands.

![Requests panel: the payments line breaks at about 04:33 and resumes at about 04:44, at deploy C.](https://raw.githubusercontent.com/pyaroslav/read-the-dashboard/main/img/example_gap.png)

Gemini 3.7 Flash, GPT-6 Astra and Claude Opus 5 all said *gap, recovering, no deploy to blame*. Claude Sonnet 4.5, Claude Haiku 4.5 and GPT-5.4 nano said *none*. Grok 4.20 (non-reasoning) said *none*, and still blamed deploy C.

### 4. Every ladder climbs, at a different angle

![Model ladders: GPT 0.56 to 0.96, Claude Opus 0.68 to 0.87, Gemini 0.74 to 0.98.](https://raw.githubusercontent.com/pyaroslav/read-the-dashboard/main/img/ladders.png)

OpenAI's ladder is the steepest. GPT-5.4 nano (0.561) misreads the axis label or marker count on 64% of panels, and GPT-6 Astra is in the top four. Anthropic's ladder climbs steadily, from Haiku 4.5 at 0.596 to Sonnet 5 at 0.801 and Opus 5 at 0.871. Its top model still sits below Google's mid-tier Flash. Within Google, generation matters more than tier: both current Flash-Lites beat the older 2.5 Flash, and every current Flash beats both Pro models.

The thinking dial helps, at a price. The same Grok 4.20 scored **0.646 without reasoning and 0.694 with it**, and the reasoning run cost 11 times as much.

### 5. Low resolution hurts everyone except Google

![Accuracy at 75 vs 200 dpi: Claude gains 0.07–0.14, OpenAI 0.02–0.09, Google ±0.02.](https://raw.githubusercontent.com/pyaroslav/read-the-dashboard/main/img/resolution.png)

Half the panels are rendered at 75 dpi, about the size of a thumbnail pasted into chat, and half at 200 dpi. For every Gemini and Gemma model the difference is noise (±0.02). For **every Claude model it is 0.07 to 0.14**, and for OpenAI's models 0.02 to 0.09. At 75 dpi Claude Haiku 4.5 reads the canary fields correctly on 62% of panels; at 200 dpi, on 94%.

At 200 dpi alone the leaders barely move, but Claude Opus 5 climbs to 0.905 and Sonnet 5 to 0.845. **If you send screenshots to a Claude or GPT model, send them big.** The decoy series splits the same way: it costs Claude Haiku 4.5 and GPT-5.4 nano about 0.10, and current Gemini Flash models nothing measurable.

### 6. Blind but confident

This was the finding I did not expect. On an earlier run of the benchmark, **GLM-5 accepted 99.7% of the images without complaint and got the canary fields right on under 1% of them.** It reported axis labels like "requests per second" on CPU panels and counted dashed markers on panels that had none. It answered **"step change" on 305 of 336 panels**, each with a start time, a deploy letter and a peak value, all invented. Qwen3-235B, given the same images, answered "none" 319 times with an empty axis label.

Same proxy, same images. Qwen answered as if every panel were empty; GLM-5 produced 336 confident, fabricated incident readings. If you wire a model into an on-call tool, **check that it can see before you check whether it is smart.** A two-field canary costs nothing.

### 7. The models found the bugs in my answer key

My first full run used an answer key built from what the generator *intended*. Then I looked for panels where four or five of the top five models agreed with each other and disagreed with my key. There were **32 of them**, in three groups:

- **Hidden gaps.** On multi-series panels, other lines were drawn over the primary series and covered its missing-data stretch. My key said "gap"; the models saw a continuous line.
- **Invisible ramp onsets.** Some ramps were so shallow that their mathematical start was buried in the noise.
- **Ambiguous recovery.** A flapping series that ended on a low phase *was* back at its earlier level at the right edge, by my own prompt's definition, and my key still said "not recovering".

I rebuilt the generator so every label comes from the rendered data, with a self-check that rejects any panel whose labels cannot be read off the chart. Then I re-ran all 30 models. On the new key, event type, recovery and peak have **zero** disputed panels. Scores rose by up to 0.06; two models dipped by under 0.01, and no model moved more than two places.

One ambiguity is real rather than a bug. **When does a gradual ramp start?** At the first bend, or when it is clearly above the noise? On 16 slow ramps the top models split between those two readings. I re-scored ramps to accept any answer between the two. Scores moved by 0.009 on average and the ranking held, except GPT-6 Astra moving from fourth to second inside the top cluster. The public leaderboard uses the stricter key.

The lesson I'd keep: **when the strongest models agree with each other and not with you, check your key first.**

### 8. Update (Oct 3): Part 2, three panels on one clock

The first item on my "measure next" list turned out cheap enough to run. **Part 2** stacks three panels for one service on a shared clock: p99 latency, error rate and CPU. Two or three of them change, 8–20 minutes apart. The model has to say which signal moved first, the order the others followed, when it started, and which deploy lines up with it. That is the actual on-call question: is the CPU saturation causing the errors, or the other way round?

![Part 1 to Part 2 per model: ten rise to 0.97–1.00, the rest fall as low as 0.36.](https://raw.githubusercontent.com/pyaroslav/read-the-dashboard/main/img/part2_slope.png)

**Reading across panels splits the field instead of shifting it.** Ten models score **0.97–1.00**, higher than on single panels, and Gemini 3.7 Flash and GPT-6 Astra are perfect. GPT-5.6 Luna gets 0.985 for **$0.08** a run. Everyone else drops, by up to 0.24. The weak models mostly lose the sequence: Claude Haiku 4.5 gets the full order right 23% of the time and GPT-5.4 nano 25%, where every top model is at 97% or better.

Start times are easier here, because the changes are abrupt steps rather than gradual ramps: the top ten land within three minutes 98% of the time. That matches Part 1, where the gradual onsets were the hard part.

One more confession: my first Part 2 grader marked "saturation (CPU)" wrong when the right answer was "saturation". The phrase came from my own prompt, and 22 of 30 scores moved when I fixed it. Every number above is from the fixed re-run. Part 2 covers 29 models: Claude Sonnet 4.6 hit the provider's rate limit on all six attempts. It is a second task on the same Kaggle benchmark.

### What surprised me?

The resolution split followed vendor lines, not size or price. The unit trap I expected to be hardest was the easiest. And the best reader cost a fifth of GPT-6 Astra, which it beat.

### What would I measure next?

- **Messier multi-panel incidents.** Part 2 used clean steps. Next: overlapping incidents, a benign panel that moves too, and gradual onsets across panels.
- **Reading into action.** Given the panel and a runbook, does the model roll back the right deploy, and does it ever roll back when the right answer is "wait"?
- **Real screenshots.** Synthetic panels made a clean answer key possible. Next comes a small set of redacted real Grafana screenshots, graded by hand, to check that the ranking transfers.

The full per-field results for all 30 models are on the [Kaggle leaderboard](https://www.kaggle.com/benchmarks/yaroslavperventsev/read-the-dashboard) and in [`results/`](https://github.com/pyaroslav/read-the-dashboard/tree/main/results) on GitHub.

## My Benchmark

- **Kaggle benchmark (leaderboard):** https://www.kaggle.com/benchmarks/yaroslavperventsev/read-the-dashboard
- **Part 1 task** (336 single panels): https://www.kaggle.com/benchmarks/tasks/yaroslavperventsev/read-the-dashboard
- **Dataset** (336 panels plus answer key): https://www.kaggle.com/datasets/yaroslavperventsev/read-the-dashboard-panels
- **Part 2 task** (120 multi-panel incidents): https://www.kaggle.com/benchmarks/tasks/yaroslavperventsev/read-the-dashboard-incidents
- **Part 2 dataset:** https://www.kaggle.com/datasets/yaroslavperventsev/read-the-dashboard-incidents
- **Code** (generator, task, analysis, every per-panel answer): https://github.com/pyaroslav/read-the-dashboard

**Reproducibility.** The generator is seeded (seed 2026) and regenerates the identical 336 images. The grader is deterministic, with no LLM judge.

**Limitations, honestly.** The panels are synthetic and single-panel. The event vocabulary is mine ("plateau" vs "saturation"). "Recovering" mostly follows from the event type. On 5 panels two deploy labels overlap, though the marker lines stay distinct. Each model saw each panel once in the final run; earlier repeats barely moved the ranking.

*I built this with an AI coding assistant (Claude) helping with the generator, the Kaggle harness and the analysis. The benchmark design, the audit decisions and every number here were checked against the run data.*
