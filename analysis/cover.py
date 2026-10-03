import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import matplotlib.image as mpimg
BG, INK, INK2, BLUE, ORANGE, AQUA = "#181b1f", "#ffffff", "#c3c2b7", "#3987e5", "#d95926", "#199e70"
fig = plt.figure(figsize=(10, 4.2), dpi=200); fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.035, 0.12, 0.52, 0.76]); ax.imshow(mpimg.imread("data/full/panel_0075.png")); ax.axis("off")
t = fig.add_axes([0.59, 0.0, 0.39, 1.0]); t.axis("off"); t.set_facecolor(BG)
t.text(0, 0.84, "Can AI read\na dashboard?", color=INK, fontsize=27, fontweight="bold", va="top", linespacing=1.05)
t.text(0, 0.47, "30 vision models · 336 monitoring panels", color=INK2, fontsize=11.5, va="top")
rows = [("Gemini 3.7 Flash", "0.978", BLUE), ("GPT-6 Astra", "0.961", ORANGE), ("Claude Opus 5", "0.871", AQUA), ("GPT-5.4 nano", "0.561", ORANGE)]
for i, (n, s, c) in enumerate(rows):
    y = 0.36 - i * 0.075
    t.add_patch(plt.Rectangle((0, y - 0.022), 0.025, 0.044, color=c, transform=t.transAxes))
    t.text(0.05, y, n, color=INK, fontsize=11, va="center"); t.text(0.78, y, s, color=INK, fontsize=11, va="center", fontweight="bold")
fig.savefig("post/img/cover.png", facecolor=BG); print("cover written")
