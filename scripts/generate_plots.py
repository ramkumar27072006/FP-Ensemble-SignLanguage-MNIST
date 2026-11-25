"""
Generate result visualisation figures for the README.
Run from the project root:
    python scripts/generate_plots.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(OUT, exist_ok=True)

# ── Experimental data ────────────────────────────────────────────────────────
labels = ["Original FP\n(R=1)", "FP-Ensemble\n(R=3)", "FP-Ensemble\n(R=5)", "FP-Ensemble\n(R=8)"]
R_vals = [1, 3, 5, 8]

test_acc_mean  = [95.01, 94.83, 96.31, 94.71]
test_acc_std   = [ 0.35,  0.85,  0.49,  1.12]

test_auc_mean  = [99.67, 99.83, 99.79, 99.76]
test_auc_std   = [ 0.07,  0.02,  0.08,  0.09]

train_times    = [ 8.76, 10.19, 12.42, 15.05]

STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
}
plt.rcParams.update(STYLE)

BLUE  = "#2563EB"
GREEN = "#16A34A"
RED   = "#DC2626"
GREY  = "#6B7280"

x = np.arange(len(R_vals))
bar_w = 0.55

# ── Figure 1: Test Accuracy bar chart ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(x, test_acc_mean, bar_w, yerr=test_acc_std,
              color=[GREY, BLUE, BLUE, BLUE],
              edgecolor="white", linewidth=0.8,
              error_kw=dict(ecolor="black", capsize=4, lw=1.2))
bars[2].set_color(GREEN)   # highlight best

for i, (m, s) in enumerate(zip(test_acc_mean, test_acc_std)):
    ax.text(x[i], m + s + 0.15, f"{m:.2f}%", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Test Accuracy (%)", fontsize=11)
ax.set_title("Test Accuracy vs Ensemble Size  —  Sign Language MNIST", fontsize=12, fontweight="bold")
ax.set_ylim(92, 97.5)

best_patch = mpatches.Patch(color=GREEN, label="Best (R=5, 96.31%)")
base_patch = mpatches.Patch(color=GREY,  label="Baseline (R=1)")
ax.legend(handles=[best_patch, base_patch], fontsize=9, loc="lower right")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "test_accuracy.png"))
plt.close(fig)
print("Saved test_accuracy.png")

# ── Figure 2: AUC bar chart ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(x, test_auc_mean, bar_w, yerr=test_auc_std,
              color=[GREY, BLUE, BLUE, BLUE],
              edgecolor="white", linewidth=0.8,
              error_kw=dict(ecolor="black", capsize=4, lw=1.2))
bars[1].set_color(GREEN)   # R=3 best AUC

for i, (m, s) in enumerate(zip(test_auc_mean, test_auc_std)):
    ax.text(x[i], m + s + 0.01, f"{m:.2f}%", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Test AUC — OvR (%)", fontsize=11)
ax.set_title("Test AUC vs Ensemble Size  —  Sign Language MNIST", fontsize=12, fontweight="bold")
ax.set_ylim(99.4, 100.05)

best_patch = mpatches.Patch(color=GREEN, label="Best AUC (R=3, 99.83%)")
base_patch = mpatches.Patch(color=GREY,  label="Baseline (R=1)")
ax.legend(handles=[best_patch, base_patch], fontsize=9, loc="lower right")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "test_auc.png"))
plt.close(fig)
print("Saved test_auc.png")

# ── Figure 3: Training time ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(R_vals, train_times, marker="o", color=BLUE, linewidth=2,
        markersize=8, markerfacecolor="white", markeredgewidth=2)

for r, t in zip(R_vals, train_times):
    ax.text(r, t + 0.2, f"{t:.2f}s", ha="center", fontsize=9, fontweight="bold")

ax.set_xlabel("Ensemble Size R", fontsize=11)
ax.set_ylabel("Total Training Time (seconds)", fontsize=11)
ax.set_title("Training Time vs Ensemble Size  —  Single Forward Pass, No Backprop",
             fontsize=12, fontweight="bold")
ax.set_xticks(R_vals)
ax.set_ylim(5, 18)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "training_time.png"))
plt.close(fig)
print("Saved training_time.png")

# ── Figure 4: Accuracy vs Time trade-off ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
scatter = ax.scatter(train_times, test_acc_mean, s=120,
                     c=[0, 1, 2, 3], cmap="Blues", zorder=5,
                     edgecolors="black", linewidths=0.8)

for r, t, a in zip(R_vals, train_times, test_acc_mean):
    ax.annotate(f"R={r}", (t, a), textcoords="offset points",
                xytext=(6, 4), fontsize=10)

ax.set_xlabel("Total Training Time (seconds)", fontsize=11)
ax.set_ylabel("Test Accuracy (%)", fontsize=11)
ax.set_title("Accuracy vs Training Time Trade-off", fontsize=12, fontweight="bold")
ax.set_ylim(93.5, 97)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "accuracy_vs_time.png"))
plt.close(fig)
print("Saved accuracy_vs_time.png")

# ── Figure 5: Architecture diagram (text-based) ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

blocks = [
    (5.0, 9.2, "Input  (1 x 28 x 28)", "#E5E7EB", "black"),
    (5.0, 7.7, "Conv2D FP-Ensemble  1->64 ch  |  3x3  |  ReLU + MaxPool + Norm  ->  14x14", "#DBEAFE", BLUE),
    (5.0, 6.2, "Conv2D FP-Ensemble  64->128 ch  |  3x3  |  ReLU + MaxPool + Norm  ->  7x7",  "#DBEAFE", BLUE),
    (5.0, 4.7, "Conv2D FP-Ensemble  128->256 ch  |  3x3  |  ReLU + MaxPool + Norm  ->  3x3", "#DBEAFE", BLUE),
    (5.0, 3.2, "Feature Fusion: Flatten + GAP + GMP  |  L2 Norm  |  + bias",                 "#FEF3C7", "#92400E"),
    (5.0, 1.7, "Dense FP-Ensemble  ->  1024 hidden  (Ridge Solve)  |  ReLU",                 "#DBEAFE", BLUE),
    (5.0, 0.4, "Ridge Readout  ->  24 class logits",                                          "#D1FAE5", GREEN),
]

for (cx, cy, txt, fc, tc) in blocks:
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=8.5,
            color=tc, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=fc, edgecolor=tc, linewidth=1.2))
    if cy > 0.5:
        ax.annotate("", xy=(cx, cy - 0.5), xytext=(cx, cy - 0.3),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))

eq_txt = r"$W_l = (A^T A + \lambda I)^{-1} A^T \tilde{Z}_{ensemble}$"
ax.text(5, -0.3, "Each block solved analytically — zero gradient computation",
        ha="center", va="center", fontsize=9, color=GREY, style="italic")

ax.set_title("FP-Ensemble CNN — Architecture Overview", fontsize=13, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "architecture.png"), bbox_inches="tight")
plt.close(fig)
print("Saved architecture.png")

print("\nAll figures generated in:", OUT)
