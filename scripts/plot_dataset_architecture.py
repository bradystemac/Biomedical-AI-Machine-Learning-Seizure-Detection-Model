"""Draw the Epileptic Seizure Recognition dataset architecture diagram.

Saves figures/dataset_architecture.png. Pass --show to also open the plot window.
"""
import sys
from pathlib import Path

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================
# Epileptic Seizure Recognition Dataset Architecture
# ============================================================

# Create a larger figure with additional whitespace
fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.axis("off")

# Consistent typography and colors
COLORS = {
    "title": "#1e293b",
    "text": "#334155",
    "muted": "#64748b",
    "arrow": "#475569",
    "line": "#cbd5e1",
    "raw_bg": "#e2e8f0",
    "raw_edge": "#64748b",
    "segment_bg": "#dbeafe",
    "segment_edge": "#2563eb",
    "final_bg": "#f0fdf4",
    "final_edge": "#16a34a",
    "seizure_bg": "#fee2e2",
    "seizure_edge": "#dc2626",
    "seizure_text": "#991b1b",
    "control_bg": "#f1f5f9",
    "control_edge": "#475569",
}

# Helper function for consistently styled boxes
def add_box(x, y, width, height, facecolor, edgecolor, linewidth=2):
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=1,
    )
    ax.add_patch(box)


# ============================================================
# TITLE
# ============================================================

ax.text(
    7.5, 9.55,
    "Epileptic Seizure Recognition Dataset Architecture",
    fontsize=19,
    fontweight="bold",
    ha="center",
    va="center",
    color=COLORS["title"],
)


# ============================================================
# STEP 1: RAW EEG RECORDINGS
# ============================================================

x1, y1, w1, h1 = 0.45, 6.55, 3.55, 1.95

add_box(
    x1, y1, w1, h1,
    COLORS["raw_bg"],
    COLORS["raw_edge"],
)

ax.text(
    x1 + w1 / 2, y1 + 1.53,
    "STEP 1: RAW EEG RECORDINGS",
    fontsize=11.5,
    fontweight="bold",
    ha="center",
    va="center",
    color="#0f172a",
)

ax.text(
    x1 + w1 / 2, y1 + 0.88,
    "500 original recordings\n"
    "Each recording: 23.6 seconds\n"
    "4,097 data points per recording",
    fontsize=10.5,
    ha="center",
    va="center",
    linespacing=1.5,
    color=COLORS["text"],
)


# ============================================================
# ARROW 1: CHUNKING
# ============================================================

ax.annotate(
    "",
    xy=(4.55, 7.48),
    xytext=(4.12, 7.48),
    arrowprops=dict(
        arrowstyle="-|>",
        color=COLORS["arrow"],
        lw=2.2,
        mutation_scale=16,
    ),
)

ax.text(
    4.33, 7.82,
    "Chunk",
    fontsize=10,
    fontweight="bold",
    ha="center",
    va="center",
    color=COLORS["segment_edge"],
)


# ============================================================
# STEP 2: SEGMENTATION
# ============================================================

x2, y2, w2, h2 = 4.65, 6.55, 4.45, 1.95

add_box(
    x2, y2, w2, h2,
    COLORS["segment_bg"],
    COLORS["segment_edge"],
)

ax.text(
    x2 + w2 / 2, y2 + 1.53,
    "STEP 2: SEGMENTATION PROCESS",
    fontsize=11.5,
    fontweight="bold",
    ha="center",
    va="center",
    color="#1e3a8a",
)

ax.text(
    x2 + w2 / 2, y2 + 0.87,
    "Divide each recording into 23 chunks\n"
    "Each chunk represents approximately 1 second\n"
    "178 sequential data points per chunk",
    fontsize=10.2,
    ha="center",
    va="center",
    linespacing=1.5,
    color="#1e40af",
)


# ============================================================
# ARROW 2
# ============================================================

ax.annotate(
    "",
    xy=(9.65, 7.48),
    xytext=(9.22, 7.48),
    arrowprops=dict(
        arrowstyle="-|>",
        color=COLORS["arrow"],
        lw=2.2,
        mutation_scale=16,
    ),
)


# ============================================================
# STEP 3: FINAL DATASET
# ============================================================

x3, y3, w3, h3 = 9.75, 6.55, 4.80, 1.95

add_box(
    x3, y3, w3, h3,
    COLORS["final_bg"],
    COLORS["final_edge"],
)

ax.text(
    x3 + w3 / 2, y3 + 1.53,
    "STEP 3: FINAL DATASET",
    fontsize=11.5,
    fontweight="bold",
    ha="center",
    va="center",
    color="#14532d",
)

ax.text(
    x3 + w3 / 2, y3 + 0.84,
    "11,500 rows (500 recordings × 23 chunks)\n"
    "178 input features (X1–X178)\n"
    "1 target label (y)",
    fontsize=10.5,
    ha="center",
    va="center",
    linespacing=1.5,
    color="#166534",
)


# ============================================================
# SECTION DIVIDER
# ============================================================

ax.plot(
    [0.5, 14.5],
    [5.95, 5.95],
    color=COLORS["line"],
    lw=1.5,
    linestyle="--",
)

ax.text(
    7.5, 5.48,
    "TARGET CLASS BREAKDOWN AND BINARY LABEL MAPPING",
    fontsize=14,
    fontweight="bold",
    ha="center",
    va="center",
    color=COLORS["title"],
)


# ============================================================
# CLASS 1: EPILEPTIC SEIZURE
# ============================================================

sx, sy, sw, sh = 0.65, 1.75, 6.45, 3.05

add_box(
    sx, sy, sw, sh,
    COLORS["seizure_bg"],
    COLORS["seizure_edge"],
    linewidth=2.2,
)

ax.text(
    sx + sw / 2, sy + 2.55,
    "CLASS 1: EPILEPTIC SEIZURE",
    fontsize=13,
    fontweight="bold",
    ha="center",
    va="center",
    color=COLORS["seizure_text"],
)

ax.text(
    sx + sw / 2, sy + 1.77,
    "Recording of active EEG seizure activity\n"
    "from the epileptic focus area",
    fontsize=11,
    ha="center",
    va="center",
    linespacing=1.6,
    color="#7f1d1d",
)

# Binary mapping badge
badge_w, badge_h = 2.8, 0.55
badge_x = sx + (sw - badge_w) / 2
badge_y = sy + 0.43

add_box(
    badge_x, badge_y, badge_w, badge_h,
    "#dc2626",
    "#dc2626",
    linewidth=0,
)

ax.text(
    badge_x + badge_w / 2,
    badge_y + badge_h / 2,
    "y = 1  |  POSITIVE",
    fontsize=10.5,
    fontweight="bold",
    ha="center",
    va="center",
    color="white",
)


# ============================================================
# CLASSES 2–5: NON-SEIZURE
# ============================================================

cx, cy, cw, ch = 7.90, 1.75, 6.45, 3.05

add_box(
    cx, cy, cw, ch,
    COLORS["control_bg"],
    COLORS["control_edge"],
    linewidth=2.2,
)

ax.text(
    cx + cw / 2, cy + 2.55,
    "CLASSES 2–5: NON-SEIZURE",
    fontsize=13,
    fontweight="bold",
    ha="center",
    va="center",
    color="#334155",
)

# List the original classes separately for readability
ax.text(
    cx + cw / 2, cy + 1.65,
    "Class 5: Eyes open\n"
    "Class 4: Eyes closed\n"
    "Class 3: Healthy area of tumor patient\n"
    "Class 2: EEG directly from tumor site",
    fontsize=10.5,
    ha="center",
    va="center",
    linespacing=1.35,
    color="#475569",
)

# Binary mapping badge
badge_x2 = cx + (cw - badge_w) / 2
badge_y2 = cy + 0.43

add_box(
    badge_x2, badge_y2, badge_w, badge_h,
    "#475569",
    "#475569",
    linewidth=0,
)

ax.text(
    badge_x2 + badge_w / 2,
    badge_y2 + badge_h / 2,
    "y = 0  |  NEGATIVE",
    fontsize=10.5,
    fontweight="bold",
    ha="center",
    va="center",
    color="white",
)


# ============================================================
# FOOTER
# ============================================================

ax.text(
    7.5, 0.85,
    "Binary classification: seizure (1) vs. non-seizure (0)",
    fontsize=11,
    fontstyle="italic",
    ha="center",
    va="center",
    color=COLORS["muted"],
)

# Render the figure without clipping
plt.tight_layout(pad=1.5)

# Save a high-resolution copy
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
out_path = FIG_DIR / "dataset_architecture.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")

if "--show" in sys.argv:
    plt.show()
