# --- Violin plots for DTU audio reconstruction results (Pearson r) ---
# Requirements: matplotlib, pandas, numpy. (No seaborn)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- 1) Load data ----------
# Your uploaded files
paths = {
    "Linear (OLS)": "/mnt/data/ols_results.csv",
    "Ridge (Tikhonov)": "/mnt/data/tikhonov_results.csv",
    "Lasso": "/mnt/data/lasso_results.csv",
    "ElasticNet": "/mnt/data/elasticnet_results.csv",
    "TRF": "/mnt/data/results_trf.csv",
    "CCA": "/mnt/data/cca_results.csv",
}

def load_test_corr(p):
    """Return a 1D numpy array of test Pearson r from a CSV with varied schemas."""
    df = pd.read_csv(p)
    cols_lower = {c.lower(): c for c in df.columns}
    if "test_r" in df.columns:
        return df["test_r"].to_numpy()
    if "pearson_corr" in cols_lower:
        return df[cols_lower["pearson_corr"]].to_numpy()
    if "r_test" in cols_lower:
        return df[cols_lower["r_test"]].to_numpy()
    if "test" in cols_lower:
        return df[cols_lower["test"]].to_numpy()
    raise ValueError(f"Could not find a test-correlation column in {p}")

models = {name: load_test_corr(p) for name, p in paths.items()}

# GraphEncoder values you pasted
graphencoder_vals = np.array([
    0.7667483290036521,0.6838066587845485,0.6499688583115737,0.6605681985616684,
    0.6814221143722534,0.6803185900052389,0.706988758345445,0.6977075609068076,
    0.6626099447409312,0.6954959308107694,0.6871348157525062,0.5844842548171679,
    0.6482306314011415,0.7052284280459086,0.6949925084908803,0.7048755342761676,
    0.6976999814311663,0.6951241652170818
])
models["GraphEncoder"] = graphencoder_vals

# ---------- 2) Helpers ----------
def tight_ylim(values, lower_q=2.5, upper_q=97.5, pad=0.02):
    """Percentile-based y-limits with a small padding."""
    v = np.asarray(values).astype(float)
    lo = np.quantile(v, lower_q/100.0)
    hi = np.quantile(v, upper_q/100.0)
    span = max(1e-6, hi - lo)
    return lo - pad*span, hi + pad*span

def style_violins(parts, alpha=0.85):
    for pc in parts['bodies']:
        pc.set_alpha(alpha)
    # thicken median line if present
    if 'cmedians' in parts:
        for med in np.atleast_1d(parts['cmedians']):
            med.set_linewidth(2.0)

# ---------- 3) Plot A: GraphEncoder-only violin (cropped) ----------
fig, ax = plt.subplots(figsize=(7, 6))
parts = ax.violinplot([models["GraphEncoder"]], showmeans=False, showmedians=True, widths=0.75)
style_violins(parts)
# mean marker (white dot with black edge)
ax.scatter([1], [graphencoder_vals.mean()], s=46, marker='o',
           facecolors='white', edgecolors='black', zorder=3)
ax.set_xticks([1]); ax.set_xticklabels(["GraphEncoder"])
ax.set_ylabel("Reconstruction score (Pearson correlation)")
ax.set_title("GraphEncoder on DTU (18 subjects)")
ax.axhline(0, linestyle="dotted")
# focused y-limits
ymin, ymax = tight_ylim(graphencoder_vals, lower_q=2.5, upper_q=97.5, pad=0.05)
ax.set_ylim(ymin, ymax)
fig.tight_layout()

# ---------- 4) Plot B: Combined violin for all models (cropped), no comparison lines ----------
labels = list(models.keys())  # keep this order
data = [models[k] for k in labels]

fig2, ax2 = plt.subplots(figsize=(13, 6))
parts2 = ax2.violinplot(data, showmeans=False, showmedians=True, widths=0.8)
style_violins(parts2)
# overlay per-model means (white dots)
means = [np.mean(v) for v in data]
ax2.scatter(range(1, len(labels)+1), means, s=46, marker='o',
            facecolors='white', edgecolors='black', zorder=3)

ax2.set_xticks(range(1, len(labels)+1))
ax2.set_xticklabels(labels, rotation=15)
ax2.set_ylabel("Reconstruction score (Pearson correlation)")
ax2.set_title("Models compared on DTU (18 subjects)")
ax2.axhline(0, linestyle="dotted")

# focused y-limits for the combined plot (use all values)
all_vals = np.concatenate(data)
ymin2, ymax2 = tight_ylim(all_vals, lower_q=1.0, upper_q=99.0, pad=0.05)
# Optional hard caps to keep the focus sensible if outliers exist
ymin2 = max(-0.1, ymin2)
ymax2 = min(0.85, ymax2)
ax2.set_ylim(ymin2, ymax2)

fig2.tight_layout()

# ---------- 5) (Optional) Save to disk ----------
out_dir = Path("/mnt/data")
(fig).savefig(out_dir / "violin_graphencoder.png", dpi=300)
(fig2).savefig(out_dir / "violin_all_models.png", dpi=300)

print("Saved:")
print(out_dir / "violin_graphencoder.png")
print(out_dir / "violin_all_models.png")
