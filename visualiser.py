# --- Box plots for DTU audio reconstruction results (Pearson r) ---
# Requirements: matplotlib, pandas, numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- 1) Load data ----------
paths = {
    "Linear (OLS)": "results/ols_results.csv",
    "Ridge (Tikhonov)": "results/tikhonov_results.csv",
    "Lasso": "results/lasso_results.csv",
    "ElasticNet": "results/elasticnet_results.csv",
    "TRF": "results/results_trf.csv",
    "CCA": "results/cca_results.csv",
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

# GraphEncoder values
graphencoder_vals = np.array([
    0.7667,0.6838,0.6500,0.6606,0.6814,0.6803,
    0.7070,0.6977,0.6626,0.6955,0.6871,0.5845,
    0.6482,0.7052,0.6950,0.7049,0.6977,0.6951
])
models["GraphEncoder"] = graphencoder_vals

# ---------- 2) Prepare plot data ----------
labels = list(models.keys())
data = [models[k] for k in labels]

# ---------- 3) Plot A: GraphEncoder-only box plot ----------
fig, ax = plt.subplots(figsize=(6, 5))
bp = ax.boxplot([graphencoder_vals], patch_artist=True, widths=0.5)
# style
for box in bp['boxes']:
    box.set(facecolor="skyblue", alpha=0.7)
for median in bp['medians']:
    median.set(color="black", linewidth=2)
# overlay mean
ax.scatter([1], [graphencoder_vals.mean()], s=60, marker='o',
           facecolors='white', edgecolors='black', zorder=3)
ax.set_xticks([1])
ax.set_xticklabels(["GraphEncoder"])
ax.set_ylabel("Pearson correlation (test)")
ax.set_title("GraphEncoder on DTU (18 subjects)")
ax.axhline(0, linestyle="dotted", color="gray")
# pleasant y-limits
ax.set_ylim(0.55, 0.80)
fig.tight_layout()

# ---------- 4) Plot B: Combined box plot ----------
fig2, ax2 = plt.subplots(figsize=(12, 6))
bp2 = ax2.boxplot(data, patch_artist=True, widths=0.5)
colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
for patch, c in zip(bp2['boxes'], colors):
    patch.set(facecolor=c, alpha=0.7)
for median in bp2['medians']:
    median.set(color="black", linewidth=2)
# overlay means
means = [np.mean(v) for v in data]
ax2.scatter(range(1, len(labels)+1), means, s=60, marker='o',
            facecolors='white', edgecolors='black', zorder=3)
ax2.set_xticks(range(1, len(labels)+1))
ax2.set_xticklabels(labels, rotation=15)
ax2.set_ylabel("Pearson correlation (test)")
ax2.set_title("Comparison of models on DTU (18 subjects)")
ax2.axhline(0, linestyle="dotted", color="gray")
# pleasant y-limits across all
ax2.set_ylim(-0.1, 0.80)
fig2.tight_layout()

# ---------- 5) Save ----------
out_dir = Path("results")
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "box_graphencoder.png", dpi=300)
fig2.savefig(out_dir / "box_all_models.png", dpi=300)

print("Saved plots to:", out_dir)
