import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from os import makedirs
from os.path import isdir
from pathlib import Path
from string import ascii_lowercase

from UTILS.mutils import load_model_files

# save data
source_data_path = '../.source_data/fig6'

seed = 3
d = 8

# TAU = 0.1
if seed == 0:
    TAU = 10
elif seed == 3:
    TAU = 15
    # TAU = 10
DM_IDXS = (-2, -3)
TK_IDXS = (127, 128)


def attn_score_from_dist(g_dist, alpha, bdwth, d_intrinsic=None):
    if alpha < 2:
        if d_intrinsic is None:
            raise ValueError("d_intrinsic is required for alpha < 2.")
        return (1 + g_dist / bdwth ** (1 / alpha)) ** (-d_intrinsic - alpha)

    return np.exp(-(g_dist / bdwth ** 0.5) ** (alpha / (alpha - 1)))


def diffusion_map_from_score(attn_score, a):
    if a > 0:
        n_r = attn_score.sum(-1)
        n_c = attn_score.sum(-2)
        k_tilde = (n_r ** (-a))[..., None] * attn_score * (n_c ** (-a))[..., None, :]
    else:
        k_tilde = attn_score

    k_tilde = k_tilde[0, 0]
    d_tilde = k_tilde.sum(-1)
    d_inv_sqrt = np.diag(d_tilde ** (-0.5))
    k_hat = d_inv_sqrt @ k_tilde @ d_inv_sqrt
    k_hat_sym = 0.5 * (k_hat + k_hat.T)

    eigvals, eigvecs = np.linalg.eigh(k_hat_sym)
    S = d_tilde.sum()
    eigvecs = np.sqrt(S) * (d_inv_sqrt @ eigvecs)
    sort_idx = eigvals.argsort()
    return eigvals[sort_idx], eigvecs[:, sort_idx]


def bandwidth_selection_curve(model_dir, bdwths):
    attn_setup, _, _, _ = load_model_files(model_dir)
    alpha = attn_setup["alpha"]
    manifold = attn_setup["manifold"]
    d_intrinsic = None
    if alpha < 2:
        d_intrinsic = 1 if "v2_" in manifold else attn_setup["d_intrinsic"]

    data_all = np.load(Path(model_dir) / "attn_graph_results.npz")
    g_dist = data_all["g_dist"].astype(np.float64)
    x_len = data_all["X_len"].item()

    # load words
    with open(Path(model_dir) / "text.txt", 'r') as f:
        lines = f.readlines()
    text = [line.strip() for line in lines]

    scores = []
    for bdwth in bdwths:
        attn_score = attn_score_from_dist(
            g_dist,
            alpha,
            bdwth,
            d_intrinsic=d_intrinsic,
        )
        scores.append(attn_score[0, 0].sum())

    scores = np.array(scores)
    slopes = np.gradient(np.log(scores), np.log(bdwths))
    best_idx = np.argmax(slopes)
    best_attn_score = attn_score_from_dist(
        g_dist,
        alpha,
        bdwths[best_idx],
        d_intrinsic=d_intrinsic,
    )
    eigvals, eigvecs = diffusion_map_from_score(best_attn_score, attn_setup["a"])
    # dm_coords = eigvecs[:x_len, DM_IDXS] * (eigvals[list(DM_IDXS)] ** TAU)
    dm_coords = eigvecs[np.ix_(TK_IDXS,DM_IDXS)] * (eigvals[list(DM_IDXS)] ** TAU)
    # dm_coords = eigvecs[:x_len, DM_IDXS] * (eigvals[list(DM_IDXS)] ** bdwths[best_idx])
    dm_coords = dm_coords - dm_coords.mean(axis=0)

    return {
        "alpha": alpha,
        "model_name": attn_setup["model_name"],
        "x_len": x_len,
        "scores": scores,
        "slopes": slopes,
        "best_idx": best_idx,
        "best_bdwth": bdwths[best_idx],
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "dm_coords": dm_coords,
        "text": text
    }


script_dir = Path(__file__).resolve().parent
shared_path = (
    script_dir
    / ".droot"
    / "L-d-grid-v1mask-v7scaling-v2"
    / f"1L-hidden={d}-max_len=512-rescaled"
    / "config_qqv"
    / "imdb"
    / "layers=1-heads=1-qqv"
)
paths = [
    shared_path / f"oprdfnsformer-imdb-qqv-alpha=1.2-eps=1.0/model={seed}",
    shared_path / f"oprdfnsformer-imdb-qqv-alpha=2.0-eps=1.0/model={seed}",
]

bdwths = np.logspace(-4, 2, 61)
colors = ["#2E63A6", "#A4292F"]
token_markers = ("o", "s")

# fig, axs = plt.subplots(1, 3, figsize=(8.1, 2.2))
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
results = []
token_labels = None
for model_dir, color in zip(paths, colors):
    result = bandwidth_selection_curve(model_dir, bdwths)
    results.append(result)
    label = rf"$\alpha = {result['alpha']}$"
    best_idx = result["best_idx"]

    axs[0].plot(bdwths, result["scores"], marker="o", markersize=2.5, linewidth=1, c=color, label=label)
    axs[0].scatter(result["best_bdwth"], result["scores"][best_idx], marker="*", s=60, c=color, zorder=3)

    axs[1].plot(bdwths, result["slopes"], marker="o", markersize=2.5, linewidth=1, c=color, label=label)
    axs[1].scatter(result["best_bdwth"], result["slopes"][best_idx], marker="*", s=60, c=color, zorder=3)

    if token_labels is None:
        token_labels = [result["text"][tk_idx].lstrip("▁") for tk_idx in TK_IDXS]

    for iidx, marker in enumerate(token_markers):
        axs[2].scatter(
            result["dm_coords"][iidx, 0],
            result["dm_coords"][iidx, 1],
            marker=marker,
            color=color,
            alpha=0.9,
            s=40,
            edgecolors="black",
            linewidths=0.35,
            zorder=3,
        )

    axs[2].plot(
        result["dm_coords"][0:len(token_markers)+1, 0],
        result["dm_coords"][0:len(token_markers)+1, 1],
        color=color,
        linestyle='--'
    )

    print(
        f"alpha={result['alpha']}, X_len={result['x_len']}, "
        f"best bdwth={result['best_bdwth']:.4g}"
    )

font_size_title = 9

axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlabel(r"$\varepsilon$")
# axs[0].set_ylabel(r"$n^{-2}\sum_{ij}\mathbf{A}_{ij}$")
axs[0].set_ylabel(r"$S$")
axs[0].set_title("Kernel mass", fontsize=font_size_title)

axs[1].set_xscale("log")
axs[1].set_xlabel(r"$\varepsilon$")
axs[1].set_ylabel(r"$\frac{ \mathrm{d}\log S }{ \mathrm{d}\log \varepsilon }$")
axs[1].set_title("Selection criterion", fontsize=font_size_title)

# axs[2].set_xlim([-6e-3, 6e-3])
# axs[2].set_ylim([-4.8e-5, 4.8e-5])
axs[2].margins(0.3)
axs[2].ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
axs[2].set_xlabel(r"DM$_1$")
# Move the y-axis scale left of the axis so it does not lift the title.
# axs[2].yaxis.get_offset_text().set_horizontalalignment("right")
# axs[2].yaxis.get_offset_text().set_horizontalalignment("center")
axs[2].yaxis.offsetText.set_position((-0.4, 0.1))  
axs[2].set_ylabel(r"DM$_2$")
axs[2].set_title("Diffusion map", fontsize=font_size_title)
token_handles = [
    Line2D(
        [],
        [],
        linestyle="none",
        marker=marker,
        # markersize=6.3,
        markersize=5,
        markerfacecolor="0.6",
        markeredgecolor="black",
        markeredgewidth=0.35,
        label=token,
    )
    for marker, token in zip(token_markers, token_labels)
]
axs[2].legend(
    handles=token_handles,
    # title="Token",
    loc="best",
    frameon=False,
    fontsize=8,
    title_fontsize=8,
    handletextpad=0.4,
    labelspacing=0.3,
    borderaxespad=0.25,
    ncols=2,
)

for idx, ax in enumerate(axs):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if idx == 0:
        ax.legend(frameon=False, fontsize=8, loc="best")

    # subfigure labels
    ax.text(-0.06, 1.135, rf'$\mathbf{{{ascii_lowercase[idx]}}}$',
            transform=ax.transAxes, ha='left', va='top', usetex=False)

fig.tight_layout()

save_dir = script_dir / ".droot" / "figs_dir" / "nlp-mechanisms"
if not isdir(save_dir):
    makedirs(save_dir)

fig_file = f"dm-bdwth-selection-d={d}-seed={seed}.pdf"
plt.savefig(save_dir / fig_file, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved in {save_dir / fig_file}")

# Save the plotted values, with one file per panel.
Path(source_data_path).parent.mkdir(parents=True, exist_ok=True)
source_data = [
    {'bandwidth': bdwths},
    {'bandwidth': bdwths},
    {'token_index': TK_IDXS, 'token': token_labels},
]
for result in results:
    source_label = f"alpha={result['alpha']:g}"
    source_data[0][f'{source_label}_S'] = result['scores']
    source_data[1][f'{source_label}_slope'] = result['slopes']
    for panel_data in source_data[:2]:
        panel_data[f'{source_label}_selected'] = np.arange(len(bdwths)) == result['best_idx']
    source_data[2][f'{source_label}_DM1'] = result['dm_coords'][:, 0]
    source_data[2][f'{source_label}_DM2'] = result['dm_coords'][:, 1]

for panel_idx, panel_data in enumerate(source_data):
    panel_path = f'{source_data_path}{ascii_lowercase[panel_idx]}.csv'
    pd.DataFrame(panel_data).to_csv(panel_path, index=False)
    print(f'Source data saved in {panel_path}')
