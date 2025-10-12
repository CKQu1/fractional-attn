import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from os import makedirs
from os.path import isdir, isfile
from scipy.stats import vonmises, vonmises_fisher
from string import ascii_lowercase
from tqdm import tqdm

from constants import *
from UTILS.mutils import njoin

from UTILS.fdm_utils import get_markov_matrix, plot_vmf_density, get_vmf_samples, uniform_sphere_rvs

# ---------- Global plot settings ----------
COLORS_ALPHA = ["#636363", "#469C76", "#2E63A6", "#C17DA5", "#C66526", "#EEE461", "#A4292F"]
# ------------------------------------------

MARKERSIZE = 4
#BIGGER_SIZE = 10
BIGGER_SIZE = 8
LEGEND_SIZE = 8
TRANSP = 1  # transparency (corresponding to alpha in plot)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# figure set up
l_ratio = 2.5
#fig = plt.figure(figsize = (l_ratio*3, (l_ratio+0.1)*2))
#spec = mpl.gridspec.GridSpec(ncols=6, nrows=4) # 6 columns evenly divides both 2 & 3

fig = plt.figure(figsize = (5.95, 2.32))
spec = mpl.gridspec.GridSpec(ncols=6, nrows=2)
ax1 = fig.add_subplot(spec[0:2,0:2])                     # (a)
ax2 = fig.add_subplot(spec[0:2,2:4], projection='3d')    # (b)
ax3 = fig.add_subplot(spec[0,4:])                      # (c)
ax4 = fig.add_subplot(spec[1,4:])
# ax5 = fig.add_subplot(spec[2:4,1:3])    # (d)
# ax6 = fig.add_subplot(spec[2:4,3:5])    # (e)
# ax5 = fig.add_subplot(spec[2:4,0:2])    # (d)
# ax6 = fig.add_subplot(spec[2:4,2:4])    # (e)
# ax7 = fig.add_subplot(spec[2:4,4:])     # (f)

#axs = [[ax1,ax2,ax3,ax4],[ax5,ax6,ax7]]
#axs = [[ax1,ax2,ax3,ax4],[ax5,ax6]]
axs = [[ax1,ax2,ax3,ax4]]

# Set seed, working: 10, 20
seed = 50
np.random.seed(seed)

# Sample 1 (2D uniform grid large)

uniform_sample_size = 500
uniform_radians = np.linspace(0,2*np.pi,uniform_sample_size)
uniform_xys = np.stack([np.cos(uniform_radians), np.sin(uniform_radians)]).T
g_dists_2 = np.arccos(uniform_xys @ uniform_xys.T)

alphas1 = [1.2, 2]
#alphas1 = [1.2, 1.6, 2]
a = 0
bandwidth1 = 1e-4
n = g_dists_2.shape[0]
d = 1

# hard set distance along diagonals to be zero
for ii in range(n):
    g_dists_2[ii,ii] = 0

idxs = np.arange(1,uniform_sample_size+1)
idx_mid = int(n/2)

# ---------- (a) ----------
ax = axs[0][0]
for aidx, alpha in enumerate(alphas1):

    # c_alpha = HYP_CMAP(HYP_CNORM(alpha))
    c_alpha = COLORS_ALPHA[round((alpha - 1)/0.2) + 1]

    t = bandwidth1**(alpha/2)
    K, D, K_tilde, D_tilde = get_markov_matrix(g_dists_2, alpha, bandwidth1, d, a)

    # ---------- Eigvals ----------    
    if a == 0:
        K, D, K_tilde, D_tilde = get_markov_matrix(g_dists_2, alpha, bandwidth1, d, a)
        K_hat = np.diag(D_tilde**(-1/2)) @ K_tilde @ np.diag(D_tilde**(-1/2))
        K_hat_sym = 0.5*(K_hat + K_hat.T)

        eigvals, eigvecs = np.linalg.eigh(K_hat_sym)
        eigvecs_transformed = np.diag(D_tilde**(-0.5)) @ eigvecs

        # eigvals
        eidx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[eidx]; eigvecs = eigvecs[:,eidx]

        # transformation for operator
        eigvals_transformed = -1/t * np.log(eigvals)          
        # eye guide
        #power = alpha if alpha < 2 else 2
        power = alpha/2 if alpha < 2 else 2
        eigvals_theory = idxs**power  
        # eigvals_theory = eigvals_theory / eigvals_theory[idx_mid]
        # eigvals_theory = eigvals_theory * eigvals[idx_mid] * 10

        ax.plot(idxs, eigvals_transformed, c=c_alpha, label=rf'$\alpha = {{{alpha}}}$')  
        ax.plot(idxs, eigvals_theory, c=c_alpha, alpha=0.5, linewidth=1, linestyle='--')

        ax.set_xlim([1, n - 30])
        if d == 1:
            #ax.set_ylim([1, 1e6])
            ax.set_ylim([1, 1e5])
        ax.set_xscale('log'); ax.set_yscale('log')

ax.tick_params(axis='both', which='minor')
ax.legend(frameon=False)

# ----- Global plot settings -----

c_node = 'grey'
lstyle = '--'

# Self-attention
a = 0

mus = np.array([[0, -1, 0],  # bimodal
                [-1, 0, 0]])
kappa = 25

# Set seed, working: 10, 20
seed = 40
np.random.seed(seed)

# Sample 1 (VMF)
sample_size = 3
samples = get_vmf_samples(sample_size, mus, kappa)

large_sample_size = 1000
# Sample 3 (uniform grid large)
uniform_xyzs, count = uniform_sphere_rvs(large_sample_size)
# Sample 2 (VMF large)
large_sample_size = count
large_sample_xyzs = get_vmf_samples(int(large_sample_size/2), mus, kappa)

# Stacked large samples
#all_radians = np.stack([large_sample_radians, uniform_radians, nonuniform_radians])
all_xyzs = np.stack([large_sample_xyzs, uniform_xyzs])
all_radians = np.empty([all_xyzs.shape[0], large_sample_size, 2])
for idx in range(all_xyzs.shape[0]):
    theta = np.arccos(all_xyzs[idx][:,1]/all_xyzs[idx][:,0])
    phi = np.arctan(np.sqrt(all_xyzs[idx][:,0]**2 + all_xyzs[idx][:,1]**2)/all_xyzs[idx][:,2])
    all_radians[idx] = np.vstack([theta, phi]).T

alphas2 = [1.2, 2]
bandwidth2 = 1e-5

n_grid = 100
u = np.linspace(0, np.pi, n_grid)
v = np.linspace(0, 2 * np.pi, n_grid)
u_grid, v_grid = np.meshgrid(u, v)
vertices = np.stack([np.cos(v_grid) * np.sin(u_grid),
                     np.sin(v_grid) * np.sin(u_grid),
                     np.cos(u_grid)],
                    axis=2)
x = np.outer(np.cos(v), np.sin(u))
y = np.outer(np.sin(v), np.sin(u))
z = np.outer(np.ones_like(u), np.cos(u))

# ---------- (b) ----------
plot_vmf_density(fig, axs, [0,1], x, y, z, vertices, mus, kappa)

# ---------- (c) ----------
# Points and interactions on circle (non-uniform)
g_dists = np.arccos(samples @ samples.T)
for ii in range(g_dists.shape[0]):
    g_dists[ii,ii] = 0

# stereographic projection
qk_share = True
if qk_share:
    Q = W = np.stack([samples[:,0] / (1 - samples[:,2]), samples[:,1] / (1 - samples[:,2])]).T    

n = Q.shape[0]
#scale = 3 * n
scale = 0.75 * n

bandwidth1 = 1e-1
#thresh = 1e-12

alphas1 = [1.2, 2]
for bidx, alpha in enumerate(alphas1):
    K, D, K_tilde, D_tilde = get_markov_matrix(g_dists, alpha, bandwidth1, d, a)
    M = np.diag(D_tilde**(-1)) @ K_tilde

    print(f'alpha = {alpha}')
    print(f'M min: {M.min()}, max: {M.max()}')
    print(f'K min: {K.min()}, max: {K.max()}')
    print(f'K_tilde min: {K_tilde.min()}, max: {K_tilde.max()}')
    print('\n')    

    ax = axs[0][bidx+2]
    # c_alpha = HYP_CMAP(HYP_CNORM(alpha))
    c_alpha = COLORS_ALPHA[round((alpha - 1)/0.2) + 1]

    if alpha < 2:
        thresh = M.min()

    for i in range(n):
        for j in range(n):
            if M[i, j] > thresh:
                if bidx ==0:
                    ax.plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], c=c_alpha, linewidth=scale * M[i, j], 
                                                                               linestyle=lstyle, zorder=1)
                elif bidx == 1:
                    ax.plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], c=c_alpha, linewidth=scale * M[i, j], 
                                                                               linestyle=lstyle, zorder=1)
                else:
                    ax.plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], c=c_alpha, linewidth=scale * M[i, j], 
                                                                               linestyle=lstyle, zorder=1)

    # c='#dd1c77'
    ax.scatter(Q[:, 0], Q[:, 1], label='Queries', lw=.25, c=c_node, edgecolors="k", s=20, zorder=2)
    if not qk_share:
        # c='#a8ddb5'
        ax.scatter(W[:, 0], W[:, 1], label='Keys', lw=.5, c=c_node,  edgecolors="k", s=20, zorder=2)

    #ax.set_xlim([-1.2,1.2]);ax.set_ylim([-1.2,1.2])
    ax.set_xticklabels([]);ax.set_yticklabels([])
    ax.set_title(rf'$\alpha = {alpha}$')
    ax.axis('off')
    #ax.set_ylabel(rf'$\alpha = {alpha}$')

# ----- plot settings/labels -----

axs[0][0].set_title('Operator')
axs[0][1].set_title('Bimodal von-Mises')

ii = 0
label_size = 13
for row in range(len(axs)):
    for col in range(len(axs[row])):  
        if (row,col) != (0,3):
            ax = axs[row][col]
            # subfigure labels
            # if '3d' in ax.name:
            #     ax.text2D(
            #         0.0, 1.0, f'({ascii_lowercase[ii]})', transform=(
            #             ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            #         fontsize=label_size, va='bottom', fontfamily='sans-serif')    
            # else:
            #     ax.text(
            #         0.0, 1.0, f'({ascii_lowercase[ii]})', transform=(
            #             ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            #         fontsize=label_size, va='bottom', fontfamily='sans-serif')       

            ii += 1

axs[0][0].spines['top'].set_visible(False)
axs[0][0].spines['right'].set_visible(False)

plt.tight_layout()
FIGS_DIR = njoin(FIGS_DIR, 'schematic')
if not isdir(FIGS_DIR): makedirs(FIGS_DIR)
plt.savefig(njoin(FIGS_DIR, 'figure_fdm_v4.pdf'), bbox_inches='tight')
print(f'Figure saved in {FIGS_DIR}')