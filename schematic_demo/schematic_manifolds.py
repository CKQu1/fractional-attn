import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import scipy.stats as stats
from os.path import isdir, isfile, join
from os import makedirs
from scipy.stats import levy_stable
from scipy.stats import multivariate_normal
from matplotlib.cm import get_cmap
from matplotlib.transforms import ScaledTranslation
from matplotlib.patches import FancyArrowPatch
from string import ascii_lowercase
from scipy.optimize import brentq
from scipy.special import gamma, hyp1f1
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

SOURCE_DATA_PATH = '../../.source_data'

# local and non-local kernels
def fdm_kernel(g_dist, alpha, d_intrinsic, bandwidth=1, is_rescaled_dist=False):

    if is_rescaled_dist:
        g_dist = g_dist / d_intrinsic**0.5

    if alpha < 2:
        #attn_score = (1 + g_dist / head_dim**0.5 / bandwidth**0.5)**(-d_intrinsic-alpha)
        attn_score = (1 + g_dist / bandwidth**0.5)**(-d_intrinsic-alpha)
    else:
        #attn_score = torch.exp(-(g_dist / head_dim**0.5 / bandwidth**0.5)**(alpha/(alpha-1)))
        #attn_score = torch.exp(-(g_dist / bandwidth**0.5)**(alpha/(alpha-1)))
        attn_score = np.exp(-(g_dist / bandwidth**0.5)**(alpha/(alpha-1)))

    return attn_score

def get_markov_matrix(C, alpha, bandwidth, d, a):

    #sphere_radius = ((np.pi**(1/d)-1)/np.pi)
    if alpha >= 2:
        alpha_hat = alpha/(alpha-1)
        K = np.exp(-(C/bandwidth**0.5)**alpha_hat)
    else:
        # K = (1 + C/bandwidth**(1/alpha))**(-d-alpha)
        K = (1 + C/bandwidth**(1/2))**(-d-alpha)

    D = K.sum(-1)  # row normalization
    if a == 0:
        K_tilde = K
    else:
        K_tilde = np.diag(D**(-a)) @ K @ np.diag(D**(-a))
    D_tilde = K_tilde.sum(-1)

    #return np.diag(D_tilde**(-1)) @ K_tilde
    return K, D, K_tilde, D_tilde

np.random.seed(seed=2)

BIGGER_SIZE = 7.5
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
MARKERSIZE = 10
LWIDTH = 4
# plt.rcParams["font.family"] = "sans-serif"

alpha = 1.2
alphas_selected = [alpha, 2]

# color scheme 2
# HYP_CM = 'turbo'
# HYP_CMAP = get_cmap(HYP_CM)
# HYP_CNORM = mpl.colors.Normalize(vmin=1, vmax=2)
COLORS_ALPHA = ["#636363", "#469C76", "#2E63A6", "#C17DA5", "#C66526", "#EEE461", "#A4292F"]

# Create figure and subplots
nrows, ncols = 1, 3
fig, axes = plt.subplots(nrows, ncols)  #  gridspec_kw={'wspace': 0.3, 'hspace': 0.3}
fig.set_size_inches(5.5,2)
axes = axes[None,:]

##### a #####

# Sample 1 (2D uniform grid large)

ax = axes[0,0]

uniform_sample_size = 500
uniform_radians = np.linspace(0,2*np.pi,uniform_sample_size)
uniform_xys = np.stack([np.cos(uniform_radians), np.sin(uniform_radians)]).T
g_dists_2 = np.arccos(uniform_xys @ uniform_xys.T)

a = 0
bandwidth1 = 1e-4
n = g_dists_2.shape[0]
d = 1

# SAVE DATA
columns = []
for alpha in alphas_selected:
    for suffix in ['theory', 'exp']:
        columns.append(f'alpha={alpha}_{suffix}')
df_a = pd.DataFrame(columns=columns)

# hard set distance along diagonals to be zero
for ii in range(n):
    g_dists_2[ii,ii] = 0

idxs = np.arange(1,uniform_sample_size+1)
idx_mid = int(n/2)

for aidx, alpha in enumerate(alphas_selected):

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
        power = alpha if alpha < 2 else 2
        # power = alpha/2 if alpha < 2 else 2
        eigvals_theory = idxs**power
        # eigvals_theory = eigvals_theory / eigvals_theory[idx_mid]
        # eigvals_theory = eigvals_theory * eigvals[idx_mid] * 10

        ax.plot(idxs, eigvals_transformed, c=c_alpha, label=rf'$\alpha = {{{alpha}}}$')
        ax.plot(idxs, eigvals_theory, c=c_alpha, alpha=0.5, linewidth=1, linestyle='--')

        # ax.set_xlim([1, n - 30])
        ax.set_xlim([1, n - 40])
        # ax.set_xlim([1,n])
        if d == 1:
            #ax.set_ylim([1, 1e6])
            ax.set_ylim([1, 1e5])
        ax.set_xscale('log'); ax.set_yscale('log')

        df_a.loc[:,f'alpha={alpha}_theory'] = eigvals_transformed
        df_a.loc[:,f'alpha={alpha}_theory'] = eigvals_theory

ax.tick_params(axis='both', which='minor')
ax.legend(frameon=False)
ax.set_title('Eigenspectrum')
ax.set_xlabel('Index')
ax.set_ylabel('Value')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# SAVE DATA
df_a.to_csv(join(SOURCE_DATA_PATH, 'fig2a.csv'))

##### b, c #####

# Define parameters for the multi-modal Gaussian distribution
means = np.array([[3, 0], [-3, 0], [0, np.sqrt(3)]])
sigma = 0.25
cov = [[sigma, 0], [0, sigma]]  # Isotropic covariance (same for both components)
total_means = means.shape[0]

# Create the 2D Gaussian distributions
rv_dists = []
for idx in range(total_means):
    rv_dists.append(multivariate_normal(means[idx], cov))

# Number of samples per mode
# samples_per_mode = 7
samples_per_mode = 6
num_samples = total_means * samples_per_mode

# Sample all at once
samples = []
for idx in range(total_means):
    samples.append(rv_dists[idx].rvs(size=samples_per_mode))
samples = np.vstack(samples)

# Compute probability density functions
# Create grid for contour plot
x = np.linspace(-7, 7, 250)
y = np.linspace(-7, 7, 250)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

Z = 0
for idx in range(total_means):
    Z += 1/total_means * rv_dists[idx].pdf(pos)

# -----------------

a = 0
bandwidth1 = 1e-4
d = 2

qk_share = True
if qk_share:
    Q = W = samples

n = Q.shape[0]
# line_scale = 0.6 * n
# line_scale = 0.02 * n
line_scale = 0.15 * n

bandwidth1 = 1e-1
#thresh = 1e-12

c_node = 'grey'
# lstyle = '--'
lstyle = '-'

g_dists = pairwise_distances(samples, Y=None, metric='euclidean')


# SAVE DATA
columns = ['coordinate_1', 'coordinate_2']
df_b_coordinates = pd.DataFrame(columns=columns)
df_c_coordinates = pd.DataFrame(columns=columns)
df_b_interactions = pd.DataFrame(np.zeros((n,n)))
df_c_interactions = pd.DataFrame(np.zeros((n,n)))

for ii in range(n):
    g_dists[ii,ii] = 0

for bidx, alpha in enumerate(alphas_selected):
    K, D, K_tilde, D_tilde = get_markov_matrix(g_dists, alpha, bandwidth1, d, a)
    M = np.diag(D_tilde**(-1)) @ K_tilde

    print(f'alpha = {alpha}')
    print(f'M min: {M.min()}, max: {M.max()}')
    print(f'K min: {K.min()}, max: {K.max()}')
    print(f'K_tilde min: {K_tilde.min()}, max: {K_tilde.max()}')
    print('\n')

    # ax = axes[0,1-bidx + 1]
    ax = axes[0,bidx+1]
    # c_alpha = HYP_CMAP(HYP_CNORM(alpha))
    c_alpha = COLORS_ALPHA[round((alpha - 1)/0.2) + 1]

    if alpha < 2:
        # thresh = M.min()
        thresh = 3e-4

    for i in range(n):
        for j in range(n):
            if M[i, j] > thresh and i != j:
                # M[i, j] is query i attending to key j.
                # Opposite directions curve onto opposite sides of the edge.
                ax.add_patch(FancyArrowPatch(
                    Q[i], W[j], arrowstyle='-|>', mutation_scale=5,
                    connectionstyle='arc3,rad=0.12', color=c_alpha,
                    linewidth=line_scale / np.abs(np.log(M[i, j])),
                    linestyle=lstyle, shrinkA=2, shrinkB=2, zorder=1,
                ))

                # Export only the directed interactions actually drawn.
                if bidx == 0:
                    df_b_interactions.iloc[i,j] = M[i, j]
                elif bidx == 1:
                    df_c_interactions.iloc[i,j] = M[i, j]

    # c='#dd1c77'
    ax.scatter(Q[:, 0], Q[:, 1], label='Queries', lw=.25, c=c_node, edgecolors="k", s=MARKERSIZE, zorder=2)
    if bidx == 0:
        df_b_coordinates.loc[:,'coordinate_1'] = Q[:,0]
        df_b_coordinates.loc[:,'coordinate_2'] = Q[:,1]
    elif bidx == 1:
        df_c_coordinates.loc[:,'coordinate_1'] = Q[:,0]
        df_c_coordinates.loc[:,'coordinate_2'] = Q[:,1]

    if not qk_share:
        # c='#a8ddb5'
        ax.scatter(W[:, 0], W[:, 1], label='Keys', lw=.5, c=c_node,  edgecolors="k", s=MARKERSIZE, zorder=2)

    #ax.set_xlim([-1.2,1.2]);ax.set_ylim([-1.2,1.2])
    ax.set_title(rf'Interactions $(\alpha = {alpha})$')
    ax.set_xticklabels([]);ax.set_yticklabels([])
    ax.axis('off')
    # ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')

# SAVE DATA
df_b_coordinates.to_csv(join(SOURCE_DATA_PATH, 'fig2b_coordinates.csv')) 
df_b_interactions.to_csv(join(SOURCE_DATA_PATH, 'fig2b_interactions.csv')) 
df_c_coordinates.to_csv(join(SOURCE_DATA_PATH, 'fig2c_coordinates.csv')) 
df_c_interactions.to_csv(join(SOURCE_DATA_PATH, 'fig2c_interactions.csv')) 

# -----------------

# subfigure labels
ii_subfigs = 0
for i in range(nrows):
    for j in range(ncols):
        ax = axes[i,j]
        ax.text(-0.2, 1.15, rf"$\mathbf{{{ascii_lowercase[ii_subfigs]}}}$",
            transform=ax.transAxes, ha='left',  va='top',
            usetex=False) # , fontfamily='sans-serif'
        ii_subfigs += 1

# Adjust layout
plt.tight_layout(w_pad=3)

savedir = join('.droot', 'figs_dir')
if not isdir(savedir): makedirs(savedir)
fig_path = join(savedir, 'fna_and_spectrum_new.pdf')
plt.savefig(fig_path, bbox_inches='tight')

# plt.show()