import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.cm import get_cmap
from scipy.stats import vonmises
from string import ascii_lowercase

from constants import *
from mutils import njoin

# ----- Global plot settings -----

#cm_name = 'rainbow'
cm_name = 'turbo'

# method 1
cmap = get_cmap(cm_name)
norm = mpl.colors.Normalize(vmin=1, vmax=2)

# method 2
# cmap = mpl.colormaps[cm_name]
# alpha_intervals = np.linspace(1, 2, 21)
# alpha_colors = cmap(alpha_intervals)

def get_alpha_index(alpha, alpha_intervals):
    if alpha in alpha_intervals:
        aidx = np.where(alpha_intervals==alpha)[0][0]
    else:
        for aidx in range(len(alpha_intervals)-1):            
            if np.abs(alpha_intervals[aidx] - alpha) < np.abs(alpha_intervals[aidx+1] - alpha):
                break

    return aidx

def get_markov_matrix(C, alpha, bandwidth, d, a):

    #sphere_radius = ((np.pi**(1/d)-1)/np.pi)
    if alpha >= 2:
        K = np.exp(-(C/bandwidth**0.5)**(alpha/(alpha-1)))
    else:
        K = (1 + C/bandwidth**0.5)**(-d-alpha)

    D = K.sum(-1)  # row normalization
    K_tilde = np.diag(D**(-a)) @ K @ np.diag(D**(-a))
    D_tilde = K_tilde.sum(-1)

    #return np.diag(D_tilde**(-1)) @ K_tilde
    return K, D, K_tilde, D_tilde


a = 0
bandwidth1 = 1e-1
thresh = 1e-5

#loc = 0.5 * np.pi  # circular mean
loc1, loc2 = -np.pi/2, np.pi/2
kappa = 8  # concentration

# Sample 1 (non-uniform small)
sample_size = 6
sample1 = vonmises(loc=loc1, kappa=kappa).rvs(sample_size)
sample2 = vonmises(loc=loc2, kappa=kappa).rvs(sample_size)
sample_radians = np.concatenate([sample1, sample2])
sample_xys = np.stack([np.cos(sample_radians), np.sin(sample_radians)]).T

# Sample 2 (uniform large)
uniform_sample_size = 500
uniform_radians = np.linspace(0,2*np.pi,uniform_sample_size)
uniform_xys = np.stack([np.cos(uniform_radians), np.sin(uniform_radians)]).T

# Sample 1 (non-uniform large)
large_sample_size = uniform_sample_size
large_sample1 = vonmises(loc=loc1, kappa=kappa).rvs(int(large_sample_size/2))
large_sample2 = vonmises(loc=loc2, kappa=kappa).rvs(int(large_sample_size/2))
large_sample_radians = np.concatenate([large_sample1, large_sample2])
large_sample_xys = np.stack([np.cos(large_sample_radians), np.sin(large_sample_radians)]).T

nrows, ncols = 2, 3
figsize = (3*ncols,3*nrows)
fig, axs = plt.subplots(nrows,ncols,figsize=figsize,
                        sharex=False,sharey=False)        
axs = np.expand_dims(axs, axis=0) if axs.ndim == 1 else axs   

# PDF
x = np.linspace(-np.pi, np.pi, 1000)
vonmises_pdf1 = vonmises.pdf(x, loc=loc1, kappa=kappa)
vonmises_pdf2 = vonmises.pdf(x, loc=loc2, kappa=kappa)
final_pdf = 0.5 * vonmises_pdf1 + 0.5 * vonmises_pdf2

# ---------------------------------------- Row 1 ----------------------------------------

# PDF in radians
ax = axs[0,0]
ax.plot(x, final_pdf)
#ax.set_yticks(ticks)
#number_of_bins = int(np.sqrt(sample_size))
#ax.hist(sample_radians, density=True, bins=number_of_bins)
#ax.set_title("Cartesian plot")
ax.set_title("Bimodal von-Mises")
ax.set_xlim(-np.pi, np.pi)
ax.grid(True)

# Points and interactions on circle (non-uniform)
g_dists = np.arccos(sample_xys @ sample_xys.T)
for ii in range(g_dists.shape[0]):
    g_dists[ii,ii] = 0

qk_share = True
if qk_share:
    Q = sample_xys
    W = sample_xys

n = sample_xys.shape[0]
d = sample_xys.shape[1] - 1
scale = 1e-10

alphas1 = [1.2, 2]
for bidx, alpha in enumerate(alphas1):
    K, D, K_tilde, D_tilde = get_markov_matrix(g_dists, alpha, bandwidth1, d, a)
    M = np.diag(D_tilde**(-1)) @ K_tilde

    print(f'alpha = {alpha}')
    print(f'M min: {M.min()}, max: {M.max()}')
    print(f'K min: {K.min()}, max: {K.max()}')
    print(f'K_tilde min: {K_tilde.min()}, max: {K_tilde.max()}')
    print('\n')    

    ax = axs[0,bidx+1]
    c_alpha = cmap(norm(alpha))

    for i in range(n):
        for j in range(n):
            if M[i, j] > thresh:
                if bidx ==0:
                    ax.plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], c=c_alpha, linewidth=scale * M[i, j], zorder=1)
                elif bidx == 1:
                    ax.plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], c=c_alpha, linewidth=scale * M[i, j], zorder=1)
                else:
                    ax.plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], c=c_alpha, linewidth=scale * M[i, j], zorder=1)

    ax.scatter(Q[:, 0], Q[:, 1], label='Queries', lw=.25, c='#dd1c77', edgecolors="k", s=20, zorder=2)
    if not qk_share:
        ax.scatter(W[:, 0], W[:, 1], label='Keys', lw=.5, c='#a8ddb5',  edgecolors="k", s=20, zorder=2)

    ax.set_xlim([-1.2,1.2]);ax.set_ylim([-1.2,1.2])
    ax.set_xticklabels([]);ax.set_yticklabels([])
    ax.set_title(rf'$\alpha = {alpha}$')

# unit circle outline
for col in [1,2]:
    axs[0,col].plot(uniform_xys[:,0], uniform_xys[:,1], c='grey', 
                       linestyle='--', linewidth=1e-2)

# ---------------------------------------- Row 2 ----------------------------------------

alphas2 = [1.2,  2]
bandwidth2 = 1e-4

# Polar histograms
axs[1,0].remove()
axs[1,0] = ax = fig.add_subplot(nrows, ncols, 4, projection='polar')
ax.plot(x, final_pdf, label="PDF")
ax.set_yticks([0.5, 1])
ax.hist(large_sample_radians, density=True, bins=int(np.sqrt(large_sample_size)), label="Histogram")
#ax.set_title("Polar plot")
#ax.legend(bbox_to_anchor=(0.15, 1.06))

# Points and interactions on circle
#g_dists_2 = np.arccos(uniform_xys @ uniform_xys.T)  # uniform sampling
g_dists_2 = np.arccos(large_sample_xys @ large_sample_xys.T)  # non-uniform sampling
n = g_dists_2.shape[0]
for ii in range(n):
    g_dists_2[ii,ii] = 0

idxs = np.arange(1,large_sample_size+1)
idx_mid = int(n/2)
for bidx, alpha in enumerate(alphas2):

    c_alpha = cmap(norm(alpha))

    t = bandwidth2**(alpha/2)
    K, D, K_tilde, D_tilde = get_markov_matrix(g_dists_2, alpha, bandwidth2, d, a)

    ax = axs[1,1+bidx]
    #ax = axs[1,1]

    if a == 0:
        # removing non-uniform sampling
        a_ = 1
        # estimate of sampling density
        q_sample = ((2*np.pi*bandwidth2)**(-d/2)/n * np.exp(-g_dists_2**2/(2*bandwidth2))).sum(-1)
        K_tilde_ = np.diag(q_sample**(-a_)) @ K @ np.diag(q_sample**(-a_))
        D_tilde_ = K_tilde_.sum(-1)
        K_hat = np.diag(D_tilde_**(-0.5)) @ K_tilde_ @ np.diag(D_tilde_**(-0.5))

    eigvals, eigvecs = np.linalg.eigh(0.5*(K_hat + K_hat.T))

    # small to large
    eidx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[eidx]; eigvecs = eigvecs[:,eidx]
    eigvals = -1/t * np.log(eigvals)

    ax.plot(idxs, eigvals, c=c_alpha, label=rf'$\alpha = {{{alpha}}}$')  # eigvals    
    eigvals_theory = idxs**alpha  # eye guide
    eigvals_theory = eigvals_theory / eigvals_theory[idx_mid]
    eigvals_theory = eigvals_theory * eigvals[idx_mid] * 50
    ax.plot(idxs, eigvals_theory, c=c_alpha, alpha=0.5, linewidth=1, linestyle='--')

    ax.set_xlim([1, 500])
    #ax.set_ylim([1e2, 5e4])
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.legend()

ii = 0
for row in range(nrows):
    for col in range(ncols):  
        ax = axs[row,col]
        ax.text(
            0.0, 1.0, f'({ascii_lowercase[ii]})', transform=(
                ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
            fontsize='medium', va='bottom', fontfamily='sans-serif')       

        ii += 1

plt.savefig(njoin(FIGS_DIR, 'figure_fdm.pdf'))