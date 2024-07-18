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

from constants import *
from mutils import njoin

from schematic.fdm_utils import get_markov_matrix, plot_vmf_density, get_vmf_samples, uniform_sphere_rvs

# ----- Global plot settings -----

c_node = 'grey'
lstyle = '--'

# Self-attention
a = 0
bandwidth1 = 1e-1
#thresh = 1e-12
#thresh = 1e-10

# VMF
# mus = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0])[None,:]  # single
# mus = np.array([[-np.sqrt(0.5), -np.sqrt(0.5), 0],  # bimodal
#                [-1, 0, 0]])
mus = np.array([[0, -1, 0],  # bimodal
               [-1, 0, 0]])
kappa = 25

# Set seed, working: 10, 20
seed = 40
np.random.seed(seed)

# Sample 1
sample_size = 3
samples = get_vmf_samples(sample_size, mus, kappa)

large_sample_size = 1000
# Sample 3 (uniform large)
uniform_xyzs, count = uniform_sphere_rvs(large_sample_size)
# Sample 2 (non-uniform probabilitic large)
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

nrows, ncols = 2, 4
figsize = (3*ncols,3*nrows)
fig, axs = plt.subplots(nrows,ncols,figsize=figsize,
                        sharex=False,sharey=False)        
axs = np.expand_dims(axs, axis=0) if axs.ndim == 1 else axs   

# # PDF (bimodel)
# x = np.linspace(-np.pi, np.pi, 1000)
# vonmises_fisher_pdf1 = vonmises_fisher.pdf(x, loc=loc1, kappa=kappa)
# vonmises_fisher_pdf2 = vonmises_fisher.pdf(x, loc=loc2, kappa=kappa)
# final_pdf = 0.5 * vonmises_fisher_pdf1 + 0.5 * vonmises_fisher_pdf2

# ---------------------------------------- Row 1 ----------------------------------------

# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises_fisher.html

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

# ----- (a) -----
plot_vmf_density(fig, axs, [0,0], x, y, z, vertices, mus, kappa)

# ----- (b) -----
axs[0,1].remove()
axs[0,1] = fig.add_subplot(nrows, ncols, 2, projection='3d')
ax = axs[0,1]

ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, alpha=0.2)
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='k', s=5, alpha=1)
for mu_idx in range(mus.shape[0]):
    ax.scatter(mus[mu_idx][0], mus[mu_idx][1], mus[mu_idx][2], c='r', s=30, alpha=0.5)
ax.set_aspect('equal')
ax.view_init(azim=-130, elev=0)
#ax.axis('off')
#ax.set_title(rf"$\kappa={kappa}$")

# ----- (c -- d) -----
# Points and interactions on circle (non-uniform)
g_dists = np.arccos(samples @ samples.T)
for ii in range(g_dists.shape[0]):
    g_dists[ii,ii] = 0

# stereographic projection
qk_share = True
if qk_share:
    Q = W = np.stack([samples[:,0] / (1 - samples[:,2]), samples[:,1] / (1 - samples[:,2])]).T    

n = samples.shape[0]
#d = samples.shape[1]
d = samples.shape[1] - 1
#scale = 3 * n
scale = 0.75 * n

alphas1 = [1.2, 2]
for bidx, alpha in enumerate(alphas1):
    K, D, K_tilde, D_tilde = get_markov_matrix(g_dists, alpha, bandwidth1, d, a)
    M = np.diag(D_tilde**(-1)) @ K_tilde

    print(f'alpha = {alpha}')
    print(f'M min: {M.min()}, max: {M.max()}')
    print(f'K min: {K.min()}, max: {K.max()}')
    print(f'K_tilde min: {K_tilde.min()}, max: {K_tilde.max()}')
    print('\n')    

    ax = axs[0,bidx+2]
    #c_alpha = cmap(norm(alpha))
    c_alpha = HYP_CMAP(HYP_CNORM(alpha))

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


# ---------------------------------------- Row 2 ----------------------------------------

alphas2 = [1.2, 2]
#alphas2 = [1.2, 1.6, 2]
#bandwidth2 = 1e-4
bandwidth2 = 1e-5

# ----- (d) Polar histograms -----
# axs[1,0].remove()
# axs[1,0] = ax = fig.add_subplot(nrows, ncols, 4, projection='polar')
# ax.plot(x, final_pdf, label="PDF")
# ax.set_yticks(pdf_yticks)
# ax.hist(large_sample_radians, density=True, bins=int(np.sqrt(large_sample_size)), label="Histogram")
# ax.set_title("Polar plot")
# ax.legend(bbox_to_anchor=(0.15, 1.06))

dist_types = ['Non-uniform', 'Uniform']
#ds_iis = [0,1]
#ds_iis = [1,0]
ds_iis = [1, 0]

for bidx, ds_ii in enumerate(ds_iis):

    dist_type = dist_types[ds_ii]

    dot = all_xyzs[ds_ii] @ all_xyzs[ds_ii].T
    # clip values
    dot[dot>=1] = 1
    dot[dot<=-1] = -1
    g_dists_2 = np.arccos(dot)

    n = g_dists_2.shape[0]
    for ii in range(n):
        g_dists_2[ii,ii] = 0

    idxs = np.arange(1,large_sample_size+1)
    idx_mid = int(n/2)
    for alpidx, alpha in enumerate(alphas2):

        #c_alpha = cmap(norm(alpha))
        c_alpha = HYP_CMAP(HYP_CNORM(alpha))

        t = bandwidth2**(alpha/2)
        K, D, K_tilde, D_tilde = get_markov_matrix(g_dists_2, alpha, bandwidth2, d, a)

        # ---------- Eigvals ----------

        ax = axs[1,bidx]
        #ax = axs[1,1]

        if a == 0:
            # removing initial sampling effect
            a_ = 1
            # estimate of sampling density
            gaussian_hk = (2*np.pi*bandwidth2)**(-d/2)/n * np.exp(-g_dists_2**2/(2*bandwidth2))
            q_sample = gaussian_hk.sum(-1)
            K_tilde_ = np.diag(q_sample**(-a_)) @ K @ np.diag(q_sample**(-a_))
            D_tilde_ = K_tilde_.sum(-1)
            K_hat_ = np.diag(D_tilde_**(-0.5)) @ K_tilde_ @ np.diag(D_tilde_**(-0.5))
            K_hat_sym_ = 0.5*(K_hat_ + K_hat_.T)

            eigvals_, eigvecs_ = np.linalg.eigh(K_hat_sym_)
            eigvecs_ = np.diag(D_tilde_**(-0.5)) @ eigvecs_

            # keep initial sampling density


# ----- (e -- f) -----

            # eigvals
            eidx = np.argsort(eigvals_)[::-1]
            eigvals_ = eigvals_[eidx]; eigvecs_ = eigvecs_[:,eidx]
            eigvals_ = -1/t * np.log(eigvals_)
            ax.plot(idxs, eigvals_, c=c_alpha, label=rf'$\alpha = {{{alpha}}}$')    
            # eye guide
            power = alpha if alpha <= 2 else 2
            eigvals_theory = idxs**power  
            # eigvals_theory = eigvals_theory / eigvals_theory[idx_mid]
            # eigvals_theory = eigvals_theory * eigvals[idx_mid] * 10
            ax.plot(idxs, eigvals_theory, c=c_alpha, alpha=0.5, linewidth=1, linestyle='--')

            ax.set_xlim([1, n])
            if d == 1:
                ax.set_ylim([1, 1e6])
            elif d == 2:
                ax.set_ylim([1e-11, 1e6])

            #ax.set_ylim([1e2, 5e4])
            ax.set_xscale('log'); ax.set_yscale('log')
            #ax.legend()

            # original markov matrix np.diag(D_tilde**(-1)) @ K_tilde
            K_hat = np.diag(D_tilde**(-1/2)) @ K_tilde @ np.diag(D_tilde**(-1/2))
            K_hat_sym = 0.5*(K_hat + K_hat.T)
            eigvals, eigvecs = np.linalg.eigh(K_hat_sym)
            eigvecs = np.diag(D_tilde**(-0.5)) @ eigvecs

            eidx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[eidx]; eigvecs = eigvecs[:,eidx]

            # ---------- Eigvecs ----------
            if dist_type == 'Non-uniform':
                #eidx1, eidx2 = large_sample_size-1, large_sample_size-2
                eidx1, eidx2 = 0, 1
                #eidx1, eidx2 = 1, 2
                ax = axs[1,alpidx+2]
                #ax.plot(large_sample_radians, np.exp(t * eigvals[eidx]) * eigvecs[:,eidx], c=c_alpha)
                ax.scatter(eigvecs[:,eidx1], eigvecs[:,eidx2], c=c_alpha, s=1)
                #ax.scatter(all_radians[ds_ii][:,0], eigvecs[:,eidx1-1], c=c_alpha, s=1)

# ----- plot settings -----

axs[1,1].legend(frameon=False)
for bidx, ds_ii in enumerate(ds_iis):
    axs[1,bidx].set_title(dist_types[ds_ii])
    #axs[1,bidx+2].set_title(dist_types[ds_ii])

ii = 0
for row in range(nrows):
    for col in range(ncols):  
        ax = axs[row,col]
        if '3d' in ax.name:
            ax.text2D(
                0.0, 1.0, f'({ascii_lowercase[ii]})', transform=(
                    ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                fontsize='medium', va='bottom', fontfamily='sans-serif')    
        else:
            ax.text(
                0.0, 1.0, f'({ascii_lowercase[ii]})', transform=(
                    ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
                fontsize='medium', va='bottom', fontfamily='sans-serif')       

        ii += 1

FIGS_DIR = njoin(FIGS_DIR, 'schematic')
if not isdir(FIGS_DIR): makedirs(FIGS_DIR)
plt.savefig(njoin(FIGS_DIR, 'figure_fdm_3d.pdf'))
print(f'Figure saved in {FIGS_DIR}')