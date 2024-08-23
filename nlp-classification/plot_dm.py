import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from constants import FIGS_DIR
from mutils import njoin

from scipy.spatial.distance import cdist

#rc = {"pdf.fonttype": 42, 'text.usetex': True, 'text.latex.preview': True}
#plt.rcParams.update(rc)

"""
For plotting the attentiong weights generated from fractional diffusion map.
"""

qk_share = True

def get_markov_matrix(C, alpha, bandwidth, d):

    if alpha >= 2:
        K = np.exp(-(C/bandwidth**0.5)**(alpha/(alpha-1)))
    else:
        K = (1 + C/bandwidth**0.5)**(-d-alpha)

    D = K.sum(-1)  # row normalization
    K_tilde = np.diag(D**(-1)) @ K @ np.diag(D**(-1))
    D_tilde = K_tilde.sum(-1)

    #return np.diag(D_tilde**(-1)) @ K_tilde
    return K, D, K_tilde, D_tilde

alphas = np.arange(1,2.1,0.5)
bandwidth = 1

n = 32  # must be even
if not qk_share:
    n = int(n/2)
d = 2  # dimension
seed = 4
rng = np.random.RandomState(seed)
#thresh = 10**(-12)
thresh = 10**(-20)
X = 0.3 * rng.randn(n//2, d)
Y = 0.3 + 0.3 * rng.rand(n//2, d)
Q = 20 * np.concatenate((X, Y), axis=0)

if qk_share:
    W = Q
else:
    X = 0.3 * rng.rand(n//2, d) + np.asarray([-0.2, 0.3])
    Y = 0.5 + 0.3 * rng.rand(n//2, d) + np.asarray([-0.2, -0.7])
    W = 20 * np.concatenate((X, Y), axis=0)

C_tilde = cdist(Q, W)

l = 0.7
#scale = 3 * n
#scale = 1.5 * n
scale = .2 * n
c = ['darkblue']
titles = ['No normalization', 'Softmax', 'Sinkhorn']
fig, ax = plt.subplots(2, len(alphas), figsize=(10 * l, 5 * l))

for bidx, alpha in enumerate(alphas):
    K, D, K_tilde, D_tilde = get_markov_matrix(C_tilde, alpha, bandwidth, d)
    M = np.diag(D_tilde**(-1)) @ K_tilde
    print(f'M min: {M.min()}, max: {M.max()}')
    print(f'K min: {K.min()}, max: {K.max()}')
    print(f'K_tilde min: {K_tilde.min()}, max: {K_tilde.max()}')
    print('\n')

    # ---------- Row 1 ----------

    for i in range(n):
        for j in range(n):
            if M[i, j] > thresh:
                if bidx ==0:
                    ax[0,bidx].plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], c='k', linewidth=0.3 * scale * M[i, j], zorder=1)

                elif bidx == 1:
                    ax[0,bidx].plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], c='k', linewidth=0.3 * scale * M[i, j],
                                    zorder=1)
                else:
                    ax[0,bidx].plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], c='k', linewidth=0.3 * scale * M[i, j], zorder=1)

    ax[0,bidx].scatter(Q[:, 0], Q[:, 1], label='Queries', lw=.25, c='#dd1c77', edgecolors="k", s=20, zorder=2)
    if not qk_share:
        ax[0,bidx].scatter(W[:, 0], W[:, 1], label='Keys', lw=.5, c='#a8ddb5',  edgecolors="k", s=20, zorder=2)

    # ---------- Row 2 ----------   

    #ax[1,bidx].hist(M.flatten())
    ax[1,bidx].hist(K.flatten(), 50)
    #ax[1,bidx].hist(K.flatten()[M.flatten() < 1], 50)
    #ax[1,bidx].hist(K.flatten()[M.flatten() < 0.95], 50)

    for ridx in range(ax.shape[0]):
        #ax[ridx,bidx].set_xticklabels(()); ax[ridx,bidx].set_yticklabels(())    
        pass

    ax[0,bidx].set_title(rf'$\alpha = {alpha}$')
    ax[1,bidx].set_xlim([0,0.3])

fig.tight_layout()
plt.savefig(njoin(FIGS_DIR,f'fig1-qk_share={qk_share}-seed={seed}.pdf'))    





