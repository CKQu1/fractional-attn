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
#alphas = np.arange(1,2.1,0.5)
alphas = [1.5,2]
bandwidth = 1

# https://math.stackexchange.com/questions/56784/generate-a-random-direction-within-a-cone/182936#182936
def generate_circle_on_sphere(r, n):
    z = np.random.uniform(0.5,1,n)
    phis = np.random.uniform(0,2*np.pi,n)
    #return np.vstack([np.sqrt(r - z**2)*np.cos(phis), np.sqrt(r - z**2)*np.sin(phis), np.repeat(z,n)]).T
    return np.vstack([np.sqrt(r - z**2)*np.cos(phis), np.sqrt(r - z**2)*np.sin(phis), z]).T

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

# General settings
r = 1  # radius
n = 32
seed = 4
rng = np.random.RandomState(seed)
thresh = 10**(-12)
#thresh = 10**(-20)

# Cluster 1
# thetas1 = rng.uniform(0,2*np.pi,n)
# phis1 = rng.uniform(0,2*np.pi,n)
# X = r * np.sin(phis1) * np.cos(thetas1)
# Y = r * np.sin(phis1) * np.sin(thetas1)
# Z = r * np.cos(phis1)
#Q = np.vstack([X,Y,Z]).T

Q = generate_circle_on_sphere(r, n)
d = Q.shape[1]

if qk_share:
    W = Q
else:
    pass

C_tilde = cdist(Q, W)

l = 0.7
#scale = 3 * n
#scale = 1.5 * n
scale = .2 * n
c = ['darkblue']
titles = ['No normalization', 'Softmax', 'Sinkhorn']
nrows, ncols = 2, len(alphas)
fig, axs = plt.subplots(nrows, ncols, figsize=(10*l, 5*l))

for bidx, alpha in enumerate(alphas):
    K, D, K_tilde, D_tilde = get_markov_matrix(C_tilde, alpha, bandwidth, d)
    M = np.diag(D_tilde**(-1)) @ K_tilde
    print(f'M min: {M.min()}, max: {M.max()}')
    print(f'K min: {K.min()}, max: {K.max()}')
    print(f'K_tilde min: {K_tilde.min()}, max: {K_tilde.max()}')
    print('\n')

    # ---------- Row 1 ----------

    axs[0,bidx].remove()
    axs[0,bidx] = fig.add_subplot(nrows,ncols,bidx+1,projection='3d')
    ax = axs[0,bidx]
    
    for i in range(n):
        for j in range(n):
            if M[i, j] > thresh:
                if bidx ==0:
                    ax.plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], [Q[i, 2], W[j, 2]], c='k', linewidth=0.3 * scale * M[i, j], zorder=1)
                elif bidx == 1:
                    ax.plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], [Q[i, 2], W[j, 2]], c='k', linewidth=0.3 * scale * M[i, j],
                                    zorder=1)
                else:
                    ax.plot([Q[i, 0], W[j, 0]], [Q[i, 1], W[j, 1]], [Q[i, 2], W[j, 2]], c='k', linewidth=0.3 * scale * M[i, j], zorder=1)

    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], label='Queries', lw=.25, c='#dd1c77', edgecolors="k", s=20, zorder=2)
    if not qk_share:
        ax.scatter(W[:, 0], W[:, 1], W[:, 2], label='Keys', lw=.5, c='#a8ddb5',  edgecolors="k", s=20, zorder=2)

    ax.set_xlim([-1,1]);ax.set_ylim([-1,1]);ax.set_zlim([-1,1])
    ax.set_xticklabels([]);ax.set_yticklabels([]);ax.set_zticklabels([])
    ax.set_title(rf'$\alpha = {alpha}$')

    # ---------- Row 2 ----------   

    ax = axs[1,bidx]
    #ax.hist(M.flatten())
    ax.hist(K.flatten(), 50)
    #ax.hist(K.flatten()[M.flatten() < 1], 50)
    #ax.hist(K.flatten()[M.flatten() < 0.95], 50)

    # for ridx in range(axs.shape[0]):
    #     axs[ridx,bidx].set_xticklabels(()); axs[ridx,bidx].set_yticklabels(())    
    #     pass
    
    #ax.set_xlim([0,0.3])

fig.tight_layout()
plt.savefig(njoin(FIGS_DIR,f'fig1-3d-qk_share={qk_share}-seed={seed}.pdf'))    
