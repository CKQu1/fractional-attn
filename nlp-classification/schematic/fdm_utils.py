import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from scipy.stats import vonmises, vonmises_fisher
from string import ascii_lowercase

def get_markov_matrix(C, alpha, bandwidth, d, a):

    #sphere_radius = ((np.pi**(1/d)-1)/np.pi)
    if alpha >= 2:
        alpha_hat = alpha/(alpha-1)
        K = np.exp(-(C/bandwidth**0.5)**alpha_hat)
    else:
        K = (1 + C/bandwidth**0.5)**(-d-alpha)

    D = K.sum(-1)  # row normalization
    K_tilde = np.diag(D**(-a)) @ K @ np.diag(D**(-a))
    D_tilde = K_tilde.sum(-1)

    #return np.diag(D_tilde**(-1)) @ K_tilde
    return K, D, K_tilde, D_tilde

def plot_vmf_density(fig, axs, axidxs, x, y, z, vertices, mus, kappa):

    nrows, ncols = fig.axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec().get_geometry()

    #cmap_name = 'inferno'
    #cmap_name = 'viridis'
    cmap_name = 'plasma'
    cmap = get_cmap(cmap_name)
    axs[axidxs[0],axidxs[1]].remove()
    axs[axidxs[0],axidxs[1]] = fig.add_subplot(nrows, ncols, axidxs[0] * ncols + axidxs[1] + 1, projection='3d')    
    ax = axs[axidxs[0],axidxs[1]]   

    vmf = vonmises_fisher(mus[0], kappa)
    pdf_values = vmf.pdf(vertices) * 1/mus.shape[0]
    for mu in mus[1:]:
        vmf = vonmises_fisher(mu, kappa)
        pdf_values += vmf.pdf(vertices) * 1/mus.shape[0]
    pdfnorm = Normalize(vmin=pdf_values.min(), vmax=pdf_values.max())
    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    facecolors=cmap(pdfnorm(pdf_values)),
                    linewidth=0)
    ax.set_aspect('equal')
    ax.view_init(azim=-130, elev=0)
    #ax.axis('off')
    #ax.set_title(rf"$\kappa={kappa}$")    

def get_vmf_samples(sample_size, mus, kappa):
    vmf = vonmises_fisher(mus[0], kappa)
    samples = vmf.rvs(sample_size)
    for mu in mus[1:]:
        vmf = vonmises_fisher(mu, kappa)
        samples = np.concatenate([samples, vmf.rvs(sample_size)])
    return samples

# https://scicomp.stackexchange.com/questions/29959/uniform-dots-distribution-in-a-sphere
def uniform_sphere_rvs(n):
    r = 1
    alp = 4.0*np.pi*r*r/n
    d = np.sqrt(alp)
    m_nu = int(np.round(np.pi/d))
    d_nu = np.pi/m_nu
    d_phi = alp/d_nu
    count = 0
    xyzs = []
    for m in range (0,m_nu):
        nu = np.pi*(m+0.5)/m_nu
        m_phi = int(np.round(2*np.pi*np.sin(nu)/d_phi))
        for n in range (0,m_phi):
            phi = 2*np.pi*n/m_phi
            xp = r*np.sin(nu)*np.cos(phi)
            yp = r*np.sin(nu)*np.sin(phi)
            zp = r*np.cos(nu)
            xyzs.append([xp,yp,zp])
            count = count +1    

    return np.array(xyzs), count