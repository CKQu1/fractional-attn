import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from os.path import isdir, isfile, join
from os import makedirs
from scipy.stats import levy_stable

# Function to generate 2D Brownian motion
def brownian_motion(n_steps, delta_t):
    delta_w = np.sqrt(delta_t) * np.random.normal(size=(n_steps, 2))
    w = np.cumsum(delta_w, axis=0)
    return w

# Function to generate 2D Lévy process (SαS)
def levy_process(n_steps, alpha, delta_t):
    delta_l = levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=1, 
                              size=(n_steps, 2)) * (delta_t ** (1/alpha))
    l = np.cumsum(delta_l, axis=0)
    return l

display = False

# Parameters
n_steps = 1000
delta_t = 1e-2
alpha = 1.5  # Lévy process stability parameter (0 < alpha <= 2)

# Generate processes
bm = brownian_motion(n_steps, delta_t)
lp = levy_process(n_steps, alpha, delta_t)

# Plotting
nrows, ncols = 1, 3
ratio = 4.5
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * ratio, ratio))

# Brownian Motion plot
ax[0].plot(bm[:, 0], bm[:, 1], label='Brownian Motion')
ax[0].set_title('Brownian Motion')
ax[0].set_xlabel(f'$x_1$')
ax[0].set_ylabel(f'$x_2$')
ax[0].legend(frameon=False)

# Lévy Process plot
ax[1].plot(lp[:, 0], lp[:, 1], label='Lévy Process (α={})'.format(alpha), color='orange')
ax[1].set_title('Lévy Process')
ax[1].set_xlabel(f'$x_1$')
ax[1].set_ylabel(f'$x_2$')
ax[1].legend(frameon=False)

lim = abs(np.array([bm.max(), bm.min(), lp.max(), lp.min()])).max()
for i in range(2):    
    ax[i].set_xlim([-lim, lim]); ax[i].set_ylim([-lim, lim])

# Histogram/PDF
# bins = 200
# ax[2].hist(bm[:,0], bins, alpha=0.5, label='Brownian Motion')
# ax[2].hist(lp[:,0], bins, alpha=0.5, label='Lévy Process (α={})'.format(alpha))

mu, sigma = 0, 1
x = np.linspace(mu - 10*sigma, mu + 10*sigma, 500)
y1 = stats.norm.pdf(x, mu, sigma)
ax[2].plot(x, y1, label='Gaussian')

y2 = levy_stable.pdf(x, alpha=alpha, beta=0, loc=0, scale=1)
ax[2].plot(x, y2, label='Levy')

#ax[2].set_xscale('log'); ax[2].set_yscale('log')
ax[2].set_xlim([-10,10])
ax[2].set_ylim([1e-5,6e-1])
ax[2].legend(frameon=False)
ax[2].set_title('Probability density')

plt.tight_layout()
if display:
    plt.show()
else:
    savedir = join('.droot', 'figs_dir')
    if not isdir(savedir): makedirs(savedir)
    fig_path = join(savedir, '2d_diffusion.pdf')
    plt.savefig(fig_path)