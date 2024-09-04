import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from os.path import isdir, isfile, join
from os import makedirs
from scipy.stats import levy_stable

from matplotlib.transforms import ScaledTranslation
from string import ascii_lowercase

# ----- Global settings -----
display = False
# set seed
np.random.seed(seed=4)
COLORS = ['tab:blue', 'tab:orange']
MARKERSIZE = 5
# ---------------------------

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

# Parameters
n_steps = 1000
delta_t = 1e-2
alpha = 1.5  # Lévy process stability parameter (0 < alpha <= 2)

# Generate processes
bm = brownian_motion(n_steps, delta_t)
lp = levy_process(n_steps, alpha, delta_t)

# Plotting
nrows, ncols = 1, 3
#ratio = 4.5
ratio = 3
fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * ratio, ratio))

step = 10
# Brownian Motion plot
axs[0].plot(bm[:, 0], bm[:, 1], c=COLORS[0], alpha=0.5, label='Brownian Motion')
# [axs[0].plot(bm[step*i:step*(i+1), 0], bm[step*i:step*(i+1), 1], 
#              alpha=0.5, c=COLORS[0]) for i in range(int(n_steps/step))]
axs[0].plot(bm[0,0], bm[0,1], c='g', marker='o', markersize=MARKERSIZE,  mfc='none')
axs[0].plot(bm[-1,0], bm[-1,1], c='r', marker='x', markersize=MARKERSIZE)

axs[0].set_title('Brownian Motion')
axs[0].set_xlabel(f'$x_1$')
axs[0].set_ylabel(f'$x_2$')

#ax[0].legend(frameon=False)

# Lévy Process plot
axs[1].plot(lp[:, 0], lp[:, 1], c=COLORS[1], alpha=0.5, label='Lévy Process (α={})'.format(alpha))
# [axs[1].plot(lp[step*i:step*(i+1), 0], lp[step*i:step*(i+1), 1], 
#              alpha=0.5, c=COLORS[1]) for i in range(int(n_steps/step))]  
# alpha=np.min([0.5 + 0.5/bm.shape[0]*i, 1])
# alpha=0.5 + 0.5/n_steps*i

axs[1].plot(lp[0,0], lp[0,1], c='g', marker='o', markersize=MARKERSIZE, mfc='none')
axs[1].plot(lp[-1,0], lp[-1,1], c='r', marker='x', markersize=MARKERSIZE)

axs[1].set_title('Lévy Process')
axs[1].set_xlabel(f'$x_1$')
axs[1].set_ylabel(f'$x_2$')
#ax[1].legend(frameon=False)

lim = abs(np.array([bm.max(), bm.min(), lp.max(), lp.min()])).max()
for i in range(2):    
    axs[i].set_xlim([-lim, lim]); axs[i].set_ylim([-lim, lim])

# Histogram/PDF
# bins = 200
# ax[2].hist(bm[:,0], bins, alpha=0.5, label='Brownian Motion')
# ax[2].hist(lp[:,0], bins, alpha=0.5, label='Lévy Process (α={})'.format(alpha))

mu, sigma = 0, 1
x_size = 1000
x = np.linspace(mu - 20*sigma, mu + 20*sigma, x_size)
#x = np.linspace(mu - 50*sigma, mu + 50*sigma, x_size)
y1 = stats.norm.pdf(x, mu, sigma)
axs[2].plot(x, y1, c=COLORS[0], label=rf'$\alpha = 2$')

y2 = levy_stable.pdf(x, alpha=alpha, beta=0, loc=0, scale=1)
axs[2].plot(x, y2, c=COLORS[1], label=rf'$\alpha = {alpha}$')

# inset
ratios = [2/3,3/4]
axin = axs[2].inset_axes([0.7, 0.7, 0.25, 0.25])
axin.tick_params(
    axis='both',          # changes apply to the x-axis
    #which='both',      # both major and minor ticks are affected
    #bottom=False,      # ticks along the bottom edge are off
    #top=False,         # ticks along the top edge are off
    #labelbottom=False
    labelsize=5
    )
axin.plot(x[int(x_size*ratios[0]):int(x_size*ratios[1])], y1[int(x_size*ratios[0]):int(x_size*ratios[1])])
axin.plot(x[int(x_size*ratios[0]):int(x_size*ratios[1])], y2[int(x_size*ratios[0]):int(x_size*ratios[1])])
axin.set_xscale('log'); axin.set_yscale('log')

#ax[2].set_xscale('log'); ax[2].set_yscale('log')
axs[2].set_xlim([-10,10])
axs[2].set_ylim([1e-5,6e-1])
axs[2].set_title('Probability density')

# legends
axs[0].plot([], [], c=COLORS[0], label='Brownian Motion')
axs[1].plot([], [], c=COLORS[1], label='Lévy Process (α={})'.format(alpha))
axs[2].legend(loc='upper left', frameon=False)

# subplot labels
for ii in range(3):
    ax = axs[ii]
    ax.text(0.0, 1.0, f'({ascii_lowercase[ii]})', transform=(
        ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
    va='bottom', fontfamily='sans-serif')  # fontsize='medium',  

plt.tight_layout()
if display:
    plt.show()
else:
    savedir = join('.droot', 'figs_dir')
    if not isdir(savedir): makedirs(savedir)
    fig_path = join(savedir, '2d_diffusion.pdf')
    plt.savefig(fig_path)