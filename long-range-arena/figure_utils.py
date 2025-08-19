import numpy as np
from string import ascii_lowercase
from matplotlib.transforms import ScaledTranslation

def matrixify_axs(axs, nrows, ncols):
    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)
        if ncols == 1:
            axs = np.expand_dims(axs, axis=1)
    elif nrows > 1 and ncols == 1:
        axs = np.expand_dims(axs, axis=1)   
    return axs

def label_axs(fig, axs):

    total_figs = 0   
    nrows, ncols = axs.shape     
    for row in range(nrows):
        for col in range(ncols):
            axs[row,col].text(
        0.0, 1.0, f'({ascii_lowercase[total_figs]})', transform=(
            axs[row,col].transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
        va='bottom')
            total_figs += 1    