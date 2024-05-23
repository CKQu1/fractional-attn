import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from os import makedirs
from os.path import isdir,  isfile

from manifold import Manifold

alphas = [1.2, 2]
final_time = 2
n_steps = 4000
sphere_manifold = Manifold(manifold='sphere', 
                           radius_sphere=1, 
                           final_time=final_time, 
                           n_steps=n_steps,
                           alphas=alphas,
                           plt_interactive=False)

sphere_simdata = sphere_manifold.simulate_brownian_sphere()

#sphere_manifold.plot_brownian_sphere(sphere_simdata,steptoplot=[n_steps],show_axes=True,)                          
sphere_manifold.plot_brownian_sphere(sphere_simdata,
                                     has_title=False)

# save_dir = os.path.join('.droot', 'figs')
# if not isdir(save_dir): makedirs(save_dir)
# plt.savefig(os.path.join(save_dir, 'bm_diffusion.pdf'))