import os
from os.path import join

droot = join(os.getcwd(), "droot")
if not os.path.isdir(droot): os.makedirs(droot)