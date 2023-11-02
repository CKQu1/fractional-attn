import os
from os.path import join, normpath

project_ls = ['ddl']
droot = join(os.getcwd(), "droot")
if not os.path.isdir(droot): os.makedirs(droot)

def njoin(*args):
    return normpath(join(*args))