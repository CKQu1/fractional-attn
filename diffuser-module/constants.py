import os
from path_setup import njoin

# path for saving all data
DROOT = njoin(os.getcwd(), "droot")
if not os.path.isdir(DROOT): os.makedirs(DROOT)

# singularity path
SPATH = "../built_containers/FaContainer_v2.sif"

# project names
PROJECTS = ["ddl"]