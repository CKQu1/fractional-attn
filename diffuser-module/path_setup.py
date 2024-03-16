import os
from os.path import join, normpath

project_ls = ["ddl"]
droot = join(os.getcwd(), "droot")
if not os.path.isdir(droot): os.makedirs(droot)

def njoin(*args):
    return normpath(join(*args))

# returns None or the string itself
def none_or_self(s):
    if s == None or s.lower() == "none":
        return None
    else: 
        return s 

# converts s to bool
def str_to_bool(s):
    if isinstance(s,bool):
        return s
    elif isinstance(s,str):
        return literal_eval(s)

def matching_subdirs(root_path, *args):
    subdirs = []
    for subdir in os.listdir(root_path):
        s_include = True
        for s in args:
            s_include = s_include and (s in subdir)
        if s_include == True:
            subdirs.append(njoin(root_path, subdir))
    return subdirs    