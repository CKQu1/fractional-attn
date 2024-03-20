import os
from os.path import join, normpath, isdir

def njoin(*args):
    return normpath(join(*args))

# for enumerating each instance of training
def get_instance(dir, s):
    if isdir(dir):
        instances = []
        dirnames = next(os.walk(dir))[1]
        if len(dirnames) > 0:
            for dirname in dirnames:        
                # len(os.listdir(njoin(dir, dirname))) > 0
                if s in dirname and "model=" in dirname and len(os.listdir(njoin(dir, dirname))) > 0:  # make sure file is non-empty
                    #try:        
                    for s_part in dirname.split(s):
                        if "model=" in s_part:
                            start = s_part.find("model=") + 6
                            end = s_part.find("_")
                            instances.append(int(s_part[start:end]))
                    #except:
                    #    pass       
            print(instances) 
            return max(instances) + 1 if len(instances) > 0 else 0
        else:
            return 0
    else:
        return 0