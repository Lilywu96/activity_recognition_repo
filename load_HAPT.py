import os, path
import numpy as np
from pandas import Series

def Loading_HAPT(filename,data):
    x,y,z=np.loadtxt(filename,delimiter=' ',unpack=True)
    data['x'].append(x)
    print(data['x'])
