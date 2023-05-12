import numpy as np
from os.path import join as pjoin
filename = 'index.txt'
path = './dataset/Mixamo'

with open(pjoin(path, filename), "w") as f:
    for i in range(1, 4098):
        string = str(i) + '.npy'
        f.write(string)
        f.write("\n")
    