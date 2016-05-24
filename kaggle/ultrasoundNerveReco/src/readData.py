import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df_train = pd.read_csv('../input/train_masks.csv')

pixels = df_train['pixels'].values
print(pixels[0])
p1 = [] # Run-start pixel locations
p2 = [] # Run-lengths
p3 = [] # Number of data points per image

for p in pixels:
    x = str(p).split(' ')
    i = 0
    for m in x:
        if i % 2 == 0:
            p1.append(m)
        else:
            p2.append(m)
        i += 1
        
