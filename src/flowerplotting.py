import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning'
                 '-databases/iris/iris.data',header=None)

#values gets rid off meta-data (index, coloumn) and returns
#a numpy array
y = df.iloc[0:150,4].values
x = df.iloc[0:150,[0,2,3]].values

#first 50 are sentosa, 50-100 are versicolor, last 50 are virginica
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x[:50,0],x[:50,1],x[:50,2],
            c='r', marker='o', label='setosa')
ax.scatter(x[50:100,0],x[50:100,1],x[50:100,2],
            c='b', marker='x', label='versicolor')
ax.scatter(x[-50:,0], x[-50:,1],x[-50:,2],
            c='g', marker='*', label='virginica')

plt.show()

