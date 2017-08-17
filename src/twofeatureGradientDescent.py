import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it

#set instance variables
cur_w2 = 1
cur_w1 = 1
cur_w0 = 1
previous_step_size = 100000

#set hyper-parameters
alpha  = 0.005
precision = 0.000001

#read and normalize data
df = pd.read_excel("housingprices.xlsx")
df = (df - df.mean()) / (df.max() - df.min())

#execute while loop until limit of convergence
while previous_step_size > precision:

      #update previous iteration variables
      prev_w0 = cur_w0
      prev_w1 = cur_w1
      prev_w2 = cur_w2

      #set summation values to 0
      sum_w0 = 0
      sum_w1 = 0
      sum_w2 = 0
      
      #get the gradient based on the graident descent rules
      for row in df.itertuples() :
            difference = row[4] - (prev_w2 * row[3] + prev_w1 * row[1] + prev_w0)
            sum_w0 += difference
            sum_w1 += row[1] * difference
            sum_w2 += row[3] * difference
            
      #update the current value scaled to the learning rate (alpha)
      cur_w0 = prev_w0 + (alpha * sum_w0)
      cur_w1 = prev_w1 + (alpha * sum_w1)
      cur_w2 = prev_w2 + (alpha * sum_w2)

      #measure the cartesian distance of the update
      previous_step_size = np.sqrt((prev_w0 - cur_w0) ** 2 +
                                   (prev_w1 - cur_w1) ** 2 +
                                   (prev_w2 - cur_w2) ** 2)

#plot the graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = range(-1,2)
y = range(-1,2)
x,y = np.meshgrid(x,y)
z = cur_w2*y + cur_w1*x + cur_w0
ax.plot_surface(x,y,z,alpha=0.5)
ax.scatter(df.iloc[:,0],df.iloc[:,2],df.iloc[:,3],c='r')
plt.show()
