import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it

#create instance variables
cur_w1 = 0
cur_w0 = 1
previous_step_size = 100000

#intialize hyper-parameters
alpha  = 0.05
precision = 0.0001

#read and normalize data
df = pd.read_excel("housingprices.xlsx")
df = (df - df.mean()) / (df.max() - df.min())

#create a figure to be drawn on 
fig = plt.figure()

#create range for mesh plot:
#np.arange(x,y,d) = [x, x+d, x+2d, ..., y]
X = np.arange(-1,1.2,0.05)
Y = np.arange(-1,1,0.05)

#let values contain the cost for each value of the meshplot
values = []
for (w1,w0) in it.product(X,Y):
      sum = 0
      #sum the squared error for each row of the data-table
      for row in df.itertuples():
            sum += (w1*row[1] + w0 - row[4]) ** 2
      values.append((w0,w1,sum))

#let scatter contain the cost for each iteration of GD
scatter = []

#run gradient descent until limit of convergence
while previous_step_size > precision:

      #update values to fit values of previous iteration
      prev_w0 = cur_w0
      prev_w1 = cur_w1
    
      sum_w0 = 0
      sum_w1 = 0
    
      #update values of weights based on update rule
      for row in df.itertuples() :
            difference = (row[4] - (prev_w1 * row[1] + prev_w0))
            sum_w0 += difference
            sum_w1 += row[1] * difference
            
      cur_w0 = prev_w0 + (alpha * sum_w0)
      cur_w1 = prev_w1 + (alpha * sum_w1)

      #calculate cost of current iteration of GD
      cost = 0
      for row in df.itertuples() :
            cost += (row[4] - (cur_w1 * row[1] + cur_w0)) ** 2
      scatter.append((cur_w0,cur_w1,cost))

      #calculate cartesian distance between this step and previous step
      previous_step_size = np.sqrt((prev_w0 - cur_w0) ** 2 +
                                   (prev_w1 - cur_w1) ** 2 )
#unzip the values of the tuple
#zip*(x) :: [(x,y,z)] -> [x],[y],[z]
w0,w1,cost = zip(*values)
#plott the costs for all values in range
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(w0,w1,cost,alpha=0.4)

#unzip iterations of gradient descent
w0,w1,cost = zip(*scatter)
#plot iterations
ax.scatter(w0,w1,cost,c='r')
plt.show()

#plot scatter plot showing line of best fit

#plot every data point
plt.scatter(df.iloc[:,0],df.iloc[:,3])
#get x and y values for line of best fit:
# y = w1*x + w0
xs = range(-1,2)
ys = [cur_w1 * x + cur_w0 for x in xs]
plt.xlabel(list(df)[0])
plt.ylabel(list(df)[3])
plt.plot(xs,ys)
plt.show()

#plot how the cost decreases for every iteration of
#gradient descent
plt.plot(range(len(cost)),cost)
plt.xlabel("iteration")
plt.ylabel("error")
plt.show()
