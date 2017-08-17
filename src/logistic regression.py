import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

cur_w2 = 0
cur_w1 = 0
cur_w0 = 1

alpha  = 0.1
precision = 0.1
previous_step_size = 100000

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning'
                 '-databases/iris/iris.data',header=None)
df.iloc[:,4] = np.where(df.iloc[:,4] == 'Iris-setosa', 1, 0)
df = df.head(100)

fig = plt.figure()

def logistic(x) :
      return 1 / (1 + math.exp(-x))

while previous_step_size > precision:
      prev_w0 = cur_w0
      prev_w1 = cur_w1
      prev_w2 = cur_w2
      
      sum_w0 = 0
      sum_w1 = 0
      sum_w2 = 0
      
      for row in df.itertuples():
            difference = row[4] - logistic(prev_w2*row[0] + prev_w1*row[2] + prev_w0)
            sum_w0 += difference
            sum_w1 += row[2] * difference
            sum_w2 += row[0] * difference

      cur_w0 = prev_w0 + (alpha * sum_w0)
      cur_w1 = prev_w1 + (alpha * sum_w1)
      cur_w2 = prev_w2 + (alpha * sum_w2)

      previous_step_size = np.sqrt((prev_w0 - cur_w0) ** 2 +
                                   (prev_w1 - cur_w1) ** 2 +
                                   (prev_w2 - cur_w2) ** 2)
      print(previous_step_size)
      
plt.scatter(df.iloc[:50,0].values , df.iloc[:50,2].values , c='r')
plt.scatter(df.iloc[-50:,0].values ,df.iloc[-50:,2].values, c='b')
plt.plot(range(-1,2),[(cur_w0 + i*cur_w1) / - cur_w2 for i in range(-1,2)])
plt.show()
