import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

#read wine dataset with two features
wines = pd.read_csv("wine.csv")
X = wines.iloc[:,[11,6]].values
Y = wines.iloc[:,0].values

#split data into training set and testing set 
#70% of the data is training 30% of the data is testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.30)

#find mesh boundries
X_min, X_max = X[:,0].min(), X[:,0].max()
padding = (X_max - X_min) / 10
X_min, X_max = X_min - padding, X_max + padding

Y_min, Y_max = X[:,1].min(), X[:,1].max() 
padding = (Y_max - Y_min) / 10
Y_min,Y_max = Y_min - padding, Y_max + padding

#create subplots
_,plots = plt.subplots(2,5)

#plot subplots
for i in range(5):
    for j in range(2):
        #use i and j to map to vales [1..10]
        depth = 2 * (i * 2 + j + 1)
        #use decision Tree model
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(max_depth = depth)
        #train model
        tree.fit(X_train, Y_train)
        
        #plot classification boundries
        xs,ys = np.meshgrid(np.arange(X_min, X_max, 0.02)
                   , np.arange(Y_min, Y_max,0.02))
        classification = tree.predict(np.c_[xs.ravel(), ys.ravel()])
        classification = classification.reshape(xs.shape)
        plots[j,i].pcolormesh(xs, ys, classification)

        #plot data
        plots[j,i].scatter(X[:59,0],X[:59,1], marker='x',
                 label="wine 1",c olor='w')
        plots[j,i].scatter(X[59:129,0],X[59:129,1], marker='+',
                 label="wine 2", color='r')
        plots[j,i].scatter(X[-49:,0],X[-49:,1], marker='*',
                 label="wine 3",color='b')
        plots[j,i].set_title("Depth: " + str(depth) + " Acc: " + 
                "{:.2f}".format(
                    metrics.accuracy_score(tree.predict(X_test), Y_test)))
plt.show()
