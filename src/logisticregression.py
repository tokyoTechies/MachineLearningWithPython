import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.cross_validation import train_test_split

#read iris dataset w`ith two features
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
Y = iris.target


#split data into training set and testing set 
#70% of the data is training 30% of the data is testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3)


#use Logistic Regression Model
lr = LogisticRegression()
#train model
lr.fit(X_train, Y_train)


#plot decisionBoundry
X_min,X_max = X[:,0].min(),X[:,0].max()
padding = (X_max - X_min) / 10
X_max = X_max + padding

Y_min,Y_max =X[:,1].min(),X[:,1].max() 
padding = (Y_max - Y_min) / 10
Y_max = Y_max + padding

xs,ys = np.meshgrid(np.arange(0,X_max,0.02)
                   ,np.arange(0,Y_max,0.02))
classification = lr.predict(np.c_[xs.ravel(),ys.ravel()])
classification = classification.reshape(xs.shape)

#plot boundry
plt.pcolormesh(xs,ys,classification)

#plot data
plt.scatter(X[:50,0],X[:50,1],marker='x',label="flower 1")
plt.scatter(X[50:100,0],X[50:100,1],marker='+',label="flower 2")
plt.scatter(X[-50:,0],X[-50:,1],marker='*',label="flower 3")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
