import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn import datasets
from sklearn.cross_validation import train_test_split

#read iris dataset with two features
iris = datasets.load_iris()
X = np.zeros((200,2))
X[:100,:] = 1 + np.random.randn(100,2) * 0.5 
X[-100:,:] = 2 + np.random.randn(100,2) * 0.5
Y = np.zeros(200)
Y[:100] = 0
Y[-100:] = 1

#split data into training set and testing set 
#70% of the data is training 30% of the data is testing
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3)


#use SVM Model
svm = svm.SVC(kernel='rbf',gamma=200,C=1.0)
#train model
svm.fit(X_train,Y_train)


#plot decisionBoundry
xs,ys = np.meshgrid(np.arange(-1,4,0.01)
                   ,np.arange(-1,4,0.01))
classification = svm.predict(np.c_[xs.ravel(),ys.ravel()])
classification = classification.reshape(xs.shape)


#plot boundry
plt.pcolormesh(xs,ys,classification)


#plot data
plt.scatter(X[:100,0],X[:100:,1],marker='x',label="flower 1")
plt.scatter(X[-100:,0],X[-100:,1],marker='+',label="flower 2")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
