import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import GridSearchCV
import pandas as pd

#import mnist dataset
digits = datasets.load_digits()
x = digits.data
y = digits.target

#split training and testing such that testing contains
#300 samples
train_x,test_x = x[:-300],x[-300:]
train_y,test_y = y[:-300],y[-300:]

#initialize svm model
svm = svm.SVC()

#create a dictionary of possible parameters
parameters = {'kernel': ('linear','rbf'),
        'C':[0.01,0.1,1,10,100], 'gamma':[0.001,0.1,1,10]}

#initialize a grid-search with the model and 
#potential paramters
clf = GridSearchCV(svm,parameters)
clf.fit(train_x,train_y)

#output the result of the grid search as a dataframe
df = pd.DataFrame(clf.cv_results_)
#print the index that contains the parameters of the best
#result
print(df.iloc[clf.best_index_,:])

