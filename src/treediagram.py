import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

#read iris dataset with two features
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
Y = iris.target

tree = DecisionTreeClassifier(max_depth = 4)
tree.fit(X, Y)

export_graphviz(tree,out_file='tree.dot',
        feature_names = ['petal length', 'petal width'])
