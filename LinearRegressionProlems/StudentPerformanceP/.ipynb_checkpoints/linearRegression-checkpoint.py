import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv" , sep = ";")

#print(data.head())
print(data)
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
#print(data.head())
X = np.array(data.drop([predict] , 1))
y = np.array(data[predict])
X_train, X_test , y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)
linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)
acc = linear.score(X_test, y_test)
print(acc)

