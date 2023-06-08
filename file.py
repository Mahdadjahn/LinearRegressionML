import pandas
import numpy
from sklearn import linear_model
from sklearn.metrics import r2_score

read_csv = pandas.read_csv("Amlak.csv")
regr = linear_model.LinearRegression()

data_train = numpy.random.rand(len(read_csv)) < 0.7

train = read_csv[data_train]
test = read_csv[~data_train]

train_x = numpy.asanyarray(train[["meterage"]])
train_y = numpy.asanyarray(train[["price"]])

test_x = numpy.asanyarray(test[["meterage"]])
test_y = numpy.asanyarray(test[["price"]])

regr.fit(train_x,train_y)

preforr2score = regr.predict(test_x)
print(r2_score(test_y,preforr2score))

print(regr.predict([[220]])[0][0])
