import numpy as np
import pandas as pd
import random as rand
from numpy.linalg import inv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

def loadData():
  # loads all of the training data into a list of dataframes
  XtrainFile = []
  YtrainFile = []
  for r in range(10):
    XtrainFile.append('trainData'+str(r+1)+'.csv')
    YtrainFile.append('trainLabels'+str(r+1)+'.csv')

  X = []
  Y = []
  for f in XtrainFile:
    x = pd.read_csv(f, header=None)
    x.insert(0, 'X0', 1)
    X.append(x)
  for f in YtrainFile:
    # label 5 = 0, label 6 = 1
    y = pd.read_csv(f, header=None)
    y = y.replace([5,6],[0,1])
    Y.append(y)
  return X,Y


def split_data(X, Y):
    # combines all the train files into a single dataframe
    X_train = pd.DataFrame()
    Y_train = pd.DataFrame()
    X_val = pd.DataFrame()
    Y_val = pd.DataFrame()
    for frame in range(7):
        X_train = X_train.append(X[frame], ignore_index=True)
        Y_train = Y_train.append(Y[frame], ignore_index=True)
    for frame in range(7, 10):
        X_val = X_val.append(X[frame], ignore_index=True)
        Y_val = Y_val.append(Y[frame], ignore_index=True)
    return X_train, Y_train, X_val, Y_val


def EuclidDist(test_inst, train):
    # calculates the euclidian distance
    dist = np.linalg.norm(train - test_inst, axis=1)
    return pd.DataFrame(dist)


def getPrediction(X_train, X_test_inst, Y_train, k):
    # finds the predicted label
    prediction = []
    # find the euclidian distance between the test instance and training data
    distances = EuclidDist(X_test_inst, X_train)
    distances = pd.concat([distances, Y_train], axis=1, sort=False)
    distances.columns = ['dist', 'Y']
    distances = distances.sort_values(by='dist')

    # gets labels of k nearest neighbors
    neighbors = distances.head(k)['Y']

    # finds the mode of labels and applies it as the prediction
    # randomly breaks ties
    mode = neighbors.mode()
    mode = mode.tolist()
    # if there is only one mode, the predicted label is that mode
    if len(mode) == 1:
        prediction = mode[0]
    else:
        # breaks ties randomly
        r = rand.randint(0, len(mode) - 1)
        prediction = mode[r]
        # returns predicted label
    return prediction


def average_predictions(df):
    mode = df.mode(axis=1)
    prediction = []
    for i in range(df.shape[0]):
        sub = mode.iloc[i]
        sub = sub[sub.notnull()]
        if len(sub) == 1:
            prediction.append(sub[0])
        else:
            # breaks ties randomly
            r = rand.randint(0, len(sub) - 1)
            prediction.append(sub[r])
    return pd.DataFrame(prediction)


def bootstrap(val, n):
    x = val.sample(n=len(val), replace=True)
    y = x[x.shape[1] - 1]
    del x[x.shape[1] - 1]
    return x, y


def getAccuracy(predictions, Y_test):
    # calculates the accuracy
    diff = predictions - Y_test
    # counts the number of times the predictions matched the actual labels
    # or in other words, diff == 0
    diff = diff.apply(pd.value_counts).fillna(0)
    # accuracy is the number correc/size of data set
    accuracy = diff[0][0] / Y_test.size
    return accuracy


X,Y = loadData()
x_train,y_train,x_val,y_val = split_data(X,Y)
train = pd.concat([x_train,y_train],axis=1,ignore_index=True,sort=False)

k = 19
accuracy = []

for baseLearn in range(2,26):
  print(baseLearn)
  for b in range(baseLearn):
    x,y = bootstrap(train,x_train.shape[0])
    predictions = []
    # iterate through test data and find nearest neighbors
    for index, row in x_val.iterrows():
      predictions.append(getPrediction(x,row,y,k))
    # gets predictions for every base learners
    if b == 0:
      pred = pd.DataFrame(predictions)
    else:
      pred[b] = predictions
  # majority votes
  pred = average_predictions(pred)
  # add accuracy
  accuracy.append(getAccuracy(pred,y_val))
  print(accuracy)
print('finished')