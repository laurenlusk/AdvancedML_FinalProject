import numpy as np
import scipy.io as sio
from sklearn import svm

def load_data():
    lteDictionary = sio.loadmat('LTEDataset_upSamp.mat', appendmat=True, variable_names='waveforms')
    lteData = lteDictionary['waveforms']

    nrDictionary = sio.loadmat('NRDataset.mat', appendmat=True, variable_names='waveforms')
    nrData = nrDictionary['waveforms']

    return lteData, nrData

def parse_data(lteData,nrData):
    #this entire section will break if the number of lte and nr signals are odd :)
    #create train and test data
    ptrain = .8
    ptest = 1-ptrain
    numTrain = int(round(len(lteData)*ptrain + len(nrData)*ptrain))
    numTest = int(round(len(lteData)*ptest + len(nrData)*ptest))
    halfTrain = int(numTrain/2)
    halfTest = int(numTest/2)

    #build training data
    x_train = np.zeros((numTrain,int(len(lteData[0]))))
    x_train[0:(halfTrain-1),:] = np.real(lteData[0:(halfTrain-1),:])
    x_train[halfTrain:(numTrain-1),:] = np.real(nrData[0:(halfTrain-1),:])

    y_train = np.zeros((numTrain,1))
    y_train[0:(halfTrain-1),:] = 0          #lte = 0
    y_train[halfTrain:(numTrain-1),:] = 1   #nr = 1

    #randomize train data in unison
    rng_state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(rng_state)
    np.random.shuffle(y_train)

    #build testing data
    x_test = np.zeros((numTest,int(len(lteData[0]))))
    x_test[0:(halfTest-1),:] = np.real(lteData[halfTest:(numTest-1),:])
    x_test[halfTest:(numTest-1),:] = np.real(nrData[halfTest:(numTest-1),:])

    y_test = np.zeros((numTest,1))
    y_test[0:(halfTest-1),:] = 0#lte = 0
    y_test[halfTest:(numTest-1),:] = 1#nr = 1

    #randomize test data in unison
    rng_state = np.random.get_state()
    np.random.shuffle(x_test)
    np.random.set_state(rng_state)
    np.random.shuffle(y_test)

    return x_train, y_train, x_test, y_test

def get_accuracy(predictions,actual):
  # calculates the accuracy
  diff = predictions - actual
  # counts the number of times the predictions matched the actual labels
  # or in other words, diff == 0
  diff = diff.size - np.count_nonzero(diff)
  # accuracy is the number correc/size of data set
  accuracy = diff/actual.size
  return accuracy

lteData, nrData = load_data()
x_train, y_train, x_test, y_test = parse_data(lteData,nrData)

clf = svm.SVC()
clf.fit(x_train,y_train)

predicted = np.array([])
for element in x_test:
    predicted = np.append(predicted, clf.predict([element]))

accuracy = get_accuracy(predicted,y_test.T)
print('Accuracy:', accuracy, '%')