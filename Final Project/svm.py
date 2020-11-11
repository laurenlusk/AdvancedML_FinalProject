from sklearn import svm
import numpy as np
import functions

def train_and_test_svm(x_train,y_train,x_test,y_test):
    """
    :param x_train,x_test: Ndarray (samples,features)
    :param y_train,y_test: Ndarray (samples,)
    :return: accuracy
    """
    # reshape labels to 1D arrays
    y_train = np.reshape(y_train, np.size(y_train))
    y_test = np.reshape(y_test, np.size(y_test))
    # create svm
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    # predict from test data
    predicted = np.array([])
    for element in x_test:
        predicted = np.append(predicted, clf.predict([element]))
    # calculate test accuracy
    accuracy = functions.get_accuracy(predicted, y_test)
    return accuracy

lteData = functions.load_data('LTEDataset_upSamp.mat', 'waveforms')
nrData = functions.load_data('NRDataset.mat', 'waveforms')
x_train, y_train, x_test, y_test = functions.parse_data(lteData, nrData)

accuracy = train_and_test_svm(x_train,y_train,x_test,y_test)
print('Accuracy:', accuracy*100, '%')
