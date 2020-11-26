import numpy as np
import functions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense  # ,Dropout


# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error,mean_absolute_error
# from tensorflow.keras.callbacks import EarlyStopping

def reshape(x):
    return np.reshape(x, (np.shape(x)[0], np.shape(x)[1], 1))


lteData = functions.load_data('LTEDataset_upSamp.mat', 'waveforms')
nrData = functions.load_data('NRDataset.mat', 'waveforms')
x_train, y_train, x_test, y_test = functions.parse_data(lteData, nrData)

x_train = np.expand_dims(x_train,axis=-1)
y_train = np.expand_dims(y_train,axis=-1)

print('loaded data')

model = Sequential()
print('created model')
# add n LSTM layer
n = 5
epoch = 100
batchSz = 10

for i in range(n):
    model.add(LSTM(units=np.shape(x_train)[1], return_sequences=True))
model.add(Dense(2))
print('layers added')
model.compile(loss='mean_squared_error', optimizer='adam')
print('model compiled')
model.fit(x_train, y_train, epochs=epoch, batch_size=batchSz)
print('model fitted to training data')

predicted = np.array([])
for element in x_test:
    predicted = np.append(predicted, model.predict([element]))
print('predicted values')
# calculate test accuracy
accuracy = functions.get_accuracy(predicted, y_test)
print('Accuracy:', accuracy * 100, '%')
