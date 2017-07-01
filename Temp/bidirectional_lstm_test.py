from __future__ import print_function
import numpy
import os
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.models import load_model

# cut texts after this number of words
# (among top max_features most common words)
maxlen = 32
batch_size = 128

print('Loading data...')

dataset_path = os.path.join('/data/engineCache/Deep/datasets/ccaau', 'lstm')
dataset_path_test = os.path.join('/data/engineCache/Deep/datasets/ccaau', 'lstm_test')
x_test = numpy.load(os.path.join(dataset_path_test, 'x_testing.npy'))
print(len(x_test), 'test sequences x')

print("Pad sequences (samples x time)")
x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post', value=-1.)
print('x_test shape:', x_test.shape)
model = load_model(os.path.join(dataset_path, 'model.h5'))

print('Test...')
predictions = model.predict(x_test, batch_size=batch_size, verbose=1)

numpy.save(os.path.join(dataset_path_test, 'testing_predictions.npy'), predictions)