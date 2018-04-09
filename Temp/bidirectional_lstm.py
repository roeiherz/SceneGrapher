from __future__ import print_function
import numpy
import os
from keras.preprocessing import sequence
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Masking

# cut texts after this number of words
# (among top max_features most common words)
maxlen = 32
batch_size = 32

print('Loading data...')

dataset_path = os.path.join('/data/engineCache/Deep/datasets/ccaau', 'lstm')
x_train = numpy.load(os.path.join(dataset_path,'x_train.npy'))
x_test = numpy.load(os.path.join(dataset_path,'x_test.npy'))
y_train = numpy.load(os.path.join(dataset_path,'y_train.npy'))
y_test = numpy.load(os.path.join(dataset_path,'y_test.npy'))

print(len(x_train), 'train sequences x')
print(len(x_test), 'test sequences x')

print(len(y_train), 'train sequences y')
print(len(y_test), 'test sequences y')

print("Pad sequences (samples x time)")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post', value=-1.)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post', value=-1.)
y_train = sequence.pad_sequences(y_train, maxlen=maxlen, padding='post', truncating='post', value=-1.)
y_test = sequence.pad_sequences(y_test, maxlen=maxlen, padding='post', truncating='post', value=-1.)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

model = Sequential()
model.add(Masking(mask_value=-1., input_shape=(maxlen, 2048)))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Dense(972, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

print('Train...')

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=100,
          validation_data=[x_test, y_test])
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(dataset_path,'accuracy.png'))
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(dataset_path,'loss.png'))

model.save(os.path.join(dataset_path,'model.h5'))
