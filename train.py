import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
import pandas as pd

df = pd.read_csv('train.csv')
x_train = df.iloc[:28000, 1].values
train_y = df.iloc[:28000, 0].values
x_test = df.iloc[28000:, 1].values
test_y = df.iloc[28000:, 0].values

X_train = list([] for i in range(len(x_train)))  # create empty 2d list
for i in range(len(x_train)):  # get train set of x
    X_train[i] = x_train[i].split()

# similarly for test set
X_test = list([] for i in range(len(x_test)))  # create empty 2d list
for i in range(len(x_test)):  # get train set of x
    X_test[i] = x_test[i].split()

X_train = np.array(X_train, 'float32')
train_y = np.array(train_y, 'float32')
X_test = np.array(X_test, 'float32')
test_y = np.array(test_y, 'float32')
print(type(X_train[0][9]))

print(X_train.shape)
print(train_y.shape)
print(test_y.shape)
print(X_test.shape)

X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

num_features = 64
num_labels = 7
batch_size = 64
epochs = 30
width = 48
height = 48

X_train = X_train.reshape(X_train.shape[0], width, height, 1)

X_test = X_test.reshape(X_test.shape[0], width, height, 1)

train_y = np_utils.to_categorical(train_y, num_classes=num_labels)
test_y = np_utils.to_categorical(test_y, num_classes=num_labels)

print('----------------------------------------------------------------------------')

model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(num_features, (3, 3), activation='relu'))
model.add(Conv2D(num_features, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
model.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(2 * 2 * 2 * 2 * num_features, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2 * 2 * 2 * 2 * num_features, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation='softmax'))
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
model.fit(X_train, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, test_y),
          shuffle=True)

model.summary()

print('============')
fer_json = model.to_json()
with open('fer.json', 'w') as json_file:
    json_file.write(fer_json)
model.save_weights('fer.h5')
