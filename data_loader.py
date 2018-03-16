from __future__ import print_function

import numpy as np
import h5py
import os
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split

# For building model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten

import matplotlib.pyplot as plt




#def load all the data
def load_data_from(num):
    dir = 'project_datasets'
    data_name = 'A0' + str(num) + 'T_slice.mat'
    data_file = os.path.join(dir, data_name)
    data = h5py.File(data_file, 'r')
    X = np.copy(data['image'])
    X = X[:, 0:22, :]
    y = np.copy(data['type'])
    y = y[0, 0:X.shape[0]:1]

    return X, y

def load_data_all():
    dir = 'project_datasets'
    data_name = 'A01T_slice.mat'
    data_file = os.path.join(dir, data_name)
    data = h5py.File(data_file, 'r')
    X = np.copy(data['image'])
    X = X[:, 0:22, :]
    y = np.copy(data['type'])
    y = y[0, 0:X.shape[0]:1]

    for i in range(2, 10):

        data_name = 'A0' + str(i) + 'T_slice.mat'
        data_file = os.path.join(dir, data_name)
        cur_data = h5py.File(data_file, 'r')
        X_tmp = np.copy(cur_data['image'])[:, 0:22, :]
        X = np.concatenate((X, X_tmp), axis=0)
        y_tmp = np.copy(cur_data['type'])[0,0:X_tmp.shape[0]:1]
        y = np.concatenate((y, y_tmp), axis=0)
        y = np.asarray(y)
    return X, y

def one_sub_split(file_num, test_size, val_size):
    X, y = load_data_from(file_num)
    X, y = remove_NAN(X, y)
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_size)
    return X_train, X_val, X_test, y_train, y_val, y_test


def all_train_val_one_test(file_num, test_size, val_size):
    dir = 'project_datasets/'
    data_path = os.listdir(dir)
    X_list = []
    y_list = []
    for i in range(len(data_path)):
        X, y = load_data_from(i+1)
        X, y = remove_NAN(X, y)
        X_list.append(X)
        y_list.append(y)

    cur_X = X_list[file_num-1]
    cur_y = y_list[file_num-1]
    X_tmp, X_test, y_tmp, y_test = train_test_split(cur_X, cur_y, test_size=test_size, random_state=42)
    for i in range(len(X_list)):
        if i != file_num-1:
            X_tmp = np.concatenate((X_tmp, X_list[i]), axis=0)
            y_tmp = np.concatenate((y_tmp, y_list[i]), axis=0)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def remove_NAN(X, y):
    N, E, T = X.shape
    X = X.reshape(N, -1)
    y = y[~np.isnan(X).any(axis=1)]
    temp = X[~np.isnan(X).any(axis=1)]
    X = temp.reshape(temp.shape[0], E, T)
    return X, y



def dim_test():
    X, y = load_data_from(1)

    print(X.shape)
    print(y.shape)
    print(y[:10])

    X, y = load_data_all()
    print(X.shape)
    print(y.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = all_train_val_one_test(1, 50, 100)
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)


def hot_code(y):
    one_hot = [0] * 4
    one_hot[int(y - 769)] = 1
    return one_hot


def mini_batch(X, y, batch_size=50):
    N = X.shape[0]
    batch_mask = np.random.choice(N, batch_size)
    X_batch = X[batch_mask]
    y_batch = y[batch_mask]
    return X_batch, y_batch


def fit_stacked_lstm(X, y, X_val, y_val, batch_size, timesteps, epoch, nuerons):
    # reshape training and val X into [trials, timesteps, features]
    X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[2], X_val.shape[1])
    # Build LSTM
    model = Sequential()
    # model.add(LSTM(nuerons, batch_input_shape=(batch_size, timesteps, X.shape[2]), stateful=True))
    model.add(LSTM(nuerons, input_shape=(timesteps, X.shape[2]), return_sequences=True))
    # model.add(LSTM(nuerons, stateful=True))
    model.add(LSTM(nuerons, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the network
    history = model.fit(X, y, batch_size=batch_size, epochs=epoch, validation_data=(X_val, y_val))

    return model, history

def save_model(model, path):
    # path = str(name) + '.h5'
    model.save(path)

def load_model(path):
    from keras.models import load_model
    # path = str(name) + '.h5'
    model = load_model(path)
    return model

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()



if __name__ == '__main__':
    # (288, 22, 1000), (288,)
    X, y = load_data_from(1)

    # for subject1 remove one trial.
    # (287, 22, 1000), (287,)
    X, y = remove_NAN(X, y)

    N, E, T = X.shape

    y = y - 769

    # one-hot encoding for categorical cross-entropy classification.
    # (287, 4) ----- because it is needed when we use categorical_crossentropy.
    y = to_categorical(y, 4)

    # split into (N-100:50:50) for train, val, test sets. ---- (187, 22, 1000) (50, :,: ), (50, :, :)
    x_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=50)
    X_train, X_val, y_train, y_val = train_test_split(x_tmp, y_tmp, test_size=50)

    # Training parameters
    input_num = X.shape[1]
    batch_size = 12
    timesteps = 1000
    num_hidden = 50
    num_classes = 4
    num_epoch = 20

    # print(y_train.shape)
    # print(X.reshape(N, T, E).shape)
    # print(y.shape[0])

    model, history = fit_stacked_lstm(X_train, y_train, X_val, y_val, batch_size, timesteps, num_epoch, num_hidden)

    save_model(model, '2_lstm.h5')
    plot_acc(history)
    plot_loss(history)









