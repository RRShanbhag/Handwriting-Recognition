import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')

import os
import cv2
import scipy.io as sio

num_classes, X_train, X_test, y_train, y_test = None, None, None, None, None


def loadData():
    global num_classes, X_train, X_test, y_test, y_train
    mat = sio.loadmat('/home/dumma/Desktop/Machine_Learning/Project/Lists/English/Hnd/lists_var_size.mat',
                      mat_dtype=True, squeeze_me=False, chars_as_strings=True)
    if not os.path.exists('mat_decode'):
        os.mkdir('mat_decode')

    for i in range(9):
        if mat['list'][0, 0][i].dtype == '<U13' or mat['list'][0, 0][i].dtype == '<U24':
            np.savetxt('mat_decode/%d.txt' % i, mat['list'][0, 0][i], fmt='%s')
        else:
            np.savetxt('mat_decode/%d.txt' % i, mat['list'][0, 0][i])

        data = open('mat_decode/%d.txt' % i).read().replace('\n\n', '\n')
        open('mat_decode/%d.txt' % i, 'w').write(data)

    ALLnames = np.genfromtxt('mat_decode/1.txt', dtype=str)
    ALLlabels = np.genfromtxt('mat_decode/0.txt')
    classlabels = np.genfromtxt('mat_decode/2.txt', dtype=int)
    TSTind = np.genfromtxt('mat_decode/5.txt', delimiter=' ')

    # TSTind[-1,:] = np.ones(TSTind.shape[1]) * 3409
    VALind = np.genfromtxt('mat_decode/6.txt', delimiter=' ')
    TRNind = np.genfromtxt('mat_decode/8.txt', delimiter=' ')
    # print(TRNind.shape)
    # print(ALLnames[int(TRNind[90, 1])][11:-11])

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    # print(ALLlabels.shape)

    for i in range(TRNind.shape[0]):
        img = cv2.imread('/home/dumma/Desktop/Machine_Learning/Project/English/Hnd/' + ALLnames[
            int(TRNind[i, TRNind.shape[1] - 1])] + '.png', 0).astype(int) / 255
        X_train.append(img)
        y_train.append(ALLlabels[int(TRNind[i, TRNind.shape[1] - 1])])
    # print(ALLnames.shape)
    # print(max(TSTind[:, TSTind.shape[1] - 1]))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    num_classes = y_train.shape[0]
    print(num_classes)
    for i in range(TSTind.shape[0]):
        img = cv2.imread('/home/dumma/Desktop/Machine_Learning/Project/English/Hnd/' + ALLnames[
            int(TSTind[i, TSTind.shape[1] - 1])] + '.png', 0).astype(int) / 255
        X_test.append(img)
        y_test.append(ALLlabels[int(TSTind[i, TSTind.shape[1] - 1])])
        # y_train = ALLlabels[int(TRNind[:,TRNind.shape[1]-1])]
        # y_train.append(ALLlabels[])
        # if i == 0:
        #     X_train = img
        # else:
        #     X_train = np.append(X_train, img, axis=0)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    # cv2.imshow('cv2.WINDOW_NORMAL',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(img.shape)
    # cv2.imshow('cv2.WINDOW_NORMAL', X_train[19000])
    # f cv2.waitKey(0) == 27:cv 2.destroyAllWindows()
    X_train = X_train.reshape(X_train.shape[0], 1, 96, 128).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 96, 128).astype('float32')
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # print(X_train.shape)
    # print(y_train.shape)


def baseline_model():
    global num_classes, X_train, X_test, y_test, y_train
    # create model
    model = Sequential()
    model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 96, 128), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# if __name__=='__main__' :

if __name__ == '__main__':
    loadData()
    # build the model
    model = baseline_model()
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
