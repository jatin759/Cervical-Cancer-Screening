import time
st = time.time()

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras import optimizers
from keras import regularizers
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

K.set_image_dim_ordering('th')
K.set_floatx('float32')

import pandas as pd
import numpy as np

np.random.seed(11)

trnd = np.load('trainnew.npy')
trnfile = np.load('traintrans.npy')

tstd = np.load('testnew.npy')
timname = np.load('test_idnew.npy')



def generate():
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, activation='relu', dim_ordering='th', input_shape=(3, 64 , 64)))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), dim_ordering='th'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(32, 4, 4, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1), dim_ordering='th'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='tanh', kernel_regularizer = regularizers.l2(0.01) ))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax' , kernel_regularizer = regularizers.l2(0.01)))

    model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

"""
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='th', input_shape=(3, 32 , 32)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.1))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid', kernel_regularizer = regularizers.l2(0.01) ))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax' , kernel_regularizer = regularizers.l2(0.01) ))
"""
    #model.compile(optimizer=opt_, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #return model


def sortim():

    clim = ImageDataGenerator(rotation_range=0.3, zoom_range=0.3)
    clim.fit(trnd)
    return clim


def modelprd():
    
    clim=sortim()

    #print(clim)
    model = generate()
    x_train,x_val_train,y_train,y_val_train = train_test_split(trnd,trnfile,test_size=0.4, random_state=11)

    #print(tr1)
    history = model.fit_generator(clim.flow(x_train,y_train, batch_size=15, shuffle=True), nb_epoch=250, samples_per_epoch=len(x_train), verbose=20, validation_data=(x_val_train, y_val_train))
    
    #print(history)

    estm = model.predict_proba(tstd)
    #print(estm)

    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    """
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

    return estm

def modelacc():

    pred=modelprd()
    df = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
    #print(np.shape(pred))
    
    ydash = {}
    for i in range(0,len(pred)):
        p = 0.0
        for j in range(0,3):
            if pred[i,j] > p:
                p = pred[i,j]
                mj = j+1
        #ydash.append(mj)
        ydash[timname[i]] = int(mj)

    #print(ydash)
            
    yor = {}
    with open("data.txt") as f:
        for line in f:
            (key,value) = line.split()
            yor[key] = int(value)

    #print(list(ydash))
    #print(len(list(ydash)))
    n = len(list(ydash))
    nc = 0
    for i in range(0,len(list(ydash))):
        if ydash[list(ydash)[i]] == yor[list(ydash)[i]] :
            nc = nc+1

    acc = float(nc)/n
    print('Accuracy obtained is :')
    print(acc*100 , "%")

    #print(yor)
    #print(ydash[4])

    #print(timname)
    #print(np.shape(timname))
    #yor = np.loadtxt('data.txt',usecols = range(2))
    #print(yor)
    #df['image_name'] = timname
    #df.to_csv('submission.csv', index=False)


if __name__ == '__main__':

    modelacc()

    end = time.time()
    #print(end-st)