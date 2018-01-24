import gc
import nndata
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.recurrent import LSTM
from keras import regularizers
from keras.callbacks import ModelCheckpoint

def create_raw_model(nchan, nclasses, trial_length=960, l1=0):
    """
    CNN model definition
    """
    input_shape = (trial_length, nchan, 1)
    model = Sequential()
    model.add(Conv2D(40, (30, 1), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="same", input_shape=input_shape))
    model.add(Conv2D(40, (1, nchan), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="valid"))
    model.add(AveragePooling2D((30, 1), strides=(15, 1)))
    model.add(Flatten())
    model.add(Dense(80, activation="relu"))
    model.add(Dense(nclasses, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    return model

def create_raw_model2(nchan, nclasses, trial_length=960, l1=0, full_output=False):
    """
    CRNN model definition
    """
    input_shape = (trial_length, nchan, 1)
    model = Sequential()
    model.add(Conv2D(40, (30, 1), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="same", input_shape=input_shape))
    model.add(Conv2D(40, (1, nchan), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="valid"))
    model.add(AveragePooling2D((5, 1), strides=(5, 1)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(40, activation="sigmoid", dropout=0.25, return_sequences=full_output))
    model.add(Dense(nclasses, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
    return model

def fit_model(model, X, y, train_idx, test_idx, input_length=50, batch_size=32, epochs=30, steps_per_epoch=1000, callbacks=None):    
    gc.collect()
    return model.fit_generator(
        nndata.crossval_gen(X,y, train_idx, input_length, batch_size),
        validation_data=nndata.crossval_test(X, y, test_idx, input_length),
        steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks                          
    )
