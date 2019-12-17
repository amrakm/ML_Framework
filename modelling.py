import os
import logging


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import joblib


def split_training_data(X, y):

    '''
    split data into training / validation set
    '''

    X_train, X_test, y_train, y_test = train_test_split(
                                                X, y,
                                                test_size=0.33, random_state=2019)
    return X_train, X_test, y_train, y_test

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop

# define base model
def nn_model(input_dim):
    # create model
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, kernel_initializer='normal'))
    
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def fit_predictive_model(X_train, y_train, saving_path, epochs=300, batch_size=5000):

    saved_model_path = os.path.join(saving_path,'fitted_model.h5')
    model = nn_model(input_dim= X_train.shape[1])
    opt = RMSprop(lr=1e-3)

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    mc = ModelCheckpoint(saved_model_path, monitor='val_loss', mode='min', verbose=0, save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    # fit model
    model.compile(loss='mse', optimizer= opt, metrics=['mse','mae'])
    training_log = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,  verbose=1, validation_split=0.2, callbacks=[es, mc])
    
    return model, training_log


def load_fitted_model(saving_path):

    saved_model_path = os.path.join(saving_path,'fitted_model.h5')

    assert os.path.exists(saved_model_path), 'fitted file does not exist in experiment folder'

    return keras.models.load_model(saved_model_path)


def make_prediction(X, saving_path):

    loaded_model = load_fitted_model(saving_path)
    predicted = loaded_model.predict(X)

    return predicted
