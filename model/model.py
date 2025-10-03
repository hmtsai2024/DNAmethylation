#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from keras import models
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Activation, Input
from keras.callbacks import EarlyStopping
from keras.models import Model


def CNNmodel(x_train1, x_val1, x_test1,
             y_train, y_val, y_test,
             x_train2=None, x_val2=None, x_test2=None):
    """
    CNN model for single or merged inputs.

    If x_train2 / x_val2 / x_test2 are provided, merges them with x_train1 etc.
    Otherwise runs single-modality CNN.

    Returns:
        mse_train, mse_val, mse_test, pred_train, pred_val, pred_test, epochs, model
    """

    # Merge if second dataset is provided
    if x_train2 is not None:
        x_train = pd.merge(x_train1, x_train2, left_index=True, right_index=True)
        x_val   = pd.merge(x_val1, x_val2, left_index=True, right_index=True)
        x_test  = pd.merge(x_test1, x_test2, left_index=True, right_index=True)
    else:
        x_train, x_val, x_test = x_train1, x_val1, x_test1

    nfeatures = x_test.shape[1]
    batch_size = 256
    filters1 = 1024
    kernel = 50
    stride = 50
    dense1 = 512
    act = 'linear'

    # Reshape
    x_train_np = x_train.to_numpy().reshape(x_train.shape[0], nfeatures, 1)
    x_val_np   = x_val.to_numpy().reshape(x_val.shape[0], nfeatures, 1)
    x_test_np  = x_test.to_numpy().reshape(x_test.shape[0], nfeatures, 1)

    # Build model (functional API, works for both single + merge)
    input_layer = Input(shape=(nfeatures, 1), name='input')
    x = Conv1D(filters=filters1, kernel_size=kernel, strides=stride,
               use_bias=True, padding='same')(input_layer)
    x = Activation(act)(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Flatten()(x)
    x = Dense(dense1, use_bias=True)(x)
    x = Activation(act)(x)
    output = Dense(y_train.shape[1], use_bias=True)(x)
    output = Activation(act)(output)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    # Train
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='min')
    history = model.fit(
        x_train_np, y_train,
        validation_data=(x_val_np, y_val),
        batch_size=batch_size,
        epochs=500,
        shuffle=True,
        callbacks=[es],
        verbose=2
    )

    # Evaluate
    cost = model.evaluate(x_test_np, y_test, verbose=2)

    mse_train = history.history['loss'][es.stopped_epoch]
    mse_val   = history.history['val_loss'][es.stopped_epoch]
    mse_test  = cost
    epochs    = es.stopped_epoch

    # Predictions
    pred_train = pd.DataFrame(model.predict(x_train_np, batch_size=batch_size, verbose=0))
    pred_val   = pd.DataFrame(model.predict(x_val_np, batch_size=batch_size, verbose=0))
    pred_test  = pd.DataFrame(model.predict(x_test_np, batch_size=batch_size, verbose=0))

    return mse_train, mse_val, mse_test, pred_train, pred_val, pred_test, epochs, model

