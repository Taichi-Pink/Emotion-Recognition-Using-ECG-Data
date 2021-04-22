from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from tensorflow import keras
import utils
import os
import numpy as np

def supervised_model_wesad1(x_tr_feature_dim, y_tr_dim, lr_super=0.001, hidden_nodes=512, dropout=0.25, L2=0):
    input_dimension     = x_tr_feature_dim
    output_dimension    = y_tr_dim
    
    model = keras.models.Sequential()
    model.add(layers.Dense(hidden_nodes, input_dim=input_dimension, activation='relu', kernel_regularizer = keras.regularizers.l2(L2)))
    model.add(layers.Dense(hidden_nodes, activation='relu', kernel_regularizer = keras.regularizers.l2(L2)))
    model.add(layers.Dense(output_dimension))
    model.add(layers.Activation('softmax'))
    
    op = keras.optimizers.Adam(lr=lr_super)
    model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])
    model.summary()
    
    return  model

def supervised_model_wesad2(x_tr_feature_dim, y_tr_dim, lr_super=0.001, hidden_nodes=512, dropout=0.25, L2=0):
    n_timesteps         = x_tr_feature_dim
    output_dimension    = y_tr_dim
    n_features          = 1
    
    model = keras.models.Sequential()
    model.add(layers.LSTM(hidden_nodes//2, return_sequences=True, input_shape=(n_timesteps,  n_features)))
    model.add(layers.Dropout(dropout))
    model.add(layers.LSTM(hidden_nodes//4, return_sequences=True))
    model.add(layers.Dropout(dropout))
    model.add(layers.LSTM(hidden_nodes//4, return_sequences=True))
    model.add(layers.Dropout(dropout))
    model.add(layers.LSTM(hidden_nodes//8, return_sequences=True))
    model.add(layers.Dropout(dropout))
    model.add(layers.LSTM(hidden_nodes//16))
    model.add(layers.Dense(output_dimension, activation='softmax'))
    op = keras.optimizers.Adam(lr=lr_super)
    model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])

    model.summary()
    return  model


def supervised_model_wesad4(x_tr_feature_dim, y_tr_dim, lr_super=0.001, hidden_nodes=512, dropout=0.5, L2=0):
    input_dimension     = x_tr_feature_dim
    output_dimension    = y_tr_dim
    n_steps, n_length, n_features   = 10, 256, 1
    
    input_  = keras.Input((None, n_length, n_features))
    out     = layers.TimeDistributed(layers.Conv1D(filters = 64, kernel_size = 3, strides = 1, activation='relu'))(input_)
    out     = layers.TimeDistributed(layers.Conv1D(filters = 64, kernel_size = 3, strides = 1, activation='relu'))(out)
    out     = layers.TimeDistributed(layers.Dropout(dropout))(out)
    out     = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(out)
    out     = layers.TimeDistributed(layers.Flatten())(out)
    out     = layers.LSTM(100)(out)
    out     = layers.Dropout(dropout)(out)
    out     = layers.Dense(100, activation='relu')(out)
    output_ = layers.Dense(output_dimension, activation='softmax')(out)
    model   = Model(inputs=input_, outputs=output_)
    
    op = keras.optimizers.Adam(lr=lr_super)
    model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])
    model.summary()
    return  model
    
def supervised_model_wesad6(x_tr_feature_dim, y_tr_dim, lr_super=0.001, hidden_nodes=512, dropout=0.2, L2=0):
    input_dimension     = x_tr_feature_dim
    output_dimension    = y_tr_dim
    n_steps, n_length, n_features = 10, 256, 1
    
    input_ = keras.Input((None, n_length, n_features))
    out = layers.TimeDistributed(layers.Conv1D(filters = 64, kernel_size = 3, strides = 1, activation='relu'))(input_)
    out = layers.TimeDistributed(layers.Conv1D(filters = 64, kernel_size = 3, strides = 1, activation='relu'))(out)
    out = layers.TimeDistributed(layers.Dropout(0.1))(out)
    out = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(out)
    out = layers.TimeDistributed(layers.Flatten())(out)
    out = layers.LSTM(128, return_sequences=True)(out)
    out = layers.Dropout(0.2)(out)
    out = layers.LSTM(64, return_sequences=True)(out)
    out = layers.Dropout(0.2)(out)
    out = layers.LSTM(32)(out)
    out = layers.Dense(32, activation='relu')(out)
    output_ = layers.Dense(output_dimension, activation='softmax')(out)
    model = Model(inputs=input_, outputs=output_)
    op = keras.optimizers.Adam(lr=lr_super)
    model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])
    model.summary()
    return  model
    
def supervised_model_wesad5(x_tr_feature_dim, y_tr_dim, lr_super=0.001, hidden_nodes=512, dropout=0.5, L2=0):
    input_dimension     = x_tr_feature_dim
    output_dimension    = y_tr_dim
    n_steps, n_length, n_features   = 10, 256, 1
    
    input_  = keras.Input((n_steps, 1, n_length, n_features))
    out     = layers.ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu')(input_)
    out     = layers.Dropout(dropout)(out)
    out     = layers.Flatten()(out)
    out     = layers.Dense(100, activation='relu')(out)
    output_ = layers.Dense(output_dimension, activation='softmax')(out)
    model   = Model(inputs=input_, outputs=output_)
    
    op = keras.optimizers.Adam(lr=lr_super)
    model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])
    model.summary()
    return  model