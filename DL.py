import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
import model, utils, data_preprocess
from tensorflow import keras

epoch      = 1000
batch      = 64
output     = 'output'
logs       = 'logs'
model_name = 'wesad_affect6' #choose the model to run!!!
data_folder= './PreprocessedData'

## load data
tb         = keras.callbacks.TensorBoard(log_dir = logs)
wesad_data = data_preprocess.load_data(os.path.join(data_folder, 'wesad_dict.npy')) #(22085, 2561)
wesad_data = data_preprocess.screen(wesad_data)
X, y       = wesad_data[:, 1:], wesad_data[:, 0]
x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=5) #put 80% data in training set

## reshape data
x_tr, x_te = utils.reshape_data(model_name=model_name, x_tr=x_tr, x_te=x_te)

## one hot encoding
y_tr = tf.keras.utils.to_categorical(y_tr, num_classes = 4) 
y_te = tf.keras.utils.to_categorical(y_te, num_classes = 4) 

## build model structure
model_path, wesad_model = utils.build_model(model_name = model_name, x_tr_dim = wesad_data[0:1,1:].shape[1])
model_path  = model_path + '_training.h5'

## load pretrained weights
if os.path.exists(model_path):
  wesad_model.load_weights(model_path)

wesad_model.fit(x_tr, y_tr, epochs=epoch, batch_size=batch, callbacks=[tb], verbose=1, validation_data = (x_te, y_te), shuffle = True) 

## measure model performennce
y_tr_pred = wesad_model.predict(x_tr, batch_size=batch)

## test on the test set
y_te_pred = wesad_model.predict(x_te, batch_size=batch)

y_tr      = np.argmax(y_tr, axis = 1)
y_te      = np.argmax(y_te, axis = 1)

y_tr_pred = np.argmax(y_tr_pred, axis = 1)
y_te_pred = np.argmax(y_te_pred, axis = 1)

flag = 0
utils.model_result_store(y_tr, y_tr_pred, os.path.join(output, str("tr_" + model_name + "_training.csv")), flag)
utils.model_result_store(y_te, y_te_pred, os.path.join(output, str("te_" + model_name + "_training.csv")), flag)

wesad_model.save_weights(model_path)
             