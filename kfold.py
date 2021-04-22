import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
import model, utils, data_preprocess
from tensorflow import keras

epoch      = 200
batch      = 64
output     = 'output_kfold'
logs       = 'logs_kfold'
model_name = 'wesad_affect6'   #choose the model to run!!! 
data_folder= './PreprocessedData'

## load data
wesad_data = data_preprocess.load_data(os.path.join(data_folder, 'wesad_dict.npy')) #(22085, 2561)
tb         = keras.callbacks.TensorBoard(log_dir = logs)

## screen data
wesad_data = data_preprocess.screen(wesad_data)                   #(11575, 2561)

## select the test and train index
total_fold = 10
kf         = KFold(n_splits=total_fold, shuffle=True, random_state=True)
wesad_train_index, wesad_test_index = utils.get_train_test_index(wesad_data, kf)

for k in range(total_fold):
  flag = k
  print('the %d kfold.'%(k))
  
  ## load train and test data               
  x_tr = wesad_data[wesad_train_index[k], 1:] #(10417, 2560)
  x_te = wesad_data[wesad_test_index[k], 1:]  #(1158,  2560)
  x_tr, x_te = utils.reshape_data(model_name=model_name, x_tr=x_tr, x_te=x_te)

  y_tr, y_te = utils.one_hot_encoding(arr = wesad_data[:, 0],  tr_index = wesad_train_index[k], te_index = wesad_test_index[k]) # (10417, 4), (1158, 4)
 
  ## build wesad prediction model
  model_path, wesad_model = utils.build_model(model_name = model_name, x_tr_dim = wesad_data[0:1,1:].shape[1])
    
  wesad_model.fit(x_tr, y_tr, epochs=epoch, batch_size=batch, callbacks=[tb], verbose=1, validation_data = (x_te, y_te), shuffle = True) 
  
  ## measure model performennce
  y_tr_pred = wesad_model.predict(x_tr, batch_size=batch)
  y_te_pred = wesad_model.predict(x_te, batch_size=batch)
  
  y_tr      = np.argmax(y_tr, axis = 1)
  y_te      = np.argmax(y_te, axis = 1)
  
  y_tr_pred = np.argmax(y_tr_pred, axis = 1)
  y_te_pred = np.argmax(y_te_pred, axis = 1)
  
  utils.model_result_store(y_tr, y_tr_pred, os.path.join(output, str("tr_" + model_name + "_kfold.csv")), flag)
  utils.model_result_store(y_te, y_te_pred, os.path.join(output, str("te_" + model_name + "_kfold.csv")), flag)

  wesad_model.save_weights(model_path + "_" + str(k) + "kfold.h5")
             