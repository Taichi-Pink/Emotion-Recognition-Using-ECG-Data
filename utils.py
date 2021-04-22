import tensorflow as tf
import numpy as np
import csv, os, time
from sklearn import metrics
from mlxtend.evaluate import confusion_matrix
import model

def get_train_test_index(data, kf):
    train_index = []
    test_index  = []
    for train_i, test_i in kf.split(data):
        train_index.append(train_i)
        test_index.append(test_i)

    return train_index, test_index
    
def one_hot_encoding(arr, tr_index, te_index):
    num_of_class = len(np.unique(arr))
    min_val      = np.min(arr)
    arr          = arr - min_val
    tr_encoded_array = tf.keras.utils.to_categorical(arr[tr_index], num_classes = num_of_class) 
    te_encoded_array = tf.keras.utils.to_categorical(arr[te_index], num_classes = num_of_class) 

    return tr_encoded_array, te_encoded_array

def build_model(model_name, x_tr_dim):
    
    if model_name == 'wesad_affect1':
        model_path  = './wesad_model1'
        wesad_model = model.supervised_model_wesad1(x_tr_feature_dim = x_tr_dim, y_tr_dim = 4)  
    elif model_name == 'wesad_affect2':
        model_path  = './wesad_model2'
        wesad_model = model.supervised_model_wesad2(x_tr_feature_dim = x_tr_dim, y_tr_dim = 4)  
    elif model_name == 'wesad_affect4':
        model_path  = './wesad_model4'
        wesad_model = model.supervised_model_wesad4(x_tr_feature_dim = x_tr_dim, y_tr_dim = 4)
    elif model_name == 'wesad_affect6':
        model_path  = './wesad_model6'
        wesad_model = model.supervised_model_wesad6(x_tr_feature_dim = x_tr_dim, y_tr_dim = 4)
    elif model_name == 'wesad_affect5':
        model_path  = './wesad_model5'
        wesad_model = model.supervised_model_wesad5(x_tr_feature_dim = x_tr_dim, y_tr_dim = 4)
    
    return model_path, wesad_model

def reshape_data(model_name, x_tr, x_te, n_steps=10, n_length=256):
                                                                           #  kfold.py                         DL.py
                                                                           #(10417, 2560)           |      (9260, 2560)
    if model_name != 'wesad_affect1':                                      #(1158 , 2560)           |      (2315, 2560)
        if model_name == 'wesad_affect4' or model_name == 'wesad_affect6':                              
          x_tr = np.reshape(x_tr, (x_tr.shape[0], n_steps, n_length, 1))   #(10417, 10, 256, 1)     |      (9260, 10, 256, 1)
          x_te = np.reshape(x_te, (x_te.shape[0], n_steps, n_length, 1))   #(1158 , 10, 256, 1)     |      (2315, 10, 256, 1)
        elif model_name == 'wesad_affect5':
          x_tr = np.reshape(x_tr, (x_tr.shape[0], n_steps, 1, n_length, 1))#(10417, 10, 1, 256, 1)  |      (9260, 10, 1, 256, 1)
          x_te = np.reshape(x_te, (x_te.shape[0], n_steps, 1, n_length, 1))#(1158 , 10, 1, 256, 1)  |      (2315, 10, 1, 256, 1)
        else:
          x_tr = np.reshape(x_tr, (x_tr.shape[0], x_tr.shape[1], 1))       #(10417, 2560, 1)        |      (9260, 2560, 1)
          x_te = np.reshape(x_te, (x_te.shape[0], x_te.shape[1], 1))       #(1158 , 2560, 1)        |      (2315, 2560, 1)
    
    return x_tr, x_te
    
def model_result_store(y, y_pred, result_store, kfold):
    """ 
    evaluate models by 4 metrics: accuracy, precision, recall and f1_score
    """
    accuracy    = np.round(metrics.accuracy_score(y, y_pred), 4)
    conf_mat    = confusion_matrix(y_target=y, y_predicted=y_pred, binary=False)
    precision   = np.round(np.mean(np.diag(conf_mat) / np.sum(conf_mat, axis = 0)), 4)
    recall      = np.round(np.mean(np.diag(conf_mat) / np.sum(conf_mat, axis = 1)), 4)
    f1          = np.round(2*precision*recall / (precision + recall), 4)
    print('display confusion matrix:\n',conf_mat)
    
    with open(result_store, 'a', newline='') as csvfile:
        fieldnames = ['fold', 'accuracy','precision', 'recall', 'f1']
        writer     = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if kfold==0:
          writer.writeheader()
        writer.writerow({'fold': kfold, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1' : f1})
                
    return 