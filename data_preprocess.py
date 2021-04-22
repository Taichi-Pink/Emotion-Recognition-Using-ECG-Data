import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import utils
import pickle
from sklearn.utils import resample

def normalize(d, d_mean, d_std):
    """ 
    normaliz data 
    """
    d_norm = (d-d_mean)/d_std
    return d_norm
    
def make_window(signal, fs, overlap, window_size_sec):
    """ 
    crop data into a fixed window size 
    """
    window_size = fs * window_size_sec
    overlap     = int(window_size * (overlap / 100))
    start       = 0   
    segmented   = np.zeros((1, window_size), dtype = int)
    while(start+window_size <= len(signal)):
        segment     = signal[start:start+window_size]
        segment     = segment.reshape(1, len(segment))
        segmented   = np.append(segmented, segment, axis =0)
        start       = start + window_size - overlap
    return segmented[1:]

def extract_wesad_dataset(overlap_pct, window_size_sec, data_save_path):
    """ 
    crop data into a fixed window size for each subject (total 15)
    stack them up
    """
    s = ['2','3','4','5','6','7','8','9','10','11','13','14','15','16','17']
    freq = 256

    if not os.path.exists(data_save_path):
      os.makedirs(data_save_path)
    window_size = window_size_sec * freq
    
    wesad_dict   = {}
    wesad_labels = {}
          
    for i in tqdm(s):
        p='./Dataset/WESAD/S'+i+'.pkl'
        with open(p, 'rb') as f:
          s_data = pickle.load(f, encoding = 'latin1')
        data   = s_data['signal']['chest']['ECG'][:,0]
        labels = s_data['label']
        
        data_sorted = np.sort(data)
        d_std       = np.std(data_sorted[np.int(0.025*data_sorted.shape[0]) : np.int(0.975*data_sorted.shape[0])])
        d_mean      = np.mean(data_sorted)
        data        = normalize(data, d_mean, d_std)
    
        data_windowed   =    make_window(data, freq, overlap_pct, window_size_sec)
        labels_windowed =    make_window(labels, freq, overlap_pct, window_size_sec)

        wesad_dict.update({i: data_windowed})
        wesad_labels.update({i: labels_windowed})
        
    final_set = np.zeros((1, window_size+1), dtype = int)
    for i in tqdm(wesad_dict.keys()):
        values     = wesad_dict[i]
        labels     = wesad_labels[i]
        labels_max = np.amax(labels, axis = 1)
        labels_max = labels_max.reshape(len(labels_max), 1)

        signal_set = np.hstack((labels_max, values))
        final_set  = np.vstack((final_set, signal_set))
    
    final_set = final_set[1:]
    np.save(data_save_path+ '/wesad_dict.npy', final_set)

       
def load_data(path):
    """ 
    load data from a given path
    """
    dataset = np.load(path, allow_pickle=True)     
    return dataset   


def screen(wesad_data):
    """ 
    screen data, save the data labeled 1, 2, 3, 4
    """
    y_stress = wesad_data[:, 0]    
    ecg      = wesad_data[:, 1:]

    ecg             = ecg[(y_stress != 0) & (y_stress != 5) & (y_stress != 6) & (y_stress != 7)]
    y_stress        = y_stress[(y_stress != 0) & (y_stress != 5) & (y_stress != 6) & (y_stress != 7)] - 1
    y_stress        = y_stress.reshape(-1, 1)
    wesad_data      = np.hstack((y_stress, ecg))
    return wesad_data
    
if __name__ == "__main__":
    preprocess_data = 0
    data_folder='./PreprocessedData'
    
    if preprocess_data == 1:
        extract_wesad_dataset(overlap_pct=0, window_size_sec=10, data_save_path= data_folder)

