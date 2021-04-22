# Emotion-Recognition-Using-ECG-Data

## Download
* [Dataset](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)
* [Preprocessed data]()

## Requirements
* Python 	     =3.7.9
*	TensorFlow   = 1.15.0
*	TensorBoard  = 1.15.0
*	NumPy 		 = 1.19.2
*	Pandas 		 = 1.2.2
*	Scikit-Learn = 0.24.1
*	Tqdm 		 = 4.50.2
*	Mlxtend 	 = 0.18.0


## Code structure

|
|------Dataset
|		|
|		|--WESAD
|			|
|			|--S2.pkl
|			|   .
|			|	.
|			|	.
|			|--S17.pkl
|
|------logs
|------output
|		|
|		|--tr_wesad_affect2.csv
|		|--te_wesad_affect2.csv
|
|------PreprocessedData
|		|
|		|--wesad_dict.npy
|
|------weights
|------logs_kfold
|------output_kfold
|------weights_kfold
|
|------data_preprocess.py
|------model.py
|------kfold.py
|------DL.py
|------utils.py
|------wesad_model1/2/4/5/6.h5

* logs/logs_kfold 	  : store the log files during training.
* output/output_kfold   : store the performence evaluation files (tr means train performence, te:test)
* weights/weights_kfold : store the weight of trained wesad_model# in wesad_model#.h5 format.
* PreprocessedData	  : store the preprocessed data obtained by runing data_preprocess.py; if you want to run data_preprocess.py, modify "preprocess_data = 0" to "preprocess_data = 1".
* model.py			  : build model structure; we have 5 models here.
* kfold.py			  : implement kfold cross-validation on models; compare and choose the best model.
* DL.py				  : train the best model.

## Implementation
* Change the "model_name" in kfold.py/DL.py; you could choose 'wesad_affect1'/'wesad_affect2'/'wesad_affect4'/'wesad_affect5'/'wesad_affect6'.
* run DL.py
	
