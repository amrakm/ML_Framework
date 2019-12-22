### Modeling framework

This is a generic ML experiment framework to be used as a starting point and a baseline. The code takes a raw csv file, summarise it, process it in a format suitable for ML algorithms then trains a neural network to regress a target variable. 


Trained model along with a performance report are stored in a separate folder for each experiment.

### Setup and Dependencies:
- Create conda environment        
`conda-env create --name ml_framework --file ml_framework.yml`
- Activate conda environment        
`conda activate ml_framework`


### Preprocessing steps:       
- For simplicity, rows with missing values are dropped from the dataset.
- Text features are represented as sentiment scores and reviews embedding extracted from `distilBERT` model (Optional).
- Categorical columns are encoded with one-hot-encoding.
- Numerical columns get scaled with MinMaxScaler.



### Modeling approach:
- Simple fully connected neural network is used.
- To experiment with different hyperparameters and topologies, check out `modelling.py` module.




### How to use it:

To train a new model you need to pass the path for the csv file along with the path for the experiment folder, experiment folder will be created automatically if folder did not exist before.

Note: the code automatically detects GPU's and use them if available.

- Training:     
`python main.py --data-path <CSV_FILE_PATH> --exp-path <EXPERIMENT_FOLDER> --no-bert`

- Inference: for inference, pass the experiment_folder path from a previously trained experiment, and add `--eval` to prevent retraining        
`python main.py --data-path <CSV_FILE_PATH> --exp-path <EXPERIMENT_FOLDER> --no-bert --eval`

#### Additional parameters:
```
--target-variable: column name for target variable      
--numerical-columns: list of names for numerical columns, enter items separated by space        
--categorical-columns: list of names for categorical columns, enter items separated by space        
--processing-batch-size: batch size for extracting features from BERT        
--training-batch-size: batch size for training neural network
--epochs: number of epochs for training neural network      
```
### Files structre:
#### Main frameowrk
>.      
|____main.py        
|____modelling.py       
|____plot.py        
|____preprocessing.py       

#### Experiment Folder:     
 after training a model, experiment folder will contain the fitted model along with any preprocessing modules and performance report plots/ logs.
>|____exp_folder     
| |____MinMaxScaler.pkl     
| |____processing.log       
| |____fitted_model.h5      
| |____training_report.png      
| |____ohe.pkl      
| |____performance_report20191129132145.png     
|____Readme.md  


#### Jupyter Notebooks
Exploration notebook contains visualisations and summary statistics about the dataset.

>|____notebooks       
| |____Data-Exploration.ipynb       
| |____Data-Exploration.html        
    



### TODO:
- Refactore `modelling.py`, move neural network model to a separate `.py` file.
- Add Unit Testing.
