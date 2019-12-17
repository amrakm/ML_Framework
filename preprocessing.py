import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
import transformers as ppb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from textblob import TextBlob, Word, Blobber
import joblib




class DataProcessor:

    '''
    A class to load wine data, extract features and provide a format suitable for ML algorithms

    
    Attributes
    ----------
    data_path: str
        path to csv file

    exp_path: str
        path to experiment, if provided folder name does not exist, it will create a new folder
        this will be used to store fitted model and other preprocessing modules like MinMaxScalers and OneHotEncoding
    
    target_variable: str
        column name that contains the target variable
    
    numerical_columns: list
        list of column names to extract numerical features from
    
    categorical_columns: list
        list of columns to extract categorical features from
    
    text_column: str
        column name of text data
    
    training_flag: bool
        boolean flag, if it is False then it uses the prefitted and saved preprocessing modules (MinMaxScaler, OneHotEncoding)


    Returns
    -------
    characters_in_the_scene: list
        list of characters appear in this scene
   
    '''

    def __init__(self, 
                 data_path,
                 exp_path, 
                 target_variable ='price', 
                 numerical_columns =  ['points'],
                 categorical_columns = ['country', 'variety'], 
                 text_column= 'text', 
                 training_flag= True,
                 use_bert= True,
                 processing_batch_size=2000):

        self.data_path = data_path
        self.exp_path= exp_path
        self.numerical_columns =  numerical_columns
        self.categorical_columns = categorical_columns
        self.text_column = text_column
        self.target_variable = target_variable
        self.training_flag = training_flag
        self.use_bert = use_bert
        self.processing_batch_size = processing_batch_size
        self.df = self.load_csv()




    def load_csv(self):

        logging.info('loading csv file')
        df = pd.read_csv(self.data_path, index_col=0)
        df.columns = ['text', 'country', 'variety', 'points', 'price']
        logging.info('data shape: {}'.format(str(df.shape)))
        logging.info('missing values: \n{}'.format( str(df.isnull().sum())))
        logging.info('data summary: \n{}'.format(df.describe(include='all').T.to_string()))
        
        return df.drop_duplicates().dropna().copy()


    @staticmethod
    def tokenize_text(text_series, max_review_len= 100):
        '''
        tokenized text data using pretrained distilBERT tokenizer

        Parameters
        ----------
        max_review_len: int
            maximum review length, makes all reviews the same length, limiting size to `max_review_len` and padding shorter reviews with 0, which is special token reserved for padding

        Returns
        -------
        padded: numpy.ndarray
            matrix of tokenized and padded reviews

        attention_mask: numpy.ndarray
            mask to ignore padded fields in the modeling step
        '''

        tokenizer_class, pretrained_weights = (ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        tokenized = text_series.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
        # release tokenizer from memory
        del tokenizer

        padded = np.array([i[:max_review_len] + [0]*(max_review_len-len(i)) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)

        return padded, attention_mask


    def get_text_features(self):
        
        '''
        extract reviews embedding from distilBERT model, process was divided into chunks to avoid filling the GPU memory
        if process is crashing because of full GPU memory, change the `config[processing_batch_size]` to a lower value 
        processing_batch_size is the number of reviews to be processed at one go

        Returns
        -------
        text_features: numpy.ndarray
            matrix of reviews embeddings, extracted from distilBERT along with sentiment polarity and subjectivity scores.
        
        '''
        logging.info('processing text features')

        # get sentiment scores, this returns two columns of [polarity, subjectivity]
        sentiment_arr = np.array(self.df[self.text_column].apply(lambda x: list(TextBlob(x).sentiment)).tolist())

        if self.use_bert == True:

            # utilize GPU if available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # tokenize text column in the dataframe
            tokenized_padded_text, attention_mask_numpy = self.tokenize_text(self.df[self.text_column])


            # load pretrained distilBERT model from transformers library (huggingface.co)
            model_class, pretrained_weights = (ppb.DistilBertModel, 'distilbert-base-uncased')
            model = model_class.from_pretrained(pretrained_weights)
            model.to(device)
            model.eval()

            # calculate number of batches needed to go through all dataset
            n_batches = int(np.ceil(tokenized_padded_text.shape[0]/ self.processing_batch_size))
            emb = [[] for _ in range(n_batches)]

            for i in range(n_batches):

                input_ids = torch.tensor(tokenized_padded_text[i*self.processing_batch_size: (i+1)*self.processing_batch_size, :]).to(device) 
                attention_mask = torch.tensor(attention_mask_numpy[i*self.processing_batch_size: (i+1)*self.processing_batch_size, :]).to(device)

                logging.info('creating embeddings for batch #{}'.format(str(i)))
                torch.cuda.empty_cache() 

                with torch.no_grad():
                    last_hidden_states = model(input_ids, attention_mask=attention_mask)

                batch_emb = last_hidden_states[0][:,0,:].cpu().detach().numpy() 

                del last_hidden_states, input_ids, attention_mask

                emb[i] = batch_emb

            emb = np.vstack(emb)
            text_features = np.hstack([emb, sentiment_arr])
            # release model from GPU memory
            del model
        else:
            text_features = sentiment_arr
        

        
        torch.cuda.empty_cache() 

        
        return text_features

    def get_cat_features_encoding(self):

        '''
        extract and encoding categorical features

        Returns
        -------
        cat_features: numpy.ndarray
            matrix of one-hot-encoded features

        '''

        logging.info('processing catergorical features')

        if self.training_flag == True:

            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            ohe_fitted = ohe.fit(self.df[self.categorical_columns])

            if not os.path.exists(self.exp_path):
                os.makedirs(self.exp_path)

            joblib.dump(ohe_fitted, os.path.join(self.exp_path,'ohe.pkl'))

        else:
            logging.info('loading prefitted OneHotEncoder')
            ohe_fitted = joblib.load(os.path.join(self.exp_path,'ohe.pkl'))

        cat_features = ohe_fitted.transform(self.df[self.categorical_columns])

        return cat_features


    def get_numerical_features(self):
        '''
        extract and scale numerical features from all numerical columns

        Returns
        -------
        numerical_arr: numpy.ndarray
            numerical columns scaled by MinMaxScaler(-0.5, 0.5)

        '''

        logging.info('processing numerical features')

        if self.training_flag == True:
            scaler = MinMaxScaler(feature_range=(-.5, .5))

            if not os.path.exists(self.exp_path):
                os.makedirs(self.exp_path)
            joblib.dump(scaler, os.path.join(self.exp_path,'MinMaxScaler.pkl'))

        else:
            logging.info('loading prefitted MinMaxScaler')
            scaler = joblib.load(os.path.join(self.exp_path,'MinMaxScaler.pkl'))

        numerical_arr = self.df[self.numerical_columns].values.reshape(-1,len(self.numerical_columns))
        
        return scaler.fit_transform(numerical_arr)

    def process_features(self):

        '''
        get all features combined

        Returns
        -------
        features: numpy.ndarray
            features combined in one numpy matrix

        '''

        
        text_embedding = self.get_text_features()
        cat_features = self.get_cat_features_encoding()
        numerical_features = self.get_numerical_features()
        features_combined = np.hstack([text_embedding, cat_features, numerical_features])
        
        return features_combined

    def get_processed_data(self):

        '''
        get all features combined along with target variable 

        Returns
        -------
        features: numpy.ndarray
            features combined in one numpy matrix

        target: numpy.ndarray
            array of target values
        '''


        target = self.df[self.target_variable].values
        features = self.process_features()

        return features, target
