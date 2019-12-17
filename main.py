import argparse
import os
import logging
import time

from preprocessing import DataProcessor
from modelling import split_training_data, load_fitted_model, make_prediction, fit_predictive_model
from plot import print_trainig_log, plot_regression_performance


if __name__ == '__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", metavar="--data_path", required=True,
        help="path to csv file")
    ap.add_argument("--exp-path", metavar="--exp_path", required=True,
        help="path to experiment folder")
    ap.add_argument("--target-variable", metavar='--target_variable',required=False,
        help="column name for target variable")
    ap.add_argument("--numerical-columns", metavar="--numerical_columns",nargs='+', required=False,
        help="list of names for numerical columns, enter items separated by space ")
    ap.add_argument("--categorical-columns", metavar="--categorical_columns", nargs='+',required=False,
        help="list of names for categorical columns, enter items separated by space")
    ap.add_argument("--processing-batch-size", metavar="--processing_batch_size", required=False, type=int,
        help="batch size for extracting features from BERT")
    ap.add_argument("--training-batch-size", metavar="--training_batch_size", required=False, type=int,
        help="batch size for training neural network")
    ap.add_argument("--epochs",  required=False, type=int,
        help="number of epochs for training neural network")
        
    bert = ap.add_mutually_exclusive_group(required=False)
    bert.add_argument('--use-bert', dest='use_bert', action='store_true')
    bert.add_argument('--no-bert', dest='use_bert', action='store_false')
    training = ap.add_mutually_exclusive_group(required=False)
    training.add_argument('--training', dest='training_flag', action='store_true')
    training.add_argument('--eval', dest='training_flag', action='store_false')
        

    ap.set_defaults(target_variable= 'price')
    ap.set_defaults(numerical_columns= ['points'])
    ap.set_defaults(categorical_columns= ['country', 'variety'])
    ap.set_defaults(processing_batch_size= 2000)
    ap.set_defaults(training_batch_size= 5000)
    ap.set_defaults(epochs= 500)
    ap.set_defaults(use_bert = False)
    ap.set_defaults(training_flag= True)

    args = ap.parse_args()

    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)

    timestamp = str(time.strftime('%Y%m%d%H%M%S'))

    handlers = [logging.FileHandler(os.path.join(args.exp_path,'processing.log')), logging.StreamHandler()]
    logging.basicConfig(level= logging.INFO, handlers= handlers)

    DP = DataProcessor(data_path= args.data_path,
                       exp_path= args.exp_path,
                       target_variable= args.target_variable,
                       numerical_columns= args.numerical_columns,
                       categorical_columns= args.categorical_columns,
                       processing_batch_size= args.processing_batch_size,
                       training_flag= args.training_flag,
                       use_bert= args.use_bert)

    X, y = DP.get_processed_data()

    if args.training_flag == True:

        X_train, X_test, y_train, y_test = split_training_data(X,y)
        fitted_model, training_log = fit_predictive_model(X_train, y_train,
                                                        saving_path= args.exp_path,
                                                        epochs=args.epochs, batch_size=args.training_batch_size)

        X_eval = X_test
        y_eval = y_test
        fig = print_trainig_log(training_log)
        fig.savefig(os.path.join(args.exp_path,'training_report.png'))


    else:
        X_eval = X
        y_eval = y

    predicted = make_prediction(X_eval, args.exp_path)
    fig = plot_regression_performance(y_eval, predicted)
    fig.savefig(os.path.join(args.exp_path,'performance_report' + timestamp +'.png'))

