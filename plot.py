import logging
logger = logging.getLogger(__name__)

import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
plt.style.use('bmh')

def get_evaluation_metrics(x, y):

    pearson = round(pearsonr(x,y.reshape(-1))[0], 5)
    mse = round(mean_squared_error(x,y),5)
    mae = round(mean_absolute_error(x,y),5)
    spearman = spearmanr(x,y)[0]
    pval = spearmanr(x,y)[1]

    return  pearson, mse, mae, spearman, pval 

def print_trainig_log(training_log):
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16,12))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5 )

    ax[0].plot(training_log.history['mse'], label='training MSE')
    ax[0].plot(training_log.history['val_mse'], label='validation MSE')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].set_title('MSE training log')

    ax[1].plot(training_log.history['mae'], label='training MAE')
    ax[1].plot(training_log.history['val_mae'], label='validation MAE')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].legend()
    ax[1].set_title('MAE training log')

    plt.show()

    return fig


def plot_regression_performance(x, y):
    
    n = len(x)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,10))
    ax[0].tick_params(axis='x', which='both', length=0)
    ax[0].tick_params(axis='y', which='both', length=0)
    

    pearson, mse, mae, spearman, pval = get_evaluation_metrics(x, y)

    annotate_text = ' mse = {} \n mae = {} \n pearson = {} \n spearman = {} '\
            .format(round(mse,2), round(mae,2),  round(pearson,2), round(spearman,2))
    
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5 )

    ax[0].plot(x, y, '.')

    # place a text box in upper left in axes coords
    ax[0].text(0.81, 0.95, annotate_text, transform=ax[0].transAxes, fontsize=14,
            verticalalignment='top', ha='left')


    ax[0].set_title('Actual VS Predicted Price n={}'.format( n)
                 , fontweight='bold', fontsize=20)
    ax[0].set_xlabel('Price', fontsize=19)
    ax[0].set_ylabel('Prediction', fontsize=19)
    fig.suptitle('')
    

    ax[1].set_title('Residuals Plot', fontweight='bold', fontsize=20)
    sns.residplot(x, y, ax=ax[1])
    ax[1].set_xlabel('Price', fontsize=19)
    ax[1].set_ylabel('Residuals', fontsize=19)
    ax[1].set_xlim(0,1000)

    logger.info('\n Evaluation metrics: \n{}'.format(annotate_text))
    plt.show()

    return fig
