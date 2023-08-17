import os
import glob
import shutil
import matplotlib.pyplot as plt
from datawig import SimpleImputer
import numpy as np
import pandas as pd
import itertools
from scipy.stats import pearsonr
import warnings

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
'''
from fancyimpute import (
    MatrixFactorization,
    IterativeImputer,
    # BiScaler,
    KNN,
    SimpleFill
)
# '''
# Mean,median,knn,matrix factorization, random forest and linear regression

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)
pd.set_option("max_rows", 100)
pd.set_option("max_columns", 10)


def sample(m, n, regular):
    result = np.empty([m, n])

    '''
    time_sampler = ts.TimeSampler(stop_time=18)
    time_points = None

    if regular:
        time_points = time_sampler.sample_regular_time(num_points=n)
    else:
        time_points = time_sampler.sample_irregular_time(num_points=2*n, keep_percentage=50)  # arbitrary chose 50%

    print(time_points)

    for i in range(m):
        red_noise = ts.noise.RedNoise(std=1, tau=2)
        pp = ts.signals.PseudoPeriodic(frequency=0.75, freqSD=0.015, ampSD=0.05)
        # pp = ts.signals.CAR(ar_param=0.9, sigma=0.01)
        pp_series = ts.TimeSeries(signal_generator=pp, noise_generator=red_noise)
        
        samples, signals, errors = pp_series.sample(time_points)
        seed = 0.75 * np.random.randn()
        samples += seed

        plt.plot(time_points, samples, marker='o', markersize=4)

        result[i] = samples
    # '''

    return result


def generate_missing_mask(x, percent_missing=10, missingness='MCAR'):
    if missingness == 'MCAR':
        # missing completely at random
        mask = np.random.rand(*x.shape) < percent_missing / 100.
    elif missingness == 'MAR':
        # missing at random, missingness is conditioned on a random other column
        # this case could contain MNAR cases, when the percentile in the other column is
        # computed including values that are to be masked
        mask = np.zeros(x.shape)
        n_values_to_discard = int((percent_missing / 100) * x.shape[0])
        # for each affected column
        for col_affected in range(x.shape[1]):
            # select a random column for missingness to depend on
            depends_on_col = np.random.choice([c for c in range(x.shape[1]) if c != col_affected])
            # pick a random percentile of values in other column
            if n_values_to_discard < x.shape[0]:
                discard_lower_start = np.random.randint(0, x.shape[0] - n_values_to_discard)
            else:
                discard_lower_start = 0
            discard_idx = range(discard_lower_start, discard_lower_start + n_values_to_discard)
            values_to_discard = x[:, depends_on_col].argsort()[discard_idx]
            mask[values_to_discard, col_affected] = 1
    elif missingness == 'MNAR':
        # missing not at random, missingness of one column depends on unobserved values in this column
        mask = np.zeros(x.shape)
        n_values_to_discard = int((percent_missing / 100) * x.shape[0])
        # for each affected column
        for col_affected in range(x.shape[1]):
            # pick a random percentile of values in other column
            if n_values_to_discard < x.shape[0]:
                discard_lower_start = np.random.randint(0, x.shape[0] - n_values_to_discard)
            else:
                discard_lower_start = 0
            discard_idx = range(discard_lower_start, discard_lower_start + n_values_to_discard)
            values_to_discard = x[:, col_affected].argsort()[discard_idx]
            mask[values_to_discard, col_affected] = 1
    return mask > 0


def evaluate_mse(x_imputed, x, mask):
    return ((x_imputed[mask] - x[mask]) ** 2).mean()


def dict_product(hp_dict):
    """
    Returns cartesian product of hyperparameters
    """
    return [dict(zip(hp_dict.keys(), vals)) for vals in
            itertools.product(*hp_dict.values())]


def fancyimpute_hpo(fancyimputer, param_candidates, x, mask, percent_validation=10):
    # first generate all parameter candidates for grid search
    all_param_candidates = dict_product(param_candidates)
    # get linear indices of all training data points
    train_idx = (mask.reshape(np.prod(x.shape)) is False).nonzero()[0]
    # get the validation mask
    n_validation = int(len(train_idx) * percent_validation / 100)
    validation_idx = np.random.choice(train_idx, n_validation)
    validation_mask = np.zeros(np.prod(x.shape))
    validation_mask[validation_idx] = 1
    validation_mask = validation_mask.reshape(x.shape) > 0
    # save the original data
    x_incomplete = x.copy()
    # set validation and test data to nan
    x_incomplete[mask | validation_mask] = np.nan
    mse_hpo = []
    for params in all_param_candidates:
        if fancyimputer.__name__ != 'SimpleFill':
            params['verbose'] = False
        x_imputed = fancyimputer(**params).fit_transform(x_incomplete)
        mse = evaluate_mse(x_imputed, x, validation_mask)
        print(f"Trained {fancyimputer.__name__} with {params}, mse={mse}")
        mse_hpo.append(mse)

    best_params = all_param_candidates[np.array(mse_hpo).argmin()]
    # now retrain with best params on all training data
    x_incomplete = x.copy()
    x_incomplete[mask] = np.nan
    x_imputed = fancyimputer(**best_params).fit_transform(x_incomplete)
    mse_best = evaluate_mse(x_imputed, x, mask)
    print(f"HPO: {fancyimputer.__name__}, best {best_params}, mse={mse_best}")
    return x_imputed


def impute_datawig(x, mask):
    x_incomplete = x.copy()
    x_incomplete[mask] = np.nan

    df = pd.DataFrame(x_incomplete)
    df.columns = [str(c) for c in df.columns]

    dw_dir = "C:\\Users\\danie\\miniconda3\\envs\\regeneron_project_2\\running files\\datawig_imputers"
    df = SimpleImputer.complete(df, output_path=dw_dir, hpo=True, verbose=0, iterations=1)
    for d in glob.glob(os.path.join(dw_dir, '*')):
        shutil.rmtree(d)

    return df.to_numpy()



imputers = [
    impute_datawig
]


def experiment(original_table, percent_missing_list=(1, 5, 10, 20), missingness_list=('MCAR', 'MAR', 'MNAR')):
    for percent_missing_i in percent_missing_list:
        for missingness_i in missingness_list:
            for imputer in imputers:
                for _ in range(3):
                    data_mask = generate_missing_mask(original_table, percent_missing_i, missingness_i)
                    imputed_table = imputer(original_table, data_mask)

                    mse = evaluate_mse(imputed_table, original_table, data_mask)
                    frobenius_norm = np.linalg.norm(original_table - imputed_table, 'fro')
                    corr = pearsonr(imputed_table[data_mask], original_table[data_mask])[0]
                    plt.scatter(original_table[data_mask], imputed_table[data_mask])

                    print("imputer: %s, percent missing: %s, missingness: %s, MSE: %s, correlation: %s, norm: %s"
                          % (imputer.__name__, percent_missing_i, missingness_i, mse, corr, frobenius_norm))
    plt.show()


data = sample(10000, 10, False)
experiment(data)
