# import timesynth as ts
import matplotlib.pyplot as plt
from datawig import SimpleImputer
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings


warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)
pd.set_option("max_rows", 100)
pd.set_option("max_columns", 10)


def sample(m, n, regular=True):
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
        white_noise = ts.noise.GaussianNoise(std=0.0)
        gp = ts.signals.PseudoPeriodic(frequency=0.75, freqSD=0.01, ampSD=0.5)
        # gp = ts.signals.CAR(ar_param=0.9, sigma=0.01)
        gp_series = ts.TimeSeries(signal_generator=gp, noise_generator=white_noise)

        samples, signals, errors = gp_series.sample(time_points)
        seed = 0.75 * np.random.randn()
        samples += seed

        plt.plot(time_points, samples, marker='o', markersize=4)

        result[i] = samples
    # '''

    return result


def generate_missing_mask(X, percent_missing=10, missingness='MCAR'):
    if missingness == 'MCAR':
        # missing completely at random
        mask = np.random.rand(*X.shape) < percent_missing / 100.
    elif missingness == 'MAR':
        # missing at random, missingness is conditioned on a random other column
        # this case could contain MNAR cases, when the percentile in the other column is
        # computed including values that are to be masked
        mask = np.zeros(X.shape)
        n_values_to_discard = int((percent_missing / 100) * X.shape[0])
        # for each affected column
        for col_affected in range(X.shape[1]):
            # select a random column for missingness to depend on
            depends_on_col = np.random.choice([c for c in range(X.shape[1]) if c != col_affected])
            # pick a random percentile of values in other column
            if n_values_to_discard < X.shape[0]:
                discard_lower_start = np.random.randint(0, X.shape[0] - n_values_to_discard)
            else:
                discard_lower_start = 0
            discard_idx = range(discard_lower_start, discard_lower_start + n_values_to_discard)
            values_to_discard = X[:, depends_on_col].argsort()[discard_idx]
            mask[values_to_discard, col_affected] = 1
    elif missingness == 'MNAR':
        # missing not at random, missingness of one column depends on unobserved values in this column
        mask = np.zeros(X.shape)
        n_values_to_discard = int((percent_missing / 100) * X.shape[0])
        # for each affected column
        for col_affected in range(X.shape[1]):
            # pick a random percentile of values in other column
            if n_values_to_discard < X.shape[0]:
                discard_lower_start = np.random.randint(0, X.shape[0] - n_values_to_discard)
            else:
                discard_lower_start = 0
            discard_idx = range(discard_lower_start, discard_lower_start + n_values_to_discard)
            values_to_discard = X[:, col_affected].argsort()[discard_idx]
            mask[values_to_discard, col_affected] = 1
    return mask > 0


def evaluate_mse(X_imputed, X, mask):
    return ((X_imputed[mask] - X[mask]) ** 2).mean()


data_np = sample(100, 10, False)
data_pd = pd.DataFrame(data_np)
print(data_pd)


dw_dir = "C:\\Users\\danie\\miniconda3\\envs\\regeneron_project_2\\running files\\datawig_imputers"
percent_missing_list = [5, 10, 30, 50]
missingness_list = ['MCAR', 'MAR', 'MNAR']

'''
for percent_missing_i in percent_missing_list:
    for missingness_i in missingness_list:
        for _ in range(3):
            data_mask = generate_missing_mask(data_np, percent_missing_i, missingness_i)

            data_incomplete = data_np.copy()
            data_incomplete[data_mask] = np.nan

            df = pd.DataFrame(data_incomplete)
            df.columns = [str(c) for c in data_pd.columns]
            df = SimpleImputer.complete(df, output_path=dw_dir, hpo=True, verbose=0, iterations=1)
            df_np = df.to_numpy()

            imputed_values = df_np[data_mask]
            original_values = data_np[data_mask]

            mse = evaluate_mse(df_np, data_np, data_mask)
            corr = pearsonr(imputed_values, original_values)[0]

            plt.scatter(original_values, imputed_values)

            print("Percent Missing: %s, Missingness: %s, MSE: %s, correlation: %s"
                  % (percent_missing_i, missingness_i, mse, corr))
'''

data_mask = generate_missing_mask(data_np, 10, 'MCAR')

data_incomplete = data_np.copy()
data_incomplete[data_mask] = np.nan

df = pd.DataFrame(data_incomplete)
df.columns = [str(c) for c in data_pd.columns]
df = SimpleImputer.complete(df, output_path=dw_dir, hpo=True, verbose=0, iterations=1)
df_np = df.to_numpy()

imputed_values = df_np[data_mask]
original_values = data_np[data_mask]

mse = evaluate_mse(df_np, data_np, data_mask)
corr = pearsonr(imputed_values, original_values)[0]

plt.scatter(original_values, imputed_values)

print("Percent Missing: %s, Missingness: %s, MSE: %s, correlation: %s"
      % (10, 'MCAR', mse, corr))

plt.show()
