# from datawig import SimpleImputer
import datawig as dw
import pandas as pd
import numpy as np


pd.set_option('max_rows', 100)

# generate some data with simple nonlinear dependency
numeric = np.random.multivariate_normal(np.full(50, 2.), np.eye(50), 50)
df = pd.DataFrame(numeric, columns=list(map(str, list(range(50)))))
# df = dw.utils.generate_df_numeric()
print(df)
print('\n')

# mask 10% of the values
removed = np.random.rand(*df.shape) > .9
df_with_missing = df.mask(removed)
print(df_with_missing)
print('\n')

# impute missing values
df_with_missing_imputed = dw.SimpleImputer.complete(df_with_missing)
print(df_with_missing_imputed)


# ---------------------------------------------------------------------
np_df = df.to_numpy()
np_df_imp = df_with_missing_imputed.to_numpy()

actual = np_df[removed]
guessed = np_df_imp[removed]

actual_mean = np.sum(np_df) / np_df.size
guessed_mean = np.sum(np_df_imp) / np_df_imp.size

print("\n\noriginal mean: %s" % actual_mean)
print("guessed mean: %s" % guessed_mean)

error = 1/actual.size * np.sum(np.abs(guessed - actual))
print("error: %s" % error)
