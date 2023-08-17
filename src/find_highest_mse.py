import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

np.set_printoptions(suppress=True)
pd.options.display.max_rows = 1000

all_data = pd.read_csv('C:\\Users\\danie\\miniconda3\\envs\\regeneron_project_2\\user written files\\highests.csv')
calculated = \
    pd.read_csv('C:\\Users\\danie\\miniconda3\\envs\\regeneron_project_2\\running files\\benchmark_results.csv')
calculated = calculated.loc[:108]

'''
thirty_MCAR = all_data.loc[(all_data['percent_missing'] == 30) & (all_data['missingness'] == 'MCAR')]
thirty_MAR = all_data.loc[(all_data['percent_missing'] == 30) & (all_data['missingness'] == 'MAR')]
thirty_MNAR = all_data.loc[(all_data['percent_missing'] == 30) & (all_data['missingness'] == 'MNAR')]

ten_MCAR = all_data.loc[(all_data['percent_missing'] == 10) & (all_data['missingness'] == 'MCAR')]
ten_MAR = all_data.loc[(all_data['percent_missing'] == 10) & (all_data['missingness'] == 'MAR')]
ten_MNAR = all_data.loc[(all_data['percent_missing'] == 10) & (all_data['missingness'] == 'MNAR')]
'''
# thirty_MCAR_max = thirty_MCAR.max(1, level='mse')

paper = all_data.loc[((all_data['percent_missing'] == 30) | (all_data['percent_missing'] == 10)) &
                     (all_data['imputer'] == 'datawig')]
# paper = paper.groupby(['data', 'missingness', 'percent_missing'])['mse'].transform(max)


paper = paper.loc[all_data['imputer'] == 'datawig']
paper.reset_index(inplace=True, drop=True)

print(paper)
print('\n')
print(calculated.mse)

print(calculated.mse / paper)

print(paper)
print('\n\n')
print(calculated)

'''
paper_np = paper.mse.to_numpy()
calculated_np = calculated.mse.to_numpy()
print(paper_np)
print(calculated_np)

corr = pearsonr(np.log(calculated_np), np.log(paper_np))[0]
print('\n correlation: %s' % corr)

plt.figure(1)
plt.subplot(2, 1, 1)
plt.scatter(calculated_np, paper_np)

plt.subplot(2, 1, 2)
plt.xscale('log'), plt.yscale('log')
plt.scatter(calculated_np, paper_np)
plt.show()


plt.figure(2)

plt.scatter(calculated_np, paper_np)
plt.show()
'''

df = pd.read_csv('C:\\Users\\danie\\miniconda3\\envs\\regeneron_project_2\\user written files\\highests.csv')
df['mse_percent'] = df.mse / df.groupby(['data', 'missingness', 'percent_missing'])['mse'].transform(max)
df.groupby(['missingness', 'percent_missing', 'imputer']).agg({'mse_percent': 'median'})

sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("RdBu_r", 7))
sns.set_context("notebook",
                font_scale=1.3,
                rc={"lines.linewidth": 1.5})
plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
sns.boxplot(hue='imputer',
            y='mse_percent',
            x='percent_missing', data=df[df['missingness'] == 'MCAR'])
plt.title("Missing completely at random")
plt.xlabel('Percent Missing')
plt.ylabel("Relative MSE")
plt.gca().get_legend().remove()

plt.subplot(1, 3, 2)
sns.boxplot(hue='imputer',
            y='mse_percent',
            x='percent_missing',
            data=df[df['missingness'] == 'MAR'])
plt.title("Missing at random")
plt.ylabel('')
plt.xlabel('Percent Missing')
plt.gca().get_legend().remove()

plt.subplot(1, 3, 3)
sns.boxplot(hue='imputer',
            y='mse_percent',
            x='percent_missing',
            data=df[df['missingness'] == 'MNAR'])
plt.title("Missing not at random")
plt.ylabel("")
plt.xlabel('Percent Missing')

handles, labels = plt.gca().get_legend_handles_labels()

l = plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.tight_layout()
# plt.savefig('benchmarks_datawig.pdf')

plt.show()
