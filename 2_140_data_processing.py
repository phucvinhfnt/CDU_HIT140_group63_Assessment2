# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import kstest
from sklearn import metrics
import math

#0 INPUT DATA
df_po2 = pd.read_csv("po2_data.csv")
# Reorder columns of df_po2
df_po2_columns = [
    'subject#', 'motor_updrs','total_updrs', 'age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]
po2_df = df_po2[df_po2_columns]

# print(po2_df)
# CREATE TWO DATA FRAME TO TEST
# DATA 1 CONTAINS Y = MOTOR_UPDRS AND 18 VARIABLES

motor_columns = [
    'subject#', 'motor_updrs', 'age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]

motor_up_df = df_po2[motor_columns]
print(motor_up_df)

# DATA 2 CONTAINS Y = TOTAL_UPDRS AND 18 VARIABLES

total_columns = [
    'subject#', 'total_updrs', 'age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]

total_up_df = df_po2[total_columns]
# print(total_up_df)

#2.2 VISUALIZATION THE HISTOGRAM
# Histogram 18 variables to check the distribution of the sample
# There are: 
# 'motor_updrs', 'total_updrs', 
# 'jitter(%)', 'jitter(abs)','jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)', 
# 'shimmer(%)','shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', shimmer(dda)', 
# 'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
columns_his = po2_df.columns[1::]  
# print(columns_his)
num_columns = len(columns_his)
plt.figure(figsize=(12, 10))
for i in range(num_columns):
    plt.subplot((num_columns // 4) + 1, 4, i + 1)
    # plt.figure(figsize=(6, 8))
    sns.histplot(df_po2[columns_his[i]], kde=True, bins=10, color='blue', alpha=0.7)
    # plt.xlabel(columns[i])
    plt.ylabel('Frequency')
    # plt.title(f'Histogram of {columns_group[i]}')
    # plt.title(f'{columns_his[i]}')
plt.tight_layout()
plt.show()

#Conclusion: There are many outliers and not normally distributed as can be seen in the diagram.

# Checking the outliers by boxplot

def check_outliers(dataframe):
    ax = sns.boxplot(data=dataframe, orient="h", palette="Set2", whis=1.5)
    plt.show()
    # Checking the outlier
    Q1 = dataframe.quantile(0.25)
    Q3 = dataframe.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q3 - 1.5 * IQR
    outliers = ((dataframe > upper_bound) | (dataframe < lower_bound)).any()
    if outliers.any():
        print("Outliers are:\n ", outliers)
        for column in outliers.index:
            if outliers[column]:
                print(f"There are outliner \n Variables '{column}':")
                print(dataframe[column][dataframe[column] > upper_bound[column]]) or print(dataframe[column][dataframe[column] < lower_bound[column]])
    else:
        print("Non-outliers are:")
check_outliers_result = check_outliers (po2_df)
# 2.3 CHECKING WHETHER SAMPLE IS NORMAL DISTRIBUTION OR NOT
def check_normality(data, alpha=0.05):
    """
    Perform the Kolmogorov-Smirnov test for normality on a DataFrame.

    Parameters:
    - data: A DataFrame containing the dataset.
    - alpha: The significance level (default is 0.05).
    Returns:
    - True if the data follows a normal distribution, False otherwise.
    """
    normality_test_results = {}
    for column in columns_his:
        ks_statistic, p_value = kstest(data[column], 'norm')
        normality_test_results[column] = {'KS Statistic': ks_statistic, 'P-Value': p_value}

    # Check if p-value is less than alpha for any column
    for column, results in normality_test_results.items():
        if results['P-Value'] < alpha:
            print(f"'{column}' does not follow a normal distribution (p-value: {results['P-Value']})")
        else:
            print(f"'{column}' follows a normal distribution (p-value: {results['P-Value']})")

# using function to check the normal distribution
Check_normal_distribution = check_normality(po2_df)

# Checking the distribution between Y= motor_updrs and 16 independent variables

# def scatterplot_matrix(dataframe):
#     y_column_index = 1
#     x_columns = dataframe.columns[2:]
#     num_x_columns = dataframe.shape[1] - 2
#     num_x_columns = len(x_columns)
#     num_rows = (num_x_columns - 1) // 4 + 1
#     num_cols = min(num_x_columns, 4)
#     plt.figure(figsize=(12, 10))
#     for i, x_column in enumerate(x_columns):
#         plt.subplot(num_rows, num_cols, i + 1)
#         x = dataframe[x_column]
#         y = dataframe.iloc[:, y_column_index]
#         plt.scatter(x, y)
#         plt.xlabel(x_column)
#         plt.ylabel(dataframe.columns[y_column_index])
#     plt.tight_layout()
#     plt.show()
# scatterplot_matrix(motor_up_df)
# scatterplot_matrix(total_up_df)