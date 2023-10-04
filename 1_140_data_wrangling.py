# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import kstest
import math
#0 INPUT DATA
df_po2 = pd.read_csv("po2_data.csv")
print("info data:")
df_po2.info()

#1 CLEANING DATA 
print("5 lines of data")
df_po2.head()
print(df_po2.head())

print("Update data when remove duplicate:")
df_po2.drop_duplicates(inplace=True)
print("DataFrame update:\n", df_po2)

print("Checking missing value")
missing_values = df_po2.isnull().sum()
print("Missing Values:", missing_values)

print("Shape of data")
print(df_po2.shape)

# 2 DESCRIPTIVE ANALYSIS
print("Describe data")
def describe_dataFrame(data_df):
    des_data = data_df.describe()
    des_data.loc['range'] = data_df.max() - data_df.min()
    des_data.loc['median'] = data_df.median()
    des_data.loc['variance'] = data_df.var()
    des_data.loc['Standard Deviation'] = data_df.std()
    des_data.loc['IQR'] = data_df.quantile(0.75) - data_df.quantile(0.25)
    print(des_data)
    return data_df
des_results=describe_dataFrame(df_po2)

# 2.1 DESCRIPTIVE ANALYSIS INDEPT
# Check value in 4 columns to count the frequency of each value: 
# Subject#: Identifying 42 people with early-stage Parkinson's disease with the frequency testing
# age: Group the same age and count how many people are in this group
# sex: Separating into 2 group male 1 and female 0
# Test time: To check the frequency of testing 

columns_fre = df_po2.columns[0:4]
len_columns_fre = len(columns_fre)
fig, axes = plt.subplots(2, 2, figsize=(16, 6))
for i, col in enumerate(columns_fre):
    count =df_po2[col].value_counts()
    print(f"Value in\n '{col}': Frequency\n {count}")
    # plt.figure(figsize=(8, 6))
    row = i // 2  
    col = i % 2
    axes[row, col].bar(count.index, count.values)
    axes[row, col].set_title(f'Bar Chart for {columns_fre[i]}')
    axes[row, col].set_xlabel(col)
    axes[row, col].set_ylabel('Frequency')
plt.tight_layout()
plt.show()
    # .T.to_csv('descriptive statistics.csv')
df_po2.boxplot(column='motor_updrs', by='age')
df_po2.boxplot(column='total_updrs', by='age')
plt.show()

fig, ax = plt.subplots(1,1)
df_po2["motor_updrs"].plot(kind="density", ax=ax, label="Motor UPDRS")
df_po2["total_updrs"].plot(kind="density", ax=ax, label="Total UPDRS")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title("Density Plot")
ax.legend()
plt.show()

# After checking 4 columns, we decided group all the dataset by subject#.
# Other columns will be calculated by getting mean in each value of subject#
# Establishing the new data to analysing
df_group = df_po2.groupby(by="subject#")
df_group.mean()   
new_df = pd.DataFrame(df_group.mean())
new_df.reset_index(inplace=True)
print("New data:\n", new_df)
print("Describe New Data:")
des_group_results=describe_dataFrame(new_df)


# Average Motor UPDRS
plt.figure(figsize=(14, 5))
sns.lineplot(data=new_df, x="subject#", y="motor_updrs", color="blue")
plt.title("Average Motor UPDRS of Subjects")
plt.ylabel("Average Motor UPDRS")
plt.xlabel("Subject#")
plt.show()

# Average Total UPDRS
plt.figure(figsize=(14, 5))
sns.lineplot(data=new_df, x="subject#", y="total_updrs", color="red")
plt.title("Average Total UPDRS of Subjects")
plt.ylabel("Average Total UPDRS")
plt.xlabel("Subject#")
plt.show()


