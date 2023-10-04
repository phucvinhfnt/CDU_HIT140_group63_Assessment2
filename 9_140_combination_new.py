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
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

#0 INPUT DATA 1

columns1 = [
    'subject#', 'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    "Harmonicity13", 'nhr', 'hnr', "Pitch16", "Pitch17", "Pitch18", "Pitch19", "Pitch20",
    "Pulse21", "Pulse22", "Pulse23", "Pulse24", "Voice25", "Voice26", "Voice27", "motor_updrs",
    "PDindicator29",
]
df1 = pd.read_csv('po1_data.txt', sep=",", header=None, names=columns1 )
df1 = pd.DataFrame(df1, columns=columns1)
df_1 = df1[df1["PDindicator29"] !=0]
# df_1.to_csv("Data1.csv")
# print(df_1)
columns2 = [
    'subject#', 'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    "Harmonicity13", 'nhr', 'hnr', "Pitch16", "Pitch17", "Pitch18", "Pitch19", "Pitch20",
    "Pulse21", "Pulse22", "Pulse23", "Pulse24", "Voice25", "Voice26", "Voice27", "total_updrs",
    "PDindicator29",
]
df2 = pd.read_csv('po1_data.txt', sep=",", header=None, names=columns2 )
df2 = pd.DataFrame(df2, columns=columns2)
df_2 = df2[df2["PDindicator29"] !=0]
# print(df_2)
# O INPUT DATA 2
df_po2 = pd.read_csv("po2_data.csv")
# Reorder columns of df_po2
df_po2_columns = [
    'subject#', 'motor_updrs','total_updrs', 'age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]
po2_df = df_po2[df_po2_columns]

fig, ax = plt.subplots(1,1)
po2_df["motor_updrs"].plot(kind="density", ax=ax, label="Motor UPDRS")
po2_df["total_updrs"].plot(kind="density", ax=ax, label="Total UPDRS")
df_1["motor_updrs"].plot(kind="density", ax=ax, label="UPDRS")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title("Density Plot")
ax.legend()
plt.show()
# When inspecting two data po1 and po2, we get same columns 
# "Jitter2", "Jitter3", "Jitter4", "Jitter5", "Jitter6", 
# "Shimmer7", "Shimmer8", "Shimmer9", "Shimmer10", "Shimmer11", "Shimmer12", 
# "Harmonicity14", "Harmonicity15"
# CREATE TWO DATA FRAME TO TEST


# DATA 0 CONTAINS 13 VARIABLES
po1_columns = [
    'motor_updrs',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(apq11)','hnr'
    # 'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    #  'nhr', 'hnr',
]
po1_test = df_1[po1_columns]
po1_2_columns = [
    'total_updrs',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(apq11)','hnr'
    # 'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    # 'nhr', 'hnr',
]
po1_test2 = df_2[po1_2_columns]



# print(po1_test)
# DATA 1 CONTAINS Y = MOTOR_UPDRS AND 13 VARIABLES

motor_columns = [
    'motor_updrs', 
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(apq11)','hnr'
    # 'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    # 'nhr', 'hnr', 
]

motor_train_df = df_po2[motor_columns]
# DATA 2 CONTAINS Y = TOTAL_UPDRS AND 13 VARIABLES

total_columns = [
    'total_updrs', 
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(apq11)','hnr'
    # 'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    # 'nhr', 'hnr', 
]

total_train_df = df_po2[total_columns]


train_test_1 = pd.concat([motor_train_df,po1_test])
# print(train_test_1)
test_size1 = len(po1_test)/len(motor_train_df)
# print(test_size1)


train_test_2 = pd.concat([total_train_df,po1_test2])
# print(train_test_2)
test_size2 = len(po1_test2)/len(total_train_df)
# print(test_size2)


def train_and_evaluate_linear_regression(dataframe, test_size=0, random_state=None):
    """
    # Split dataset into 50%,60%,70%,80% training and 50%,40%,30%,20% test sets, respectively
    # Note: other % split can be used.
    """
    x = dataframe.iloc[:,1::].values
    y = dataframe.iloc[:,0].values
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    # Build a linear regression model
    model = LinearRegression()
    # Train (fit) the linear regression model using the training set
    model.fit(X_train, y_train)
    
    # Use linear regression to predict the values of (y) in the test set
    # based on the values of x in the test set
    y_pred = model.predict(X_test)
    
    # Compute standard performance metrics of the linear regression:
    # Mean Absolute Error
    mae = metrics.mean_absolute_error(y_test, y_pred)
    # Mean Squared Error
    mse = metrics.mean_squared_error(y_test, y_pred)
    # Root Mean Square Error
    rmse =  math.sqrt(mse)
    # Normalised Root Mean Square Error
    y_max = y.max()
    y_min = y.min()
    rmse_norm = rmse / (y_max - y_min)
    # R-Squared
    r_2 = metrics.r2_score(y_test, y_pred)
    # Adjusted R-Squared
    n = len(y_test)  # Number of data points
    p = X_test.shape[1]  # Number of predictors (independent variables)

    adj_r_2 = 1 - ((1 - r_2) * (n - 1) / (n - p - 1))

    
    # Store the performance metrics results in DataFrame
    performance_metrics = pd.DataFrame({
        "Test Size": [test_size],
        "MAE": [mae],
        "MSE": [mse],
        "RMSE": [rmse],
        "RMSE (Normalized)": [rmse_norm],
        "R^2": [r_2],
        "Adjusted R^2": [adj_r_2]
    })
    # Print the intercept and coefficient learned by the linear regression model
    model_coffi = pd.DataFrame({
        "Test Size": [test_size],
        "Intercept": [model.intercept_],
        "Coefficient": [model.coef_]
        })
    # Optional: Show the predicted values of (y) next to the actual values of (y)
    df_pred = pd.DataFrame({
        "Actual": y_test, 
        "Predicted": y_pred})
    
    # Print the output
    print("Performance Metrics:")
    print(performance_metrics)
    print("Model:")
    print(model_coffi)
    print("\nPredictions:")
    print(df_pred)
    
    return performance_metrics, model_coffi, df_pred


# Iterating the model with different test_size in a loop and output the results
print("Motor UPDRS result:")
result_motor_com = train_and_evaluate_linear_regression(train_test_1,test_size=test_size1)
print("Total UPDRS result:")
result_total_com = train_and_evaluate_linear_regression(train_test_2,test_size=test_size2)