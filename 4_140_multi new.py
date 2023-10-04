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

# CREATE TWO DATA FRAME TO TEST
# DATA 1 CONTAINS Y = MOTOR_UPDRS AND 18 VARIABLES

motor_columns = [
    'motor_updrs', 'age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]

motor_up_df = df_po2[motor_columns]
# print(motor_up_df)

# DATA 2 CONTAINS Y = TOTAL_UPDRS AND 18 VARIABLES

total_columns = [
   'total_updrs', 'age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]

total_up_df = df_po2[total_columns]

"""
BUILD AND EVALUATE A LINEAR REGRESSION MODEL
"""
# 3 LINEAR REGRESSION


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
    print("\nPredictions:")
    print(df_pred)
    
    return performance_metrics, model_coffi, df_pred

# Check the model Y = motor_updrs and 16 independent variables
test_sizes = [0.5, 0.4, 0.3, 0.2]

motor_results_df = pd.DataFrame()
motor_model_df = pd.DataFrame()
motor_predictions_df = pd.DataFrame()
# Iterating the model with different test_size in a loop and output the results
for test_size in test_sizes:
    print(f"Testing with test_size = {test_size}")
    performance_metrics,model_coffi, df_pred = train_and_evaluate_linear_regression(motor_up_df, test_size=test_size)
    
    # store the result in DataFrame
    motor_results_df = pd.concat([motor_results_df, performance_metrics], axis=0, ignore_index=True)
    # Store the model in DataFrame
    motor_model_df = pd.concat([motor_model_df, model_coffi], axis=0, ignore_index=True)
    
    # store the prediction in DataFrame
    motor_predictions_df = pd.concat([motor_predictions_df, df_pred], axis=1)
    
    
print("Performance Metrics of motor:")
print(motor_results_df)
print("Motor Model of Motor")
print (motor_model_df)
# motor_model_df.to_csv("Motor model.csv")
print("\nPredictions of Motor:")
print(motor_predictions_df)
# 
# Check the model Y = total_updrs and 18 independent variables
test_sizes = [0.5, 0.4, 0.3, 0.2]
# x_total = total_up_df.iloc[:,2::].values
# # print("X are")
# # print(x_total)
# y_total = total_up_df.iloc[:,1].values
# # print("Y is:")
# # print(y_total)
# # Store data 
total_results_df = pd.DataFrame()
total_model_df = pd.DataFrame()
total_predictions_df = pd.DataFrame()
# Iterating the model with different test_size in a loop and output the results
for test_size in test_sizes:
    print(f"Testing with test_size = {test_size}")
    performance_metrics,model_coffi, df_pred = train_and_evaluate_linear_regression(total_up_df, test_size=test_size)
    
    # store the result in DataFrame
    total_results_df = pd.concat([total_results_df, performance_metrics], axis=0, ignore_index=True)
    # Store the model in DataFrame
    total_model_df = pd.concat([total_model_df, model_coffi], axis=0, ignore_index=True)
    
    # store the prediction in DataFrame
    total_predictions_df = pd.concat([total_predictions_df, df_pred], axis=1)
    
print("Performance Metrics of Total:")
print(total_results_df)
# total_results_df.T.to_csv('total_result.csv')
print("Motor Model of Total")
print (total_model_df)
# total_model_df.T.to_csv('total_model.csv')
print("\nPredictions of Total:")
print(total_predictions_df)
# total_predictions_df.T.to_csv('Total_Prediction.csv')