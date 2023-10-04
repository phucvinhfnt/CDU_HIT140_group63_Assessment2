# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score  # Import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import kstest
from sklearn import metrics
import math
import statsmodels.api as sm
from sklearn.model_selection import KFold
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
     'motor_updrs','subject#', 'age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]

motor_up_df = df_po2[motor_columns]
print(motor_up_df)

# DATA 2 CONTAINS Y = TOTAL_UPDRS AND 18 VARIABLES

total_columns = [
    'total_updrs', 'subject#','age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]

total_up_df = df_po2[total_columns]

"""
BUILD AND EVALUATE A LINEAR REGRESSION MODEL
"""


def train_and_evaluate_linear_regression(dataframe, test_size=0, random_state=None, kf=None):
    """
    # Split dataset into 50%,60%,70%,80% training and 50%,40%,30%,20% test sets, respectively
    # Note: other % split can be used.
    """
    x = dataframe.iloc[:,1::].values
    y = dataframe.iloc[:,0].values
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    if kf is not None:
        # Perform K-Fold Cross Validation
        mae_scores = []
        mse_scores = []
        rmse_scores = []
        rmse_norm_scores = []
        r_2_scores = []
        adj_r_2_scores = []
        
        for train_index, test_index in kf.split(x):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
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
            mae_scores.append(mae)
            # Mean Squared Error
            mse = metrics.mean_squared_error(y_test, y_pred)
            mse_scores.append(mse)
            # Root Mean Square Error
            rmse =  math.sqrt(mse)
            rmse_scores.append(rmse)
            # Normalised Root Mean Square Error
            y_max = y.max()
            y_min = y.min()
            rmse_norm = rmse / (y_max - y_min)
            rmse_norm_scores.append(rmse_norm)
            # R-Squared
            r_2 = metrics.r2_score(y_test, y_pred)
            r_2_scores.append(r_2)
            # Adjusted R-Squared
            n = len(y_test)  # Number of data points
            p = X_test.shape[1]  # Number of predictors (independent variables)
            adj_r_2 = 1 - ((1 - r_2) * (n - 1) / (n - p - 1))
            adj_r_2_scores.append(adj_r_2)
        
        # Calculate the average scores over all folds
        avg_mae = sum(mae_scores) / len(mae_scores)
        avg_mse = sum(mse_scores) / len(mse_scores)
        avg_rmse = sum(rmse_scores) / len(rmse_scores)
        avg_rmse_norm = sum(rmse_norm_scores) / len(rmse_norm_scores)
        avg_r_2 = sum(r_2_scores) / len(r_2_scores)
        avg_adj_r_2 = sum(adj_r_2_scores) / len(adj_r_2_scores)
        
        # Store the performance metrics results in DataFrame
        performance_metrics = pd.DataFrame({
            "Test Size": [test_size],
            "MAE": [avg_mae],
            "MSE": [avg_mse],
            "RMSE": [avg_rmse],
            "RMSE (Normalized)": [avg_rmse_norm],
            "R^2": [avg_r_2],
            "Adjusted R^2": [avg_adj_r_2]
        })
    else:
        # If kf is not provided, perform a single train-test split
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


print("Motor sumary")
motor_model_train_summary = train_and_evaluate_linear_regression(motor_up_df, test_size=0.4)
# print(motor_model_train_summary)

print("Total sumary")
total_model_train_summary = train_and_evaluate_linear_regression(total_up_df, test_size=0.4)
# print(total_model_train_summary)