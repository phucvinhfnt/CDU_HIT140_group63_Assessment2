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

#0 INPUT DATA
df_po2 = pd.read_csv("po2_data.csv")
# Reorder columns of df_po2
df_po2_columns = [
    'subject#', 'motor_updrs','total_updrs', 'age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]

po2_df = df_po2[df_po2_columns].copy()





# CREATE TWO DATA FRAME TO TEST
# DATA 1 CONTAINS Y = MOTOR_UPDRS AND 18 VARIABLES

motor_columns = [
    'motor_updrs', 
    'test_time','sex','age',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe', 
    # 'nhr', 'hnr', 'rpde', 'dfa', 'ppe','4SET'
    # 'motor_updrs', 
]

motor_up_df = po2_df[motor_columns]
print(motor_up_df)
# DATA 2 CONTAINS Y = TOTAL_UPDRS AND 18 VARIABLES

total_columns = [
    'total_updrs', 
    'test_time', 'sex','age',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe', 
    # 'nhr', 'hnr', 'rpde', 'dfa', 'ppe','4SET'
    # 'total_updrs', 'test_time','sex','age_group',
    # 'jitter(abs)', 'jitter(ddp)',
    # 'shimmer(apq11)','shimmer(dda)',
    # 'nhr', 'hnr', 'rpde', 'dfa', 'ppe',
    # '4SET', '3SET', 'RPDE.DFA','JIT.SHI', 'SHI.NHR', 'JIT.NHR' 
]

total_up_df = po2_df[total_columns]


def train_and_evaluate_linear_regression(dataframe, test_size=0.4, random_state=0):
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
    # """
    # COMPARE THE PERFORMANCE OF THE LINEAR REGRESSION MODEL
    # VS.
    # A DUMMY MODEL (BASELINE) THAT USES MEAN AS THE BASIS OF ITS PREDICTION
    # """

    # Compute mean of values in (y) training set
    y_base = np.mean(y_train)

    # Replicate the mean values as many times as there are values in the test set
    y_pred_base = [y_base] * len(y_test)


    # Optional: Show the predicted values of (y) next to the actual values of (y)
    df_base_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_base})
    print(df_base_pred)

    # Compute standard performance metrics of the baseline model:

    # Mean Absolute Error
    mae = metrics.mean_absolute_error(y_test, y_pred_base)
    # Mean Squared Error
    mse = metrics.mean_squared_error(y_test, y_pred_base)
    # Root Mean Square Error
    rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred_base))

    # Normalised Root Mean Square Error
    y_max = y.max()
    y_min = y.min()
    rmse_norm = rmse / (y_max - y_min)

    # R-Squared
    r_2 = metrics.r2_score(y_test, y_pred_base)
    # # Adjusted R-Squared
    n = len(y_test)  # Number of data points
    p = X_test.shape[1]  # Number of predictors (independent variables)

    adj_r_2 = 1 - ((1 - r_2) * (n - 1) / (n - p - 1))

    print("Baseline performance:")
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("RMSE (Normalised): ", rmse_norm)
    print("R^2: ", r_2)
    print("Adjusted R^2: ", adj_r_2)
    


print("BASELINE MODEL OF MOTOR_UPDRS")
motor_model_train_summary = train_and_evaluate_linear_regression(motor_up_df, test_size=0.4)
print(motor_model_train_summary)

print("BASELINE MODEL OF TOTAL_UPDRS")
total_model_train_summary = train_and_evaluate_linear_regression(total_up_df, test_size=0.4)
print(total_model_train_summary)



