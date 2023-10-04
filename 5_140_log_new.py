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
     'motor_updrs','age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]

motor_up_df = df_po2[motor_columns]
# print(motor_up_df)

# DATA 2 CONTAINS Y = TOTAL_UPDRS AND 18 VARIABLES

total_columns = [
    'total_updrs','age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]

total_up_df = df_po2[total_columns]

"""
BUILD AND EVALUATE A LINEAR REGRESSION MODEL
"""
# 4 LOGRARIT LINEAR REGRESSION

"""
Because all of explanatory variables and the response variable are not normally distributed. 
So that, build the model with all variales in lograrit to check the result of each model.
"""

def linear_regression_summary(dataframe):
    # x_column = dataframe.columns[2]
    # x = dataframe[x_column].values.reshape(-1, 1)
    x = dataframe.iloc[:,1::]
    y = dataframe.iloc[:,0]
    # Add a constant term to the input features (x)
    x = sm.add_constant(x)
    # Build the linear regression model
    model = sm.OLS(y, x).fit()
    # Generate predictions from the model
    pred = model.predict(x)
    # Get the summary details of the model
    model_details = model.summary()
    return model_details

# Using function and print the results
motor_model_summary = linear_regression_summary(motor_up_df)
print(motor_model_summary)
total_model_sumary = linear_regression_summary(total_up_df)
print(total_model_sumary)

"""Non-linear transformation of all variables, except sex and test_time column"""

def convert_to_log(dataframe):
    # None return
    if dataframe.empty or len(dataframe.columns) == 0:
        return dataframe
    # Convert to another datafame base on the original data
    df_log = dataframe.copy()
    # Check and computing log for value
    for col in df_log.columns[0:]:
        if col != "sex" and col != "test_time":
            if "LOG" not in col:
                # Computing from the first column
                df_log[col] = np.log(df_log[col])
                # Rename after log
                df_log.rename(columns={col: col + '_LOG'}, inplace=True)
    return df_log

# Using function to change log value

motor_log = convert_to_log(motor_up_df)
# motor_log.T.to_csv('motor_log.csv')
total_log = convert_to_log(total_up_df)
def scatter_with_linear_regression(dataframe, test_size=0.4):
    y_column = dataframe.columns[0]  # Select the 1st column as the Y variable
    x_columns = dataframe.columns[1:]  # Select all other columns as X variables

    # Create the figure and main axes
    num_x_columns = len(x_columns)
    num_cols = 4  # Number of columns for subplots
    num_rows = (num_x_columns // num_cols)+1   # Number of rows for subplots
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    
    # Iterate a dataFrame to store the results
    # performance_single = pd.DataFrame(columns=["Variable", "Test Size", "MAE", "MSE", "RMSE", "RMSE (Normalized)", "R^2"])
    performance_list = []
    
    # Iterate through X variables and plot scatter plots with linear lines on subplots
    for i, x_column in enumerate(x_columns):
        ax = axes[i // num_cols, i % num_cols]  # Get the corresponding subplot
        plt.subplot(num_rows, num_cols, i+1)
        x = dataframe[x_column].values.reshape(-1, 1)
        y = dataframe[y_column].values.reshape(-1, 1)
        
        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        # Initialize the Linear Regression model
        model = LinearRegression()

        # Fit the model
        model.fit(x_train, y_train)

        # Predict Y values based on the model
        y_pred = model.predict(x_test)
        # Compute standard performance metrics of the linear regression:
        # Plot the scatter plot
        sns.scatterplot(x=x_test.flatten(), y=y_test.flatten(), ax=ax)
        # sns.scatterplot(x=x_column, y=y_column, data=dataframe, label='Data', ax=ax)

        # Plot the linear line
        ax.plot(x_test.flatten(), y_pred, color='red', label='Linear Line')
        # ax.plot(dataframe[x_column], y_pred, color='red', label='Linear Line')

        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        # ax.set_title(f'Scatter with Linear Regression for {x_column} vs {y_column}')
        # ax.legend()
        # Mean Absolute Error
        mae = metrics.mean_absolute_error(y_test, y_pred)
        # Mean Squared Error
        mse = metrics.mean_squared_error(y_test, y_pred)
        # Root Mean Square Error
        rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))
        # Normalised Root Mean Square Error
        y_max = y_test.max()
        y_min = y_test.min()
        rmse_norm = rmse / (y_max - y_min)
        # Calculate R-squared
        r_squared = model.score(x_test, y_test)
        
        # Store the performance metrics results in DataFrame
        performance_data = {"Variable": [x_column], "Test Size": [test_size], "MAE": [mae], "MSE": [mse], "RMSE": [rmse], "RMSE (Normalized)": [rmse_norm], "R^2": [r_squared]}
        performance_single = pd.concat([pd.DataFrame(performance_data)], ignore_index=True)
        # Append the performance DataFrame to the list
        performance_list.append(performance_single)
       
    # Concatenate all performance DataFrames into a single DataFrame
    performance_single = pd.concat(performance_list, ignore_index=True)
    print(performance_single)
      
    plt.tight_layout()
    plt.show()

print("Mortor results: ")
scatter_with_linear_regression(motor_log, test_size=0.4)
print("Motor sumary")
motor_model_log_summary = linear_regression_summary(motor_log)
print(motor_model_log_summary)
print("Total results: ")

scatter_with_linear_regression(total_log, test_size=0.4)
print("Total sumary")
total_model_log_summary = linear_regression_summary(total_log)
print(total_model_log_summary)
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
    


print("Motor sumary")
motor_model_train_summary = train_and_evaluate_linear_regression(motor_log, test_size=0.4)
print(motor_model_train_summary)

print("Total sumary")
total_model_train_summary = train_and_evaluate_linear_regression(total_log, test_size=0.4)
print(total_model_train_summary)




