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

# CREATE TWO DATA FRAME TO TEST
# DATA 1 CONTAINS Y = MOTOR_UPDRS AND 18 VARIABLES

motor_columns = [
    'motor_updrs','age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]

motor_up_df = df_po2[motor_columns]
# DATA 2 CONTAINS Y = TOTAL_UPDRS AND 18 VARIABLES

total_columns = [
    'total_updrs', 'age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]

total_up_df = df_po2[total_columns]


def scatter_with_linear_regression(dataframe, test_size=0.4):
    y_column = dataframe.columns[0]  # Select the 1st column as the Y variable
    x_columns = dataframe.columns[1:]  # Select all other columns as X variables

    # Create the figure and main axes
    num_x_columns = len(x_columns)
    num_cols = 4  # Number of columns for subplots
    num_rows = (num_x_columns // num_cols)+1    # Number of rows for subplots
    
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

# Use the function to plot scatter plots with linear lines for all X variables
print("Mortor results: ")
scatter_with_linear_regression(motor_up_df, test_size=0.4)
print("Total results: ")
scatter_with_linear_regression(total_up_df, test_size=0.4)