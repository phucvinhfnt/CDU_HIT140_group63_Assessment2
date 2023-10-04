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
# DATA 2 CONTAINS Y = TOTAL_UPDRS AND 18 VARIABLES

total_columns = [
    'total_updrs', 'age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]

total_up_df = df_po2[total_columns]



def calculate_correlation(df):
    df = df.iloc[:,0::]
    # Computing the matrix correlation
    correlation_matrix = df.corr()
    correlation_matrix = correlation_matrix.round(2)
    # Plot the pairwise correlation as heatmap
    plt.figure(figsize=(8, 6))
    ax= sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, center=0,
                cmap=sns.diverging_palette(20, 220, n=200), square=False, linewidths=0.5)
    plt.title("Correlation Heatmap")
    # customise the labels
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
        )

    # Find strongly correlation variables from 0.07
    strong_correlations = correlation_matrix.iloc[0].abs() >= 0.07
    
    # Make a new list to store correlated variables
    correlated_variables = [col for col in correlation_matrix.columns[strong_correlations].tolist() if col != 'motor_updrs' and col != 'total_updrs' ]

    correlated_variables = list(set(correlated_variables))

    return correlation_matrix, correlated_variables
# Check the corrlation of motor and 18 variables
mot_correlation_matrix, mot_correlated_variables = calculate_correlation(motor_up_df)
print("Motor Correlation Matrix:")
print(mot_correlation_matrix)
print("\nMotor Correlated Variables:")
print(mot_correlated_variables)
plt.show()

# Check the corrlation of total and 18 variables
tol_correlation_matrix, tol_correlated_variables = calculate_correlation(total_up_df)
print("Correlation Matrix:")
print(tol_correlation_matrix)
print("\nCorrelated Variables:")
print(tol_correlated_variables)
plt.show()


# Print the new data with significant variables
motor_cor_df = motor_up_df[["motor_updrs"]+mot_correlated_variables]
print(motor_cor_df)


total_cor_df = total_up_df[["total_updrs"]+tol_correlated_variables]
print(total_cor_df)
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

# Using function and print the results of new df contains strong relationship variables
motor_cor_summary = linear_regression_summary(motor_cor_df)
print(motor_cor_summary)
total_cor_summary = linear_regression_summary(total_cor_df)
print(total_cor_summary)


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
motor_model_train_summary = train_and_evaluate_linear_regression(motor_cor_df, test_size=0.4)
print(motor_model_train_summary)

print("Total sumary")
total_model_train_summary = train_and_evaluate_linear_regression(total_cor_df, test_size=0.4)
print(total_model_train_summary)
