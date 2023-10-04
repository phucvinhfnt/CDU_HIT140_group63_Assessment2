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
group1 = ['hnr','rpde', 'dfa', 'ppe']
po2_df.loc[:,'4SET'] = po2_df[group1].mean(axis=1)
group2 = ['rpde', 'dfa', 'ppe']
po2_df.loc[:,'3SET'] = po2_df[group2].mean(axis=1)
group3 = ['rpde', 'dfa']
po2_df.loc[:,'2SET'] = po2_df['jitter(abs)']*po2_df['shimmer(abs)']
po2_df.loc[:,'DFAPPE'] = po2_df['dfa']-po2_df['ppe']
groupJ = ['jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)']
po2_df.loc[:,'JITTER'] = po2_df[groupJ].mean(axis=1)
groupS = ['shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)']
po2_df.loc[:,'SHIMMER'] = po2_df[groupS].mean(axis=1)
# print(po2_df)

# CREATE TWO DATA FRAME TO TEST
# DATA 1 CONTAINS Y = MOTOR_UPDRS AND 18 VARIABLES

motor_columns = [
    'subject#', 'motor_updrs', 'age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe','4SET','3SET','2SET','DFAPPE', 'JITTER', 'SHIMMER'
    # ,,'DFAPPE',
]

motor_up_df = po2_df[motor_columns]
print(motor_up_df)
# DATA 2 CONTAINS Y = TOTAL_UPDRS AND 18 VARIABLES

total_columns = [
    'subject#', 'total_updrs', 'age', 'sex', 'test_time',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe', '4SET', '3SET','2SET','DFAPPE', 'JITTER', 'SHIMMER'
    # ,'2SET',
    
]

total_up_df = po2_df[total_columns]

def train_and_evaluate_linear_regression(dataframe, test_size=0, random_state=None):
    """
    # Split dataset into 50%,60%,70%,80% training and 50%,40%,30%,20% test sets, respectively
    # Note: other % split can be used.
    """
    x = dataframe.iloc[:,2::].values
    y = dataframe.iloc[:,1].values
    
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
# result_motor_com = train_and_evaluate_linear_regression(train_test_1,test_size=test_size1)
# result_total_com = train_and_evaluate_linear_regression(train_test_2,test_size=test_size2)


def gaussian_transform(dataframe):
    """
    Apply the Yeo-Johnson transformation to make explanatory variables more Gaussian-looking.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing explanatory variables.

    Returns:
    pd.DataFrame: A DataFrame with variables transformed using the Yeo-Johnson method.
    """

    # Initialize the PowerTransformer with the Yeo-Johnson method
    scaler = PowerTransformer(method='yeo-johnson', standardize=False)

    # Drop the previously added constant column 'const' (if present)
    if 'const' in dataframe.columns:
        dataframe = dataframe.drop(["const"], axis=1)

    # Apply the Yeo-Johnson transformation to make variables more Gaussian-looking
    transformed_data = scaler.fit_transform(dataframe)

    # Restore column names of explanatory variables
    transformed_dataframe = pd.DataFrame(transformed_data, index=dataframe.index, columns=dataframe.columns)

    return transformed_dataframe


gau_mot_df = gaussian_transform(motor_up_df)
print(gau_mot_df)
gau_tot_df = gaussian_transform(total_up_df)
print(gau_tot_df)
def linear_regression_summary(dataframe):
    # x_column = dataframe.columns[2]
    # x = dataframe[x_column].values.reshape(-1, 1)
    x = dataframe.iloc[:,2::]
    y = dataframe.iloc[:,1]
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
print("Gaussian model")
motor_gau_summary = linear_regression_summary(gau_mot_df)
print(motor_gau_summary)
total_gau_summary = linear_regression_summary(gau_tot_df)
print(total_gau_summary)

print("Train and Test model")

result_motor_gau_com = train_and_evaluate_linear_regression(gau_mot_df,test_size=0.4)
result_total_gau_com = train_and_evaluate_linear_regression(gau_tot_df,test_size=0.4)