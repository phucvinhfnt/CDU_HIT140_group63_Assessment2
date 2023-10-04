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
# Create a new group age based on the bar chart in the cleaning.
# We have 2 group: under 65, above 65, which 65 is a mean value of this column.

bins = [0, 65, float('inf')]  
labels = [0, 1] 
po2_df['age_group'] = pd.cut(po2_df['age'], bins=bins, labels=labels)
print(po2_df)




# CREATE TWO DATA FRAME TO TEST
# DATA 1 CONTAINS Y = MOTOR_UPDRS AND 18 VARIABLES

motor_columns = [
    'motor_updrs', 'test_time','sex','age_group',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe', 
  
]

motor_up_df = po2_df[motor_columns]
print(motor_up_df)
# DATA 2 CONTAINS Y = TOTAL_UPDRS AND 18 VARIABLES

total_columns = [
    'total_updrs', 'test_time', 'sex','age_group',
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe', 
    
]

total_up_df = po2_df[total_columns]




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
motor_sta_summary = linear_regression_summary(motor_up_df)
print(motor_sta_summary)
total_sta_summary = linear_regression_summary(total_up_df)
print(total_sta_summary)



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
motor_model_train_summary = train_and_evaluate_linear_regression(motor_up_df, test_size=0.2)
print(motor_model_train_summary)

print("Total sumary")
total_model_train_summary = train_and_evaluate_linear_regression(total_up_df, test_size=0.2)
print(total_model_train_summary)


