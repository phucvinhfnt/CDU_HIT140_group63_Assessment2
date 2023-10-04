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
po2_df.loc[:,'RPDE.DFA'] = po2_df[group3].mean(axis=1)
# po2_df.loc[:,'2SET'] = po2_df['jitter(ddp)']/po2_df['shimmer(dda)']
# po2_df.loc[:,'HRN-NHR'] = po2_df['hnr']-po2_df['nhr']
groupJ = ['jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
          'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)']
po2_df.loc[:,'JIT.SHI'] = po2_df[groupJ].mean(axis=1)
groupS = ['shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)','nhr' ]
po2_df.loc[:,'SHI.NHR'] = po2_df[groupS].mean(axis=1)
groupN = ['jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)','nhr' ]
po2_df.loc[:,'JIT.NHR'] = po2_df[groupN].mean(axis=1)
# Create a new group age based on the bar chart in the cleaning.
# We have 6 group: under 30, 30-40,40-50,50-60,60-70, above 70.
# bins = [0, 30, 40, 50, 55,60,65, 70,75, float('inf')]  
# labels = [0, 1, 2, 3, 4, 5,6,7,8]  
bins = [0, 65, float('inf')]  
labels = [0, 1] 
po2_df['age_group'] = pd.cut(po2_df['age'], bins=bins, labels=labels)
print(po2_df)
# 
# df_group = po2_df.groupby(by="subject#")
# df_group.median()   
# new_df = pd.DataFrame(df_group.median())
# new_df.reset_index(inplace=True)
# print("New data:\n", new_df)
# print("Describe New Data:")



# CREATE TWO DATA FRAME TO TEST
# DATA 1 CONTAINS Y = MOTOR_UPDRS AND 18 VARIABLES

motor_columns = [
    'motor_updrs', 
    # 'test_time','sex','age',
    # 'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    # 'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    # 'nhr', 'hnr', 'rpde', 'dfa', 'ppe', 
    # 'nhr', 'hnr', 'rpde', 'dfa', 'ppe','4SET'
    # 'motor_updrs', 
    'test_time','sex','age_group',
    'jitter(abs)', 'jitter(ddp)',
    'shimmer(apq11)','shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe',
    '4SET', '3SET', 'RPDE.DFA','JIT.SHI', 'SHI.NHR', 'JIT.NHR',
]

motor_up_df = po2_df[motor_columns]
print(motor_up_df)
# DATA 2 CONTAINS Y = TOTAL_UPDRS AND 18 VARIABLES

total_columns = [
    # 'total_updrs', 'test_time', 'sex','age',
    # 'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    # 'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    # 'nhr', 'hnr', 'rpde', 'dfa', 'ppe', 
    # 'nhr', 'hnr', 'rpde', 'dfa', 'ppe','4SET'
    'total_updrs', 'test_time','sex','age_group',
    'jitter(abs)', 'jitter(ddp)',
    'shimmer(apq11)','shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe',
    '4SET', '3SET', 'RPDE.DFA','JIT.SHI', 'SHI.NHR', 'JIT.NHR',
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


def train_and_evaluate_linear_regression(dataframe, test_size=0, random_state=42):
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
    fig, ax = plt.subplots()
    x = np.arange(0, len(df_pred), 1)
    ax.scatter(x, df_pred["Actual"], c='b', label="Actual Values")
    ax.scatter(x, df_pred["Predicted"], c='r', label="Predicted Values")
    ax.legend(loc=(1, 0.5))
    ax.set_title('Distribution of Total Prediction')
    plt.show()
    # Print the output
    print("Performance Metrics:")
    print(performance_metrics)
    print("\nPredictions:")
    print(df_pred)
    print("\nModel:")
    print(model_coffi)
    # model_coffi.to_csv("total_Log_model_new.csv")
    

def convert_to_log(dataframe):
    # None return
    if dataframe.empty or len(dataframe.columns) == 0:
        return dataframe
    # Convert to another datafame base on the original data
    df_log = dataframe.copy()
    # Check and computing log for value
    for col in df_log.columns[0:]:
        if col != "sex" and col != "test_time" and col !='age_group':
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
print("Motor sumary")
motor_model_log_summary = linear_regression_summary(motor_log)
print(motor_model_log_summary)
print("Motor sumary")
motor_model_train_summary = train_and_evaluate_linear_regression(motor_log, test_size=0.4)
print(motor_model_train_summary)

print("Total sumary")
total_model_log_summary = linear_regression_summary(total_log)
print(total_model_log_summary)
print("Total sumary")
total_model_train_summary = train_and_evaluate_linear_regression(total_log, test_size=0.4)
print(total_model_train_summary)