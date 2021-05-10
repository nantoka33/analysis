import pandas as pd
import numpy as np

df_data = pd.read_csv('to_csv.csv')
df_future = pd.read_csv('to_future_csv.csv')

df_data.drop(columns='Unnamed: 0', inplace=True)
df_future.drop(columns=['Unnamed: 0','Impact'], inplace=True)

print(df_data)
print(df_future)

X_name = ['Upper and lowercase','Character length','Number length','Number of special characters','Number of words','General regularity','Special regularity','Commonly used passwords']
x = df_data[X_name]
y = df_data['Impact']


import statsmodels.api as sm

#model = sm.OLS(y,sm.add_constant(x))
model = sm.OLS(y,x)
result = model.fit()

print(result.summary())

df_result = result.predict(df_future[X_name])
df_future["Result"] = df_result
df_result_data = result.predict(df_data[X_name])
df_data["Result"] = df_result_data


df_future.to_csv('to_future_result.csv')
df_data.to_csv('to_data_result.csv')