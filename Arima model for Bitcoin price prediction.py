!pip install yfinance

import numpy as np,pandas as pd, matplotlib.pyplot as plt
import math
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error,mean_absolute_error
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

bitstamp=yf.download('BTC-USD')

bitstamp=bitstamp.resample("24h").agg({
    "Open":"first",
    "High":"max",
    "Low":"min",
    "Close":"last"
})

def fill_missing(df):
    ### function to impute missing values using interpolation ###
    df['Open'] = df['Open'].interpolate()
    df['Close'] = df['Close'].interpolate()
    df['High'] = df['High'].interpolate()
    df['Low'] = df['Low'].interpolate()

    print(df.head())
    print(df.isnull().sum())

fill_missing(bitstamp)

bitstamp_non_indexed = bitstamp.copy()

bitstamp.shape
bitstamp

df = bitstamp

to_row=int(len(df)-730)
print(to_row)
training_data=list(df[0:to_row]['Close'])
testing_data=list(df[to_row:]['Close'])
print(len(training_data))
print(len(testing_data))


import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

  # warnings.warn(ARIMA_DEPRECATION_WARN, FutureWarning)

model_predictions=[]
n_test_obser=len(testing_data)

for i in range(n_test_obser):
  model=ARIMA(training_data,order =(5,2,0))
  model_fit=model.fit()
  output=model_fit.forecast()
  yhat=list([output[0]])[0]
  model_predictions.append(yhat)
  actual_test_value=testing_data[i]
  training_data.append(actual_test_value)

print(model_fit.summary())

plt.figure(figsize=(15,9))
plt.grid(True)
plt.plot(model_predictions, color = 'red', label = 'Predicted testing Bitcoin Price')
plt.plot(testing_data, color = 'green', label = 'Real testing Bitcoin Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

mape=np.mean(np.abs(np.array(model_predictions)-np.array(testing_data))/np.abs(testing_data))
print("MAPE:  "+str(mape))

import math 
mse = mean_squared_error(testing_data, model_predictions, squared=False)
mse


rmse=math.sqrt(mse)
rmse

import seaborn as sns

fig = plt.figure()
sns.distplot((np.array(testing_data) - np.array(model_predictions)), bins =20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 12)   

from sklearn.metrics import r2_score
a=r2_score(testing_data,  model_predictions)
print(a*100)

model_prediction=[]
n_test_obser=len(training_data)
for i in range(n_test_obser):
  model1=ARIMA(training_data,order =(5,2,0))
  model_fit1=model1.fit()
  output=model_fit1.forecast()
  yhat=list([output[0]])[0]
  model_prediction.append(yhat)
  actual_test_value1=training_data[i]
  training_data.append(actual_test_value1)

plt.figure(figsize=(15,9))
plt.grid(True)
plt.plot(model_prediction, color = 'red', label = 'Predicted training Bitcoin Price')
plt.plot(training_data, color = 'green', label = 'Real training Bitcoin Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# mape1=np.mean(np.abs(np.array(model_prediction)-np.array(training_data))/np.abs(training_data))
# print("MAPE:  "+str(mape1))
from sklearn.metrics import mean_absolute_percentage_error
mape1= mean_absolute_percentage_error(df['Close'], model_prediction)
print("MAPE:  "+str(mape1))

import math 
mse1 = mean_squared_error(df['Close'], model_prediction, squared=False)
mse1


rmse1=math.sqrt(mse1)
rmse1

from sklearn.metrics import r2_score
a1=r2_score(df['Close'],  model_prediction)
print(a*100)

import seaborn as sns

fig = plt.figure()
sns.distplot((np.array(df['Close']) - np.array(model_prediction)), bins =20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 12)   

import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(header=dict(values=['classifications','MSE', 'RMSE','MAPE','R2 score']),
                 cells=dict(values=[['Testing data','training_data'], [mape,mape1], [mse,mse1], [rmse,rmse1], [a,a1]]))
                     ])
fig.show()






