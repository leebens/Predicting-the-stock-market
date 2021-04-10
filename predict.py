import pandas as pd
from datetime import datetime
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sp = pd.read_csv('/home/dq/scripts/sphist.csv')
sp['Date'] = pd.to_datetime(sp['Date'])

sp = sp.sort_values(by=['Date'], ascending=True)
print(sp["Date"] > datetime(year=2015, month=4, day=1))

#Calculate the mean for the past 5, 30, 365 days
sp['day_5'] = sp['Close'].rolling(5).mean().shift(1)
sp['day_30'] = sp['Close'].rolling(30).mean().shift(1)
sp['day_365'] = sp['Close'].rolling(365).mean().shift(1)

#Calculate the STD for the past 5, 365 days
sp['std_5'] = sp['Close'].rolling(5).std().shift(1)
sp['std_365'] = sp['Close'].rolling(365).std().shift(1)

#Calculate the mean volume for the past 5, 365 days
sp['day_5_volume'] = sp['Volume'].rolling(5).mean().shift(1)
sp['day_365_volume'] = sp['Volume'].rolling(365).mean().shift(1)

#Calculate the STD of the average volume over the past five days
sp['5_volume_std'] = sp['day_5_volume'].rolling(5).std().shift(1)

#remove dates that don't have mean averages (i.e. 5 day and 365 day lead times)
sp = sp[sp["Date"] > datetime(year=1951, month=1, day=2)]
sp = sp.dropna(axis=0)

#split into train and test datasets by year
train = sp[sp['Date'] < datetime(year=2013, month=1, day=1)]
test = sp[sp['Date'] > datetime(year=2013, month=1, day=1)]

#make prediction model and calculate mean squared error
lr = LinearRegression()
lr.fit(train[['day_5','day_30', 'day_365', 'std_5', 'std_365', 'day_5_volume', 'day_365_volume','5_volume_std']],train['Close'])
predictions = lr.predict(test[['day_5','day_30', 'day_365', 'std_5', 'std_365', 'day_5_volume', 'day_365_volume', '5_volume_std']])
mse = mean_squared_error(test['Close'], predictions)

print(mse)