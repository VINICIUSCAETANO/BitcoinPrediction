#!/usr/bin/env python
# coding: utf-8


'''Importação de bibliotecas'''

import math
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.arima_model import ARIMA
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


'''Leitura, Exploração e Tratamento de dados'''

# parametros para interpretar corretamente as vírgulas, os "-" como nan e fazer o parse das datas
bitcoin_csv = pd.read_csv("bitcoin_price_training.csv", thousands=',', na_values="-", parse_dates=['Date'], dtype={'Market Cap' : 'float64'})
bitcoin_test = pd.read_csv("bitcoin_price_test.csv", thousands=',', na_values="-", parse_dates=['Date'], dtype={'Market Cap' : 'float64'})

bitcoin_test.info()

bitcoin_csv.head()

bitcoin_csv.describe()

bitcoin_csv.info()


# susbstituir os valores faltantes da coluna Volume pela média da coluna
bitcoin_csv['Volume'].fillna((bitcoin_csv['Volume'].mean()), inplace=True)


# visualização temporal das features
bitcoin_visualization = bitcoin_csv.copy()
bitcoin_visualization['Market Cap'] = bitcoin_visualization['Market Cap'] / 1000000
bitcoin_visualization['Volume'] = bitcoin_visualization['Volume'] / 100000
bitcoin_visualization.set_index('Date', inplace=True)
plt.figure(1)
bitcoin_visualization[['Open', 'Close', 'Low', 'High']].plot(figsize=(20,10), linewidth=1, fontsize=20)
plt.figure(2)
bitcoin_visualization[['Volume', 'Market Cap']].plot(figsize=(20,10), linewidth=2, fontsize=20)
plt.figure(3)
bitcoin_visualization.plot(figsize=(20,10), linewidth=2, fontsize=20)


'''Aplicando Regressão Linear'''


bitcoin_trainregr_x = bitcoin_csv[['Open', 'High', 'Low', 'Volume', 'Market Cap']]
scale(bitcoin_trainregr_x)
bitcoin_trainregr_Y = bitcoin_csv['Close']

lin_regressor = LinearRegression(fit_intercept=False)
scores = cross_val_score(lin_regressor, bitcoin_trainregr_x, bitcoin_trainregr_Y, 
                                                         scoring="neg_mean_squared_error", cv=10)
lin_reg_scores = np.sqrt(-scores)

print(lin_reg_scores)


'''Algoritmos de Sequência Temporal'''

# faz split temporal
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        # separa em input e output
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


data = np.array(bitcoin_csv['Close'])
data = data[::-1]


'''Aplicando LSTM de uma camada'''

n_steps = 7
X, y = split_sequence(data, n_steps)

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

model = Sequential()
#lstm com 25 células
model.add(LSTM(25, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=100, verbose=0)
# demonstrate prediction
x_input = data[-7:]
x_input = x_input.reshape((1, n_steps, n_features))
target = model.predict(x_input)
print(target)


''' Aplicando algoritmo ARIMA'''

#modelo ARIMA para dados nao estacionários
model = ARIMA(data, order=(0, 2, 1))
model_fit = model.fit(disp=False)
# faz predicao
target = model_fit.predict(len(data), len(data), typ='levels')
print(target)
