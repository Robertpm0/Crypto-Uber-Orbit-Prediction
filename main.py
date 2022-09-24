from datetime import datetime
from fileinput import close
from http.client import responses
from time import time
from turtle import color
import pandas as pd
from orbit.models.lgt import LGTFull
from orbit.utils.dataset import load_iclaims
from orbit.diagnostics.plot import plot_predicted_data, plot_predicted_components,plot_posterior_params
import orbit
import matplotlib.pyplot as plt
import  matplotlib


import requests
import numpy as  np
import  talib


matplotlib.use('Qt4Agg')


#$###  TO DO ADD  MORE REGRESSORS AND FEATURE IMPORTANCE AND DIFFERENT DATA SET



def main():

    print(orbit.__version__)
    coin = input("enter a Token:",)
    ##print("Available Params: \n close vix smix mayer rsi macd rolling rolling2 rolling 3")
    #xxxregressor_col = ("enter params:", [])
    f = requests.get(f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={coin}&tsym=USD&limit=2000").json()['Data']['Data']
    g = pd.DataFrame(f)
    print(g)
    df = g[['time','high', 'low', 'volumeto', 'close']]

    df['daily_change'] = df['high'].pct_change()
    round(df['daily_change'],2).quantile(0.05)
    df['vix'] = df['daily_change'].rolling(200).std()*(200**0.5)
    df['smix'] = (df['close'] / df['close'].rolling(45).mean()) / (df['close'].rolling(15).mean() / df['close'].rolling(45).mean())
    df['mayer'] = df['close'] / df['close'].rolling(180).mean()
    df['rolling'] = df['close'].rolling(60).mean()
    df['rolling2'] = df['close'].rolling(200).mean()
    df['rolling3'] = df['close'].rolling(7).mean()
    df['rsi'] = talib.RSI(df['close'],14)
    df['macd'],df['macd2'],df['macd3'] = talib.MACD(df['close'],fastperiod=12,slowperiod=26,signalperiod=9)
    df.dropna(inplace=True)
    print(df)


    df.time = pd.to_datetime(df.time,unit='s')
    prediciotnDays = int(15)
    df['preds'] = df['close'].shift(-prediciotnDays)
    regressor_col=['close','vix','volumeto','rsi','high','low','mayer']
    response_col='preds'
    print(df)
    
    train_df2 = df[:-prediciotnDays]
    train_df = train_df2.copy()
    train_df = pd.DataFrame(train_df)

    train_df.loc[:,regressor_col] = np.log(train_df.loc[:,regressor_col]).diff()
    train_df.loc[:,regressor_col].iloc[0] = np.log(train_df.loc[:,regressor_col].iloc[0])
    train_df = train_df.drop(index=200)
    #train_df.loc[:,response_col] = np.log(train_df.loc[:,response_col]).diff()
    #train_df.loc[:,response_col].iloc[0] = np.log(train_df.loc[:,response_col].iloc[0])
    print(train_df)

    x = df.drop(['preds'],1)

    pred_df =  x[-prediciotnDays:]
    pred_df.loc[:,regressor_col] = np.log(pred_df.loc[:,regressor_col]).diff()
    pred_df = pred_df.drop(index=1986)
    #pred_df.loc[:,regressor_col].iloc[0] = np.log(pred_df.loc[:,regressor_col].iloc[0])
    print(pred_df)

    #print(train_df)
    #print(test_df)
    #print(df.dtypes)
  
    ets = LGTFull(
        response_col='preds',
        date_col= 'time',
        #regressor_col=['close','vix','volumeto','rsi','mayer'],
        #regressor_sign=['=','-','+','+','+'],
        seasonality=90,
        seed=2364,
    )
    ets.fit(df=train_df)
    p = ets.predict(pred_df)
    print(p)
    p = pd.DataFrame(p)

    p2 = p.copy()
    p2 = pd.DataFrame(p2)
    p2.index = range(2001,2015) ## error  usually  means range is  too big

    #p2.loc[:,['prediction_5','prediction','prediction_95']] =  np.exp(p2.loc[:,['prediction_5','prediction','prediction_95']].cumsum())

    print(p2)

    plt.plot(df.index,df.close,color='blue')
    plt.plot(p2.index,p2.prediction,color='red')
    plt.fill_between(p2.index,p2.prediction_5,p2.prediction_95)
    plt.show()



    ppface = plot_predicted_data(train_df2,p2,'time',actual_col='preds',pred_col='prediction',title='wiHELLO  WORLDLSDL')
    #_ = plot_posterior_params(ets, kind='pair', pair_type='reg',
                       #   incl_trend_params=False, incl_smooth_params=False)



if __name__ =='__main__':
    main()


