import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def clean(df, drop_list):
    df_ = df.drop(drop_list, axis=1)
    cities = list(df_['city'].unique())
    for index, i in enumerate(cities):
        df_['city'].where(df_['city'] != i, index + 1, inplace=True)
    phones = df_['phone'].unique()
    for index, i in enumerate(phones):
        df_['phone'].where(df_['phone'] != i, index + 1, inplace=True)
    df_['signup_date'] = pd.to_datetime(df_['signup_date'], infer_datetime_format=True)
    df_['last_trip_date'] = pd.to_datetime(df_['last_trip_date'], infer_datetime_format=True)
    date = pd.to_datetime('20140601', infer_datetime_format=True)
    df_['churn'] = df_['last_trip_date'] >= date
    df_ = df_.drop(['signup_date','last_trip_date'], axis=1)
    df_ = df_.fillna(0)
    return df_