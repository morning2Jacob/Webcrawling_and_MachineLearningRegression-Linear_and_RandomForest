# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:14:33 2023

@author: Jacob
"""

import pandas as pd
import numpy as np
import requests as req
import json
import time
import random
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import re

def get_page_count(url, header):
    try:
        resp = req.get(url, headers=header)
        resp.raise_for_status()
        data=json.loads(resp.text)
        page_count=data['pa']['totalPageCount']
    except Exception as err:
        print(err)
        page_count=0
    return page_count

def web_one_page(url_p, header, ret_data):
    try:
        response = req.get(url_p, headers=header)
        response.raise_for_status()
        info = json.loads(response.text)
    except Exception as err:
        print(err)
    for a in range(len(info['webRentCaseGroupingList'])):
        ret_data.append(info['webRentCaseGroupingList'][a])
    return ret_data

#---------------------------------------------------------------------------------------------
#Web Crawling
url = 'https://rent.houseprice.tw/ws/list/%E5%8F%B0%E4%B8%AD%E5%B8%82_city/%E8%A5%BF%E5%B1%AF%E5%8D%80_zip/'
header={
  'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36 LBBROWSER'
  }
ret_data = []
for i in range(get_page_count(url, header)):
    url_p = 'https://rent.houseprice.tw/ws/list/%E5%8F%B0%E4%B8%AD%E5%B8%82_city/%E8%A5%BF%E5%B1%AF%E5%8D%80_zip/?p={}'.format(i+1)
    web_one_page(url_p, header, ret_data)
    
df = pd.DataFrame(ret_data)
df.columns.unique()
dataset = df.iloc[:, [2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 22, 23, 27, 37]]
dataset.to_csv('D0790192_rent.csv', index=False)

#Import data ------------------------------------------------------------------

df = pd.read_csv('C:\\Users\\Jacob Kai\\D0790192_rent.csv')

df1 = df[['rm', 'livingRm', 'bathRm', 'toFloor', 'buildPin', 'managementFee', 'rentPurPoseName', 'rentPrice']]
df1.rename(columns={'rm':'???',
                    'livingRm':'???',
                    'bathRm':'???',
                    'toFloor':'??????',
                    'buildPin':'??????',
                    'managementFee':'?????????_str',
                    'rentPurPoseName':'????????????',
                    'rentPrice':'??????'}, inplace=True)

#Data cleaning ----------------------------------------------------------------
#Drop duplicates and nan, fill 0 into nan -------------------------------------

df1.drop_duplicates()
df2 = df1.copy()
df2['???'] = df2['???'].fillna(0)
df2['???'] = df2['???'].fillna(0)
df2['???'] = df2['???'].fillna(0)

df2['?????????_str'] = df2['?????????_str'].fillna('0')

lst1 = []
for i in df2['?????????_str']:
    b = re.sub('\D', '', i)
    lst1.append(b)
df2['?????????'] = lst1
df2['?????????'] = df2['?????????'].replace('', '0')
df2['?????????'] = df2['?????????'].astype(int)

lst2 = []
for i in df2['??????']:
    c = re.sub('\D', '', i)
    lst2.append(c)
df2['??????'] = lst2
df2['??????'] = df2['??????'].replace('', '0')
df2['??????'] = df2['??????'].astype(int)

df2.drop(columns=['?????????_str'], inplace=True)

#Select Object Type -----------------------------------------------------------

df2 = df2[df2['????????????'] == '??????']

#Split the data into train data and test data ---------------------------------

X = df2[['??????', '??????', '?????????']]
y = df2['??????']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

# define the outliers and remove outliers -------------------------------------

def define_outliers(data_column):
    p75, p25 = np.percentile(data_column, [75, 25])
    upper = p75 + 1.5*(p75 - p25)
    bottom = p25 - 1.5*(p75 - p25)
    return upper, bottom

lst = ['Floor', 'Pin', 'ManFee']
d={}
d1={}
d2={}
d3={}
d4={}

for (i, u) in zip(X_train.columns, lst):
    d1['upper_X_train_{}'.format(u)] = define_outliers(X_train[i])[0]
    d2['bottom_X_train_{}'.format(u)] = define_outliers(X_train[i])[1]

for (t, r) in zip(X_test.columns, lst):
    d3['upper_X_test_{}'.format(r)] = define_outliers(X_test[t])[0]
    d4['bottom_X_test_{}'.format(r)] = define_outliers(X_test[t])[1]
    
d1['upper_y_train'] = define_outliers(y_train)[0]
d2['bottom_y_train'] = define_outliers(y_train)[1]
d3['upper_y_test'] = define_outliers(y_test)[0]
d4['bottom_y_test'] = define_outliers(y_test)[1]

dt1 = pd.Series(d1).to_frame(name='Upper_X_train')
dt2 = pd.Series(d2).to_frame(name='Bottom_X_train')
dt3 = pd.Series(d3).to_frame(name='Upper_X_test')
dt4 = pd.Series(d4).to_frame(name='Bottom_X_test')

Train_data = X_train.join(y_train)
for s in X_train.columns:
    Train_data = Train_data[Train_data[s] >= 0 ]

for (f, j) in zip(Train_data.columns, dt1['Upper_X_train']):
    if (f == '??????'):
        continue
    Train_data = Train_data[(Train_data[f] <= j)]

for (f, j) in zip(Train_data.columns, dt2['Bottom_X_train']):
    if (f == '??????'):
        continue
    Train_data = Train_data[(Train_data[f] >= j)]
    
#Train_data = Train_data[(Train_data['???'] <= 5)]
    
Test_data = X_test.join(y_test)
for t in X_test.columns:
    Test_data = Test_data[Test_data[t] >= 0 ]
    
for (f, j) in zip(Test_data.columns, dt3['Upper_X_test']):
    if (f == '??????'):
        continue
    Test_data = Test_data[(Test_data[f] <= j)]
    
for (f, j) in zip(Test_data.columns, dt4['Bottom_X_test']):
    if (f == '??????'):
        continue
    Test_data = Test_data[(Test_data[f] >= j)]
    
#Test_data = Test_data[(Test_data['???'] <= 3)]

X_train_fnl = Train_data[['??????', '??????', '?????????']]
y_train_fnl = Train_data['??????']
X_test_fnl = Test_data[['??????', '??????', '?????????']]
y_test_fnl = Test_data['??????']

#Multi-Linear Regression Module & R2 Evaluation for ??????-----------------------

ln_reg = linear_model.LinearRegression()
ln_reg.fit(X_train_fnl, y_train_fnl)
ln_y_pred = ln_reg.predict(X_test_fnl)
r2_Linear = metrics.r2_score(y_test_fnl, ln_y_pred)
m_Linear = ln_reg.coef_
b_Linear = ln_reg.intercept_
print('R2 score of Multi-Linear Regression is: ')
print(r2_Linear)

#Define Multi-Linear Regression Module for ??????--------------------------------

def LinearRegression_Office(x_test):
    reg = linear_model.LinearRegression()
    reg.fit(X_train_fnl, y_train_fnl)
    prediction = reg.predict(x_test)
    return prediction

#Random Forest Regression Module & R2 Evaluation for ??????----------------------

rf_reg = RandomForestRegressor(random_state=2023)
rf_reg.fit(X_train_fnl, y_train_fnl)
rf_y_pred = rf_reg.predict(X_test_fnl)
r2_rf = metrics.r2_score(y_test_fnl, rf_y_pred)
print('R2 score of Random Forest Regression is: ')
print(r2_rf)

#Define Multi-Linear Regression Module for ??????--------------------------------

def RandomForestRegression_Office(x_test):
    reg = RandomForestRegressor()
    reg.fit(X_train_fnl, y_train_fnl)
    prediction = reg.predict(x_test)
    return prediction