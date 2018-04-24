# coding=utf-8
import numpy as np
import math
import pymysql
import matplotlib as plb
import pandas as pd
import statsmodels
import datetime as dt
import sklearn
import scipy
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import sqlalchemy
import Apriori
from scipy.interpolate import lagrange


def read_data(filepath='C:/Users/Alex/Desktop/故障数据.csv'):
    df=pd.read_csv('C:/Users/Alex/Desktop/故障数据.csv',engine='python')
    return df

def data_trans(df):
    tt=[]
    k=-1
    for i in df.columns:
        temp=[]
        for j in range(1,len(df)):
            t=df[i][j]-df[i][j-1]
            if t<0:
                temp.append(i+'-')
            elif t==0:
                temp.append(i + '~')
            else:
                temp.append(i + '+')
        tt.append(temp)
    df2 = pd.DataFrame({'ch4': tt[0],
                        'c2h6': tt[1],
                        'c2h4': tt[2],
                        'c2h2': tt[3],
                        'h2': tt[4],
                        'co': tt[5],
                        'co2': tt[6],
                        })
    return df2

df=read_data()
df=data_trans(df)
df.to_excel('C:/Users/Alex/Desktop/故障数据关联规则.xls')
print(df.describe())



train_data = np.array(df)
train_x_list=train_data.tolist()
L, suppData = Apriori.apriori(train_x_list, minSupport=0.1 )
rules=Apriori.generateRules(L,suppData,minConf=0.7)
print ("频繁项集L：", L)
print ("所有候选项集的支持度信息：", suppData)
print ("关联规则：", rules)

rules.sort()
pd.Series(rules).to_excel('C:/Users/Alex/Desktop/rules-guzhang.xls')
pd.Series(suppData).to_excel('C:/Users/Alex/Desktop/suppData-guzhang.xls')

