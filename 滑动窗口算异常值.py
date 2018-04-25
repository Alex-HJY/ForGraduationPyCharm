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

name = ['ch4', 'c2h6', 'c2h4', 'c2h2', 'h2', 'co', 'co2', 'water']

def read_mysql(sql='select * from test.total where transformer like \'省检修分公司南京分部东善桥变重庆ABB变压器有限公司2号主变A相\''):  # 读取数据
    try:
        conn = pymysql.connect(host="localhost", user="root", password="123456789", db="test", port=3306,
                               charset='utf8')
    except pymysql.err.OperationalError as e:
        print('Error is ' + str(e))
    try:
        engine = sqlalchemy.create_engine('mysql+pymysql://root:123456789@localhost:3306/test')
    except sqlalchemy.exc.OperationalError as e:
        print('Error is ' + str(e))

    except sqlalchemy.exc.InternalError as e:
        print('Error is ' + str(e))
    try:
        df = pd.read_sql(sql, con=conn)
    except pymysql.err.ProgrammingError as e:
        print('Error is ' + str(e))
    # print(df.head())
    conn.close()
    print(df.head())
    print('ok')
    return df


def ployinterp_column(s, n, k=5):
    y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)


def meaninterp_column(s, n, k=5):
    y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]
    y = y[y.notnull()]
    return sum(y) / len(y)



def read_data(filepath='C:/Users/Alex/Desktop/故障数据.csv'):
    df = pd.read_csv('C:/Users/Alex/Desktop/故障数据.csv', engine='python')

    return df


def dftolist(df):
    train_data = np.array(df)
    train_x_list = train_data.tolist()
    return train_x_list

def getwrongbypoly(df=[],w=5,wucha=0.13):
    ans = []
    x = [ii for ii in range(0, w)]
    for j in range(3,11):
        for i in range(w,len(df)):
            t1=[]
            t2=[]
            for k in range(i-w,i):
                t1.append(df[k][j])
            y=t1
            z1 = np.polyfit(x, y, 1)
            p1 = np.poly1d(z1)
            kk=z1[0]*(w)+z1[1]
            tt=df[i][j]-kk
            # print(tt,' ',kk,' ',df[i][j])
            if (abs(tt)/kk)>wucha:
                ans.append(df[i])
    print(ans.__len__())
    return ans

def getwrong(df=[],w=10,wucha=2):
    ans = []
    for j in range(3,11):
        for i in range(w+1,len(df)):
            t1=[]
            t2=[]
            avgz=0
            for k in range(i-w,i):
                t1.append(df[k][j])
            avg1=sum(t1)/t1.__len__()
            for k in range(t1.__len__()):
                avgz=avgz+(t1[k]-avg)*(t1[k]-avg)
            avgz=avgz/t1.__len__()
            avgmax=avg1+3*avgz
            avgmin = avg1 - 3 * avgz
            # print(tt,' ',avg,' ',df[i][j])
            if df[i][j]<avgmin or df [i][j]>avgmax :
                print(avgmin, ' ', avgmax, ' ', df[i][j])
                df[i].append((name[j-3],j,avg1))
                ans.append(df[i])
    print(ans.__len__())
    pd.DataFrame(df).to_excel('C:/Users/Alex/Desktop/T.xls')
    return ans


def getwronglunwen(df=[],w=10,wucha=2):
    ans = []
    for j in range(3,11):
        for i in range(w+1,len(df)):
            t1=[]
            t2=[]
            for k in range(i-w,i):
                t1.append(df[k][j])
                t2.append(df[k][j]-df[k-1][j])
            avg1=sum(t1)/t1.__len__()
            avgz=sum(t2)/t2.__len__()
            avg=avg1+avgz*w/2
            tt = df[i][j] - avg
            # print(tt,' ',avg,' ',df[i][j])
            if avg!=0 and (abs(tt) / avg) > wucha :
                print(tt, ' ', avg, ' ', df[i][j])
                df[i].append((name[j-3],j,avg))
                ans.append(df[i])
    print(ans.__len__())
    pd.DataFrame(df).to_excel('C:/Users/Alex/Desktop/T.xls')
    return ans

def judgewrong(df,wrongdata):
    ans = []
    return ans

df = read_mysql()

for i in df.columns:
    for j in range(len(df)):
        if (df[i].isnull())[j]:
            df[i][j] = meaninterp_column(df[i], j)

df=dftolist(df)

wrongdata = getwrong(df)
wrongdata_judged=judgewrong(df,wrongdata)