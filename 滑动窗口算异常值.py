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


def getwrong(df):
    ans = []
    return ans

def judgewrong(df,wrongdata):
    ans = []
    return ans

df = read_mysql()
print(df.describe())
for i in df.columns:
    for j in range(len(df)):
        if (df[i].isnull())[j]:
            df[i][j] = meaninterp_column(df[i], j)

df = dftolist(df)
wrongdata = getwrong(df)
wrongdata_judged=judgewrong(df,wrongdata)