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


# 省检修分公司南京分部东善桥变重庆ABB变压器有限公司1号主变B相

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


df = read_mysql()
# df.to_excel('C:/Users/Alex/Desktop/TETS.xls')
print(df.iloc[0])
for i in df.columns:
    for j in range(len(df)):
        if (df[i].isnull())[j]:
            df[i][j] = meaninterp_column(df[i], j)


df.to_excel('C:/Users/Alex/Desktop/TETS1.xls')


name = ['ch4', 'c2h6', 'c2h4', 'c2h2', 'h2', 'co', 'co2', 'water']

j=10;
x = [i for i in range(0,j)]
ans = []
s = []
for i in range(0, 8):
    ans.append( list(df[name[i]]))
    mini=min(ans[i])
    maxi=max(ans[i])
    for k in range(0,ans[i].__len__()):
        if maxi!=mini :
            ans[i][k]=(ans[i][k]-mini)/(maxi-mini)




X=[]
for row in range(0,ans[0].__len__()):
    s=[ans[6][row],ans[5][row]]
    X.append(s)

print(X)

X=np.array(X)
y_pred = DBSCAN(eps = 0.1,min_samples=5).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

df2 = pd.DataFrame({'ch4': ans[0],
                    'c2h6': ans[1],
                    'c2h4': ans[2],
                    # 'c2h2': ans[3],
                    'h2': ans[4],
                    'co': ans[5],
                    'co2': ans[6],
                    'water': ans[7]})


df2.to_excel('C:/Users/Alex/Desktop/TETS2.xls')



