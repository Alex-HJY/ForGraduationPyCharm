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
import sqlalchemy
import Apriori
from scipy.interpolate import lagrange


# 省检修分公司南京分部东善桥变重庆ABB变压器有限公司1号主变B相

def read_mysql(sql='select * from test.total where transformer like \'省检修分公司南京分部东善桥变重庆ABB变压器有限公司2号主变A\''):  # 读取数据
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

print(df.describe())
df.to_excel('C:/Users/Alex/Desktop/TETS1.xls')


name = ['ch4', 'c2h6', 'c2h4', 'c2h2', 'h2', 'co', 'co2', 'water']

j=10;
x = [i for i in range(0,j)]
ans = []
s = []
for i in range(0, 8):
    s.append( list(df[name[i]]))
    ans.append([])
    for row in range(0, len(df[name[i]]) - j+1):
        if row % j ==0:
            y = s[i][row:row + j]
            z1 = np.polyfit(x, y, 1)
            p1 = np.poly1d(z1)
            ans[i].append(z1[0])
            print(z1[0])
            print(p1)
    ans[i]=pd.Series(ans[i])
    if (ans[i].max() - ans[i].min())!=0:
        ans[i]=2*(ans[i] - ans[i].min()) / (ans[i].max() - ans[i].min())-1
    for k in range(0,ans[i].__len__()):
        ans[i][k]=name[i]+' '+str((ans[i][k]+1)//0.4)

df2 = pd.DataFrame({'ch4': ans[0],
                    'c2h6': ans[1],
                    'c2h4': ans[2],
                    # 'c2h2': ans[3],
                    'h2': ans[4],
                    'co': ans[5],
                    'co2': ans[6],
                    'water': ans[7]})


df2.to_excel('C:/Users/Alex/Desktop/TETS2.xls')

train_data = np.array(df2)
train_x_list=train_data.tolist()



L, suppData = Apriori.apriori(train_x_list, minSupport=0.1 )
rules=Apriori.generateRules(L,suppData,minConf=0.5)
print ("频繁项集L：", L)
print ("所有候选项集的支持度信息：", suppData)
print ("关联规则：", rules)


rules.sort()
pd.Series(rules).to_excel('C:/Users/Alex/Desktop/rules.xls')
pd.Series(suppData).to_excel('C:/Users/Alex/Desktop/suppData.xls')
