# coding=utf-8
import copy

import numpy as np

import pymysql

import pandas as pd

import matplotlib.pyplot as plt
import sqlalchemy

name = ['ch4', 'c2h6', 'c2h4', 'c2h2', 'h2', 'co', 'co2', 'water']


# 获取数据
def read_mysql(sql='select * from test.total where transformer like \'省检修分公司苏州分部车坊变特工变电沈阳变压器集团有限公司3号主变C相\' '):  # 读取数据
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
        df = pd.read_sql(sql + 'ORDER BY time', con=conn)
    except pymysql.err.ProgrammingError as e:
        print('Error is ' + str(e))
    # print(df.head())
    conn.close()
    print(df.head())
    print('ok')
    return df



# 平均值插值aa
def meaninterp_column(s, n, k=5):
    y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]
    y = y[y.notnull()]
    return sum(y) / len(y)

#DATAFRAME转LIST
def dftolist(df):
    train_data = np.array(df)
    train_x_list = train_data.tolist()
    return train_x_list


#画图
def printgraph(df=[], wrongdata=[], lieshu=[3],df_fix=[]):
    for row in lieshu:

        x = []
        y = []
        xfix = []
        yfix = []
        xx1 = []
        yy1 = []
        xx0 = []
        yy0 = []
        for i in range(df.__len__()):
            x.append(df[i][2])
            y.append(df[i][row])
            xfix.append(df_fix[i][2])
            yfix.append(df_fix[i][row])
        for i in wrongdata[row - 3]:
            if i[3] == 1:
                xx1.append(i[1])
                yy1.append(i[2])
            else:
                xx0.append(i[1])
                yy0.append(i[2])
        plt.scatter(xx1, yy1, c='r', marker='x', zorder=2)
        plt.scatter(xx0, yy0, c='y', marker='>', zorder=2)
        plt.title(name[row - 3], fontsize=12)
        plt.plot(x, y,zorder=1)
        # plt.plot(xfix, yfix, c='k', zorder=0)
        plt.xlim(735954.1918969348,736154.1918969348)
        plt.show()
    return

#计算异常值
def getwrong(d, w=50, w2=10, beishu=5):
    wrongdata = []

    xxx = [ii for ii in range(0, w)]
    for j in range(3, 11):
        temp = []
        i=w+1
        while (i<len(d)):
            t1 = []
            t2 = []
            avgz = 0
            for k in range(i - w, i):
                t1.append(d[k][j])
            avg1 = sum(t1) / t1.__len__()
            for k in range(t1.__len__()):
                # avgz=avgz+(t1[k]-avg1)*(t1[k]-avg1)
                avgz = avgz + abs((t1[k] - avg1))
            avgz = avgz / t1.__len__()
            avgmax = avg1 + beishu * avgz
            avgmin = avg1 - beishu * avgz

            if d[i][j] < avgmin or d[i][j] > avgmax:
                print(avgmin, ' ', avgmax, ' ', d[i][j], avgz, avg1)
                z1 = np.polyfit(xxx, t1, 1)
                p1 = np.poly1d(z1)
                kk = 0
                # kk=1 传感器 0 变压器异常
                for t in range(i, min(i + w2, len(d))):
                    xt = w + t - i
                    # yt = xt * z1[0] + z1[1]
                    yt=avg1
                    if d[t][j] < yt + beishu * avgz and d[t][j] > yt - beishu * avgz:
                        kk = 1
                for t in range(i, min(i + w2, len(d))):
                    xt = w + t - i
                    # yt = xt * z1[0] + z1[1]
                    yt = avg1
                    if not (d[t][j] < yt + beishu * avgz and d[t][j] > yt - beishu * avgz):
                        temp.append([t, d[t][2], d[t][j], kk])

                    if kk:
                        d[t][j]=yt
                i=min(i + w2, len(d))
            else:
                i=i+1
        wrongdata.append(temp)
    return wrongdata,d



# 读取数据
df = read_mysql()

# 平均值插值
for i in df.columns:
    for j in range(len(df)):
        if (df[i].isnull())[j]:
            df[i][j] = meaninterp_column(df[i], j)
# 转LIST
df = dftolist(df)


# 获取异常数据
daa=copy.deepcopy(df)
wrongdata,df_fix = getwrong(daa)
#画图


printgraph(df, wrongdata, [3, 4, 5, 6, 7, 8, 9],df_fix)

