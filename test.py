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
import numpy as np


def read_mysql(sql='select * from test.total '):  # 读取数据
    try:
        conn = pymysql.connect(host="localhost", user="root", password="123456789", db="test", port=3306,
                               charset='utf8')
    except pymysql.err.OperationalError as e:
        print('Error is ' + st(e))
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
s=0
df = read_mysql()
for row in df.columns[2:]:
   s=s+df[row].count()
   print(s,' ',row)
print(df.describe(),s)