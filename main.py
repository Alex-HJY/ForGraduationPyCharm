# coding=utf-8
import numpy as np
import math
import pymysql as mysql
import matplotlib as plb
import pandas as pd
import sklearn.cluster as skc
from sklearn import datasets

def List_Where():
    db = mysql.connect(host="localhost", user="root", password="123456789", db="test", port=3306, charset='utf8')
    cur = db.cursor()
    fromwhere = []
    sql = "select * from dataset"
    cur.execute(sql)
    result = cur.fetchall()
    for row in result:
        s = row[2] + row[3] + row[4] + row[5]
        fromwhere.append(s)
    fromwhere = list(set(fromwhere))
    fromwhere.sort()
    cur.close()
    db.close()
    return fromwhere


Where=List_Where()
