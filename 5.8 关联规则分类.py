# coding=utf-8
import copy

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

name = ['CH4', 'C2H6', 'C2H4', 'C2H2', 'H2', '故障类型']

weights = {'CH4': 0.2, 'C2H6': 0.2, 'C2H4': 0.2, 'C2H2': 0.2, 'H2': 0.2}

problems=['低温过热','高温过热','低能放电','高能放电']

# 数据格式
# 'ch4', 'c2h6', 'c2h4', 'c2h2', 'h2', 'problem'
# 0.0-0.1 : A
# 0.1-0.2 : B
# 0.2-0.3 : C
# 0.3-0.4 : D
# 0.4-0.1 : E


def read_training_data():
    df = pd.read_excel('C:/Users/Alex/Desktop/input58.xls')
    train_data = np.array(df).tolist()
    return train_data


def read_test_data():
    df = pd.read_excel('C:/Users/Alex/Desktop/testinput58.xls')
    train_data = np.array(df).tolist()
    return train_data


def judge(a):
    if a >= 0 and a < 0.1:
        return 'A'
    elif a >= 0.1 and a < 0.2:
        return 'B'
    elif a >= 0.2 and a < 0.3:
        return 'C'
    elif a >= 0.3 and a < 0.4:
        return 'D'
    elif a >= 0.4:
        return 'E'


def initial_data(df=[]):
    for row in df:
        for i in range(5):
            row[i] = name[i] + judge(row[i])
    return df


def data_analyse(df):
    L, suppData = Apriori.apriori(df, minSupport=0)
    rules = Apriori.generateRules(L, suppData, minConf=0)
    # print("频繁项集L：", L)
    # print("所有候选项集的支持度信息：", suppData)
    # print("关联规则：", rules)
    rules.sort()
    pd.Series(rules).to_excel('C:/Users/Alex/Desktop/rules-guzhang.xls')
    pd.Series(suppData).to_excel('C:/Users/Alex/Desktop/suppData-guzhang.xls')
    return rules


def transfer_to_rules(rules=[]):
    a = list((filter(lambda x: len(x[1]) == 1 and len(x[0]) == 1, rules)))
    rules={}
    for row in a:
        str1 = (list(row[0])[0])
        str2 = (list(row[1])[0])
        a = (row[2])
        rules[str1+str2]=a
    return rules


def data_calculate(test_data=[], rules={}):
    accuracy = 0
    ans=[]
    #算分判断错误类型
    for row in test_data:
        maxscore=0
        row_problem=''
        for problem in problems:
            temp=0
            for i in range(len(row)-1):
                if rules.get(problem+row[i]):
                    t=rules.get(problem+row[i])
                else:
                    t=0
                temp+=t*weights.get(name[i])
            if maxscore<=temp:
                maxscore=temp
                row_problem=problem
        ans.append(row_problem)
    print('test:',ans)
    print('answer:',[x[5] for x in test_data ])

    #算正确率
    for i in range(test_data.__len__()):
        if ans[i]==[x[5] for x in test_data ][i]:
            accuracy+=1
    accuracy/=test_data.__len__()

    print("accuracy:",str(accuracy*100)+'%')
    return str(accuracy*100)+'%'


training_data = read_training_data()
training_data = initial_data(training_data)
rules = data_analyse(training_data)
rules = transfer_to_rules(rules)
test_data=read_test_data()
test_data=initial_data(test_data)
accuracy=data_calculate(test_data,rules)
