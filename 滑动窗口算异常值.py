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

df = read_data()
df = dftolist(df)
wrongdata = getwrong(df)
wrongdata_judged=judgewrong(df,wrongdata)