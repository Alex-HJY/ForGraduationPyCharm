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
import easygui
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
x=[1,2,3,4,5,6,7,8]
y1=[2,3,3,2,3,2,2,1]
y2=[2,3,3,2,4,2,2,1]
plt.scatter(x,y1,c="r",marker="x",label=r'神经网络判断故障类型',zorder=1)
plt.plot(x,y2,marker='.',label=r'实际故障类型',zorder=0)
plt.legend(loc='upper left',frameon=False)
plt.show()
