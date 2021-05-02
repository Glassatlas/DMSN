# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:44:41 2021

@author: putta
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import model_selection
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# GRAPHING

data = pd.read_csv('data200.csv', sep=',')
data[:10]

X = data['length_title']
Y = data['avg_score']
Size = data['num_posts']/200000
plt.scatter(x = X, y = Y, s = Size )
plt.title('Length of Post Title vs Average Score of Post')
plt.xlabel('Length of Post Title (# of Characters)')
plt.ylabel('Average Score of Post')
plt.show()


X = data['length_title']
Y = data['avg_comments']
Size = data['num_posts']/200000
plot = plt.scatter(x = X, y = Y, s = Size)
plt.title('Avg Number of Comments vs Length of Post title')
plt.xlabel('Length of Post Title (# of Characters)')
plt.ylabel('Average Comments')
plt.show()
'''
data2 = pd.read_csv('1820domain.csv', sep=',')
data2[:10]

X2 = data2['domain_link']
Y2 = data2['avg_score']
Size2 = data2['num_post']/20000
plt.scatter(x = X2, y = Y2, s = Size2 )
plt.title('Domain name vs Average Score of Post')
plt.xlabel('Domain name')
plt.ylabel('Average Score of Post')
plt.show()

#plot


X = data['length_title']
Y = data['avg_comments']
Size = data['num_posts']/200000
plot = plt.scatter(x = X, y = Y, s = Size)
plt.title('Avg Number of Comments vs Length of Post title')
plt.xlabel('Length of Post Title (# of Characters)')
plt.ylabel('Average Comments')
plt.show()
'''
#Linear Regression (Length of Post)

Y = data['avg_score']
X = data['length_title']

linear_regression = LinearRegression()
reshapedX = X.values.reshape(-1, 1)
linear_regression.fit(reshapedX, Y)
model = linear_regression.predict(reshapedX)

plt.figure(figsize=(10,8));
plt.scatter(X, Y);
plt.plot(X, model);
plt.title('Length of Post Title vs Average Score of Post (Filtered)')
plt.xlabel('Length of Post Title (# of Characters)')
plt.ylabel('Average Score of Post')
plt.show()

# Polynomial Regression (Length of Post)

poly_reg = PolynomialFeatures(degree=4)
reshapedX = X.values.reshape(-1, 1)
poly = poly_reg.fit_transform(reshapedX)


linear_regression2 = LinearRegression()
reshapedY = Y.values.reshape(-1, 1)
linear_regression2.fit(poly, reshapedY)
y_pred = linear_regression2.predict(poly)
plt.figure(figsize=(10,8));
plt.scatter(X, Y);
plt.plot(X, y_pred);
plt.title('Length of Post Title vs Average Score of Post (Filtered)')
plt.xlabel('Length of Post Title (# of Characters)')
plt.ylabel('Average Score of Post')
plt.show()

# Random Forest 

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(max_depth = 4)
reshapedX = X.values.reshape(-1, 1)
forest.fit(reshapedX, Y)
forest_mod = forest.predict(reshapedX)

plt.figure(figsize=(10,8));
plt.scatter(X, Y);
plt.plot(X, forest_mod);
plt.title('Length of Post Title vs Average Score of Post (Filtered)')
plt.xlabel('Length of Post Title (# of Characters)')
plt.ylabel('Average Score of Post')
plt.show()

#


# Converts the numbers 0-23 to their respective times
def convertHourToTime(num):
    if num == 0:
        timeOfDay = '12 AM'
    elif num <= 11:
        timeOfDay = str(num) + ' AM'
    elif num == 12:
        timeOfDay = '12 PM'
    else: 
        timeOfDay = str(num-12) + ' PM'
    return timeOfDay

# Converts the numbers 1-7 to their respective weekdays
def convertNumToDay(num):
    weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    return weekdays[num-1]

timeScore = pd.read_csv('timescore.csv', sep=',')
formattedTimeScore = timeScore.copy()
formattedTimeScore['hourofday'] = formattedTimeScore['hourofday'].apply(convertHourToTime)
formattedTimeScore['dayofweek'] = formattedTimeScore['dayofweek'].apply(convertNumToDay)


timeScoreMatrix = timeScore.pivot(index='dayofweek', columns='hourofday', values='avg_score')
cols = timeScoreMatrix.columns.tolist()
cols.insert(0, cols.pop(cols.index(23)))
fixedTimeScoreMatrix = timeScoreMatrix.reindex(columns=cols)
timeScoreMatrix


# heat map for time

import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl

fig = plt.figure()
fig, ax = plt.subplots(1,1,figsize=(15,15))
heatmap = ax.imshow(fixedTimeScoreMatrix, cmap='BuPu')
ax.set_xticklabels(np.append('', formattedTimeScore.hourofday.unique())) # columns
ax.set_yticklabels(np.append('', formattedTimeScore.dayofweek.unique())) # index

tick_spacing = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.set_title("Time of Day Posted vs Average Score")
ax.set_xlabel('Time of Day Posted (EST)')
ax.set_ylabel('Day of Week Posted')

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", "3%", pad="1%")
fig.colorbar(heatmap, cax=cax)
plt.show()