import pandas as pd
import quandl, math, datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pickle

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
# A single label, e.g. 5 or 'a',
# (note that 5 is interpreted as a label of the index. This use is not an integer position along the index)
# A list or array of labels ['a', 'b', 'c']
# A slice object with labels 'a':'f' (note that contrary to usual python slices,
# both the start and the stop are included, when present in the index! - also see Slicing with labels)

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
# High and low percentage change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
# Close and open percentage change


df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
# There is an optional parameter inplace
# so that the original data can be modified without creating a copy:

forecast_out = int(math.ceil(0.1 * len(df)))
print(forecast_out, "Days in Advance")
# 0.01 is equivalent of 1% of total data in future forecasting

df['label'] = df[forecast_col].shift(-forecast_out)
# Briefly, feature is input; label is output.

# df.dropna(inplace=True)

# print("Head:", df.head)
# print("Tail:", df.tail)

x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x_recent = x[-forecast_out:]
x = x[:-forecast_out]
# df.drop: Return new object with labels in requested axis removed.
# axis = 1: Whether to drop labels from the index (0 / ‘index’) or columns (1 / ‘columns’).

df.dropna(inplace=True)

# y = np.array(df)

y = np.array(df['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
# Train on two independent sets of data to avoid selection bias

clf = LinearRegression(n_jobs=-1)
# clf = svm.SVR()
# SVR = support vector regression
# kernel: Way to make optimization efficient when there are lots of features

clf.fit(x_train, y_train)

with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test, y_test)

forecast_set = clf.predict(x_recent)

print(forecast_set)
print(accuracy)
print(forecast_out)


df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')

plt.show()