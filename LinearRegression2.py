import pandas as pd
import quandl
import math
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
#df = quandl.get("WIKI/GOOGL")
#df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
df = pd.read_csv("c:/Users/HIMAT/Desktop/machinelearnign/datasets/wiki.csv")
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100
# Daily change
df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100
df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

forecast_cal = "Adj. Close"

df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.1 * len(df)))
df["label"] = df[forecast_cal].shift(-forecast_out)
df.dropna(inplace=True)
X = np.array(df.drop(["label"], 1))

forecast_out = int(math.ceil(0.01 *len(df)))
df["lablel"] = df[forecast_cal].shift(-forecast_out)
df.dropna(inplace=True)
X = np.array(df.drop(["label"], 1))
y = np.array(df["label"])
X = preprocessing.scale(X)
y = np.array(df["label"])

X_train, X_test, y_train,y_test =  train_test_split(X, y, test_size=0.2)

reg = LinearRegression()
reg.fit(X_train, y_train)
pr = reg.score(X_test, y_test)
print (pr)
y = np.array(df["label"])
X = preprocessing.scale(X)
y = np.array(df["label"])


X_train, X_test, y_train,y_test =  train_test_split(X, y, test_size=0.2)
reg = LinearRegression()
reg.fit(X_train, y_train)
pr = reg.score(X_test, y_test)
print (pr)


