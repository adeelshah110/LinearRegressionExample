from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("c:/Users/HIMAT/Desktop/machinelearnign/datasets/price.csv")
import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)

#%matplotlib inline

plt.xlabel('area(sq ft)')
plt.ylabel("price(us$)")
plt.scatter(df.area, df.price, color ="blue", marker ="+")
# Import necessary libraries
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("c:/Users/HIMAT/Desktop/machinelearnign/datasets/price.csv")

# Create a LinearRegression model
reg = linear_model.LinearRegression()

# Fit the model
reg.fit(df[["area"]], df["price"])

# Make a prediction for a single data point (3300)
prediction = reg.predict([[3300]])

# Print the prediction
print(prediction)

########################################################################
#Second Example
#######################################################################
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("c:/Users/HIMAT/Desktop/machinelearnign/datasets/age.csv")
df
reg = linear_model.LinearRegression()
reg.fit(df[["age"]], df["weight"])
pr = reg.predict([[33]])
print (pr)