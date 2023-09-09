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

                 
d = pd.read_csv("c:/Users/HIMAT/Desktop/machinelearnign/datasets/predict.csv")

p = reg.predict(d)

d["prices"] = p

#d.to_csv("c:/Users/HIMAT/Desktop/machinelearnign/datasets/predicts_prices.csv",index= False)

plt.xlabel('area(sq ft)')
plt.ylabel("price(us$)")
plt.scatter(df.area, df.price, color ="red", marker ="+")
plt.plot(df.area, reg.predict(df[["area"]]), color ="blue")
# Import necessary libraries