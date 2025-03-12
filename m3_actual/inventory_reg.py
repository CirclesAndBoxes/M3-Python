import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df_indexed = pd.read_csv("inventory.csv")

# Remove rows with missing values for separate regressions
df_shopping = df_indexed.dropna(subset=["shopping_center_sf"])
df_office = df_indexed.dropna(subset=["office_sf_inventory"])

# Prepare data for Shopping Center SF regression
X_shopping = df_shopping[["index"]]
y_shopping = df_shopping["shopping_center_sf"]

# Prepare data for Office SF Inventory regression
X_office = df_office[["index"]]
y_office = df_office["office_sf_inventory"]

# Fit Linear Regression Models
model_shopping = LinearRegression()
model_shopping.fit(X_shopping, y_shopping)

model_office = LinearRegression()
model_office.fit(X_office, y_office)

# Get equations and R² values
slope_shopping = model_shopping.coef_[0]
intercept_shopping = model_shopping.intercept_
r_squared_shopping = model_shopping.score(X_shopping, y_shopping)

slope_office = model_office.coef_[0]
intercept_office = model_office.intercept_
r_squared_office = model_office.score(X_office, y_office)

# Plot results
plt.figure(figsize=(10, 5))
plt.scatter(df_shopping["index"], df_shopping["shopping_center_sf"], color='blue', label="Actual Shopping Center SF")
plt.plot(df_shopping["index"], model_shopping.predict(X_shopping), color='red', linestyle='--', label="Shopping Center SF Regression")
plt.xlabel("Index")
plt.ylabel("Shopping Center SF")
plt.title("Linear Regression for Shopping Center SF")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(df_office["index"], df_office["office_sf_inventory"], color='green', label="Actual Office SF Inventory")
plt.plot(df_office["index"], model_office.predict(X_office), color='orange', linestyle='--', label="Office SF Inventory Regression")
plt.xlabel("Index")
plt.ylabel("Office SF Inventory")
plt.title("Linear Regression for Office SF Inventory")
plt.legend()
plt.show()

(slope_shopping, intercept_shopping, r_squared_shopping), (slope_office, intercept_office, r_squared_office)
