import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import clean_data as cd


# Read and review csv file.
nyc = pd.read_csv("data.csv", sep=';')

print(nyc.head())
print(nyc.tail())


# Clean data. Review the data.
nyc_dict = {}
for string in nyc.values:
    data_list = string[0].split(",")
    if cd.y_verify(data_list[0]):
        year = data_list[0]
        year = int(year[:4])
        temperature = float(data_list[1])
        if temperature > 0:
            nyc_dict.update({year: temperature})

nyc_clear = pd.DataFrame(nyc_dict.items(), columns=["Year", "Temperature"])

print(nyc_clear.head(), nyc_clear.tail(), sep="\n")


# Split cleaned data to learn and check model.
learning_nyc_dict = {i: e for i, e in nyc_dict.items() if i < 2019}
learning_nyc = pd.DataFrame(learning_nyc_dict.items(), columns=["Year", "Temperature"])

for_check_nyc_dict = {i: e for i, e in nyc_dict.items() if i > 2018}
for_check_nyc = pd.DataFrame(for_check_nyc_dict.items(), columns=["Year", "Temperature"])


# Create the model.
x = np.array(learning_nyc.Year).reshape((-1, 1))
y = np.array(learning_nyc.Temperature)

model = LinearRegression()
model.fit(x, y)

# These formulas below are to calculate slope, intercept of linear regression without using appropriate numpy methods.
# x = np.array(learning_nyc.Year)
# y = np.array(learning_nyc.Temperature)
# n = np.size(x)
# x_mean = np.mean(x)
# y_mean = np.mean(y)
# xy = (np.sum(x * y)) - (n * x_mean * y_mean)
# xx = (np.sum(x * x)) - (n * x_mean * x_mean)
# slope = xy / xx
# intercept = y_mean - slope * x_mean


# Calculate prediction using Numpy build-in methods.
years_pred = [2019, 2020, 2021, 2022]

x_pred = np.array(years_pred).reshape(-1, 1)
y_pred = model.predict(x_pred)


# Perform diagram for learning and predicted.
sns.set_style("whitegrid")
# axes = sns.regplot(x=nyc_clear.Year, y=nyc_clear.Temperature)
# axes.set_ylim(10, 70)
axes_for_check = sns.regplot(x=for_check_nyc.Year, y=for_check_nyc.Temperature, label="Checking")
axes_pred = sns.regplot(x=x_pred, y=y_pred, label="Predicted")
plt.legend(fontsize=10)


# Calculate predicted temperature in 1890.
print(f"The temperature in 1890 was {model.coef_ * 1890 + model.intercept_}")

plt.show()

