# Ex.No: 6               HOLT WINTERS METHOD
### Date: 08-09-2026


### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# Read CSV file
df = pd.read_csv(r"C:\Users\admin\Downloads\bmw (1).csv")

# Display first five rows
print("FIRST FIVE ROWS:")
print(df.head())

# Select price column
data = df["price"].dropna().reset_index(drop=True)

# Plot original data
plt.figure(figsize=(12, 5))
plt.plot(data)
plt.title("BMW Car Price Data")
plt.xlabel("Observation")
plt.ylabel("Price")
plt.show()

# Decompose the time series
decomposition = seasonal_decompose(
    data,
    model="additive",
    period=12
)

# Plot decomposition
decomposition.plot()
plt.suptitle("HOLT-WINTERS TIME SERIES DECOMPOSITION")
plt.tight_layout()
plt.show()

# Split data into training and testing
train_size = int(len(data) * 0.8)

train = data[:train_size]
test = data[train_size:]

# Holt-Winters model
model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=12
)

fit = model.fit()

# Test prediction
test_prediction = fit.forecast(len(test))

# Calculate RMSE
rmse = np.sqrt(
    np.mean((test.values - test_prediction.values) ** 2)
)

print("\nTEST_PREDICTION")
print(test_prediction)

print("\nRMSE:")
print(rmse)

# Fit model on complete dataset
final_model = ExponentialSmoothing(
    data,
    trend="add",
    seasonal="add",
    seasonal_periods=12
)

final_fit = final_model.fit()

# Future prediction
future_steps = 12
future_prediction = final_fit.forecast(future_steps)

print("\nFINAL_PREDICTION")
print(future_prediction)

# Plot test prediction
plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Training Data")
plt.plot(test.index, test, label="Actual Test Data")
plt.plot(test.index, test_prediction, label="Test Prediction")

plt.title("Holt-Winters Test Prediction")
plt.xlabel("Observation")
plt.ylabel("Price")
plt.legend()
plt.show()

# Plot final prediction
plt.figure(figsize=(12, 5))
plt.plot(data.index, data, label="Original Data")

future_index = range(len(data), len(data) + future_steps)

plt.plot(
    future_index,
    future_prediction,
    label="Future Prediction"
)

plt.title("Holt-Winters Final Prediction")
plt.xlabel("Observation")
plt.ylabel("Price")
plt.legend()
plt.show()
```
### OUTPUT:

<img width="919" height="583" alt="image" src="https://github.com/user-attachments/assets/af56f418-571c-4422-9ae9-c7e76c74e550" />

<img width="805" height="871" alt="image" src="https://github.com/user-attachments/assets/603eb7fe-fe07-445a-a87d-4e3c44fd9eac" />


TEST_PREDICTION

<img width="855" height="360" alt="image" src="https://github.com/user-attachments/assets/ac061ea7-5f19-4079-98b0-b6a9cdd1216a" />


FINAL_PREDICTION

<img width="848" height="376" alt="image" src="https://github.com/user-attachments/assets/6967ea47-999c-468a-a14e-ab20897c6d53" />



### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
