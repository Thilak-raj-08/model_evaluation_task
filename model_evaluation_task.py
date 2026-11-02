import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("students.csv")

print(df.head())
print("Dataset shape:", df.shape)
print("Missing values:")
print(df.isna().sum())

X = df[["Study_Hours", "Attendance", "Assignment_Score", "Previous_Score"]]
y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(comparison.head().round(2))


mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

metrics = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Value": [mae, mse, rmse, r2]
})

print(metrics.round(4))

plt.figure(figsize=(7, 5))
plt.scatter(y_test, predictions, alpha=0.7)
plt.plot([0, 100], [0, 100], color="red", label="Actual = Predicted")
plt.title("Actual vs Predicted Exam Scores")
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.legend()
plt.tight_layout()
plt.show()


comparison["Residual"] = comparison["Actual"] - comparison["Predicted"]

plt.figure(figsize=(7, 5))
plt.scatter(comparison["Predicted"], comparison["Residual"], alpha=0.7)
plt.axhline(0, color="red", label="Zero error")
plt.title("Residuals vs Predicted Scores")
plt.xlabel("Predicted Score")
plt.ylabel("Residual")
plt.legend()
plt.tight_layout()
plt.show()

q1 = comparison["Residual"].quantile(0.25)
q3 = comparison["Residual"].quantile(0.75)
iqr = q3 - q1

lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

outlier_mask = (
    (comparison["Residual"] < lower_limit)
    | (comparison["Residual"] > upper_limit)
)

print("Lower limit:", round(lower_limit, 2))
print("Upper limit:", round(upper_limit, 2))
print("Possible outlier count:", outlier_mask.sum())
print(comparison[outlier_mask].round(2))


simple_model = LinearRegression()
simple_model.fit(X_train[["Study_Hours"]], y_train)

simple_predictions = simple_model.predict(X_test[["Study_Hours"]])

simple_mae = mean_absolute_error(y_test, simple_predictions)
simple_mse = mean_squared_error(y_test, simple_predictions)
simple_rmse = simple_mse ** 0.5
simple_r2 = r2_score(y_test, simple_predictions)

model_comparison = pd.DataFrame({
    "Model": ["Four features", "Study Hours only"],
    "MAE": [mae, simple_mae],
    "MSE": [mse, simple_mse],
    "RMSE": [rmse, simple_rmse],
    "R2": [r2, simple_r2]
})

print(model_comparison.round(4))