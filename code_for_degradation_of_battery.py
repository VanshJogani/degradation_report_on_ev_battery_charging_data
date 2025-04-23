import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("ev_battery_charging_data.csv")  # Replace with your actual path

# Drop rows with missing target
df.dropna(subset=["Degradation Rate (%)"], inplace=True)

# Add lag features for time series
df["Degradation Rate_lag1"] = df["Degradation Rate (%)"].shift(1)
df["Degradation Rate_lag2"] = df["Degradation Rate (%)"].shift(2)
df.dropna(inplace=True)

# Features and target
X = df.drop("Degradation Rate (%)", axis=1)
y = df["Degradation Rate (%)"]

# One-hot encode categorical columns (if any)
X = pd.get_dummies(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Models dictionary
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR()
}
time_series_model = LinearRegression()

results = []

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    cv_rmse = -np.mean(cross_val_score(model, X_scaled, y, scoring="neg_root_mean_squared_error", cv=5))

    # Print each metric
    print(f"\n📌 {name}")
    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"R² Score : {r2:.4f}")
    print(f"CV RMSE  : {cv_rmse:.4f}")

    results.append({
        "Model": name,
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R²": round(r2, 4),
        "CV RMSE": round(cv_rmse, 4)
    })

    # Visualization
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_test, y=preds)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Degradation Rate (%)")
    plt.ylabel("Predicted")
    plt.title(f"{name} - Predicted vs Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Time Series Model
time_series_model.fit(X_scaled, y)
ts_preds = time_series_model.predict(X_scaled)

mae = mean_absolute_error(y, ts_preds)
mse = mean_squared_error(y, ts_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y, ts_preds)
cv_rmse_ts = -np.mean(cross_val_score(time_series_model, X_scaled, y, scoring="neg_root_mean_squared_error", cv=tscv))

print(f"\n📌 Time Series Model (LR + Lag)")
print(f"MAE      : {mae:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")
print(f"CV RMSE  : {cv_rmse_ts:.4f}")

results.append({
    "Model": "Time Series (LR + Lag)",
    "MAE": round(mae, 4),
    "MSE": round(mse, 4),
    "RMSE": round(rmse, 4),
    "R²": round(r2, 4),
    "CV RMSE": round(cv_rmse_ts, 4)
})

# Plot full time series prediction
plt.figure(figsize=(10, 4))
plt.plot(y.values, label="Actual")
plt.plot(ts_preds, label="Predicted", alpha=0.7)
plt.title("Time Series Model Fit (LR + Lag)")
plt.xlabel("Index")
plt.ylabel("Degradation Rate (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Hyperparameter Tuning for Random Forest
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5,
                           scoring="neg_root_mean_squared_error", n_jobs=-1)
grid_search.fit(X_scaled, y)
best_rf = grid_search.best_estimator_

# Evaluate Tuned RF
best_rf_preds = best_rf.predict(X_test)
mae = mean_absolute_error(y_test, best_rf_preds)
mse = mean_squared_error(y_test, best_rf_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, best_rf_preds)
cv_rmse = -grid_search.best_score_

print(f"\n📌 Tuned Random Forest")
print(f"Best Params : {grid_search.best_params_}")
print(f"MAE         : {mae:.4f}")
print(f"MSE         : {mse:.4f}")
print(f"RMSE        : {rmse:.4f}")
print(f"R² Score    : {r2:.4f}")
print(f"CV RMSE     : {cv_rmse:.4f}")

results.append({
    "Model": "Tuned Random Forest",
    "MAE": round(mae, 4),
    "MSE": round(mse, 4),
    "RMSE": round(rmse, 4),
    "R²": round(r2, 4),
    "CV RMSE": round(cv_rmse, 4)
})

# Scatter for Tuned RF
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=best_rf_preds)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Degradation Rate (%)")
plt.ylabel("Predicted")
plt.title("Tuned Random Forest - Predicted vs Actual")
plt.grid(True)
plt.tight_layout()
plt.show()

# Summary Table
comparison_df = pd.DataFrame(results)
print("\n✅ Final Model Comparison Table:\n")
print(comparison_df.to_string(index=False))

# Metric Bar Plots
for metric in ["MAE", "MSE", "RMSE", "R²", "CV RMSE"]:
    plt.figure(figsize=(8, 4))
    sns.barplot(data=comparison_df, x="Model", y=metric)
    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
