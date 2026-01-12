# -*- coding: utf-8 -*-
"""
Solution Q1: Predicting Weight (Regression)
"""

import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# 1. Load Data
df_prod = pd.read_csv('Production_Log_01.csv')
df_mach = pd.read_csv('Machine_Settings_Log_01.csv')

# Clean column names
df_prod.columns = df_prod.columns.str.strip()
df_mach.columns = df_mach.columns.str.strip()

# Merge tables
df = pd.merge(df_prod, df_mach, on='configuration_log_ID', how='left')

# 2. Preprocessing
# Drop ID columns
cols_drop = ['id', 'configuration_log_ID', 'weight_in_g']
df_clean = df.drop(columns=cols_drop, errors='ignore')

# Clean target variable (convert to numeric)
df_clean['weight_in_kg'] = pd.to_numeric(df_clean['weight_in_kg'], errors='coerce')
df_clean = df_clean.dropna(subset=['weight_in_kg'])

# Remove invalid data (simple filter)
df_clean = df_clean[df_clean['weight_in_kg'] < 1000000]

# Encode text columns using LabelEncoder
for col in df_clean.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_clean[col] = df_clean[col].fillna('Missing')
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

# 3. Model Training
X = df_clean.drop(columns=['weight_in_kg', 'error', 'error_type'], errors='ignore')
y = df_clean['weight_in_kg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost Regressor model
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation
preds = model.predict(X_test)

# Calculate and print Mean Absolute Error
mae = mean_absolute_error(y_test, preds)
print("\n--- Q1 RESULTS ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Plot the most important features
plot_importance(model, max_num_features=10, title='Important Factors for Weight')
plt.show()

