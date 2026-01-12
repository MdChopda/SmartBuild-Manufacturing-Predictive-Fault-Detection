# -*- coding: utf-8 -*-
"""
Solution Q2: Predicting Error Type (Classification)
"""

import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("1. Loading Data...")
df_prod = pd.read_csv('Production_Log_01.csv')
df_mach = pd.read_csv('Machine_Settings_Log_01.csv')

# Clean column names
df_prod.columns = df_prod.columns.str.strip()
df_mach.columns = df_mach.columns.str.strip()

# Merge tables
df = pd.merge(df_prod, df_mach, on='configuration_log_ID', how='left')

# ---------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------
print("2. Cleaning Data...")

# Drop IDs and post-production metrics to prevent data leakage
cols_drop = [
    'id', 'configuration_log_ID', 'weight_in_g', 
    'distortion', 'roughness', 'reflectionScore', 
    'nicesness', 'smartness', 'Quality'
]
df_clean = df.drop(columns=cols_drop, errors='ignore')

# Handle missing values in target variable
df_clean['error_type'] = df_clean['error_type'].fillna('No Error')

# Encode Target Variable
le_target = LabelEncoder()
df_clean['error_type'] = le_target.fit_transform(df_clean['error_type'].astype(str))

# Encode Feature Columns
for col in df_clean.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_clean[col] = df_clean[col].fillna('Missing')
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

# ---------------------------------------------------------
# 3. MODEL TRAINING
# ---------------------------------------------------------
print("3. Training Model...")

# Define Features (X) and Target (y)
X = df_clean.drop(columns=['error_type', 'error'], errors='ignore')
y = df_clean['error_type']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Classifier
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. EVALUATION
# ---------------------------------------------------------
print("4. Evaluating...")
preds = model.predict(X_test)

# Calculate Accuracy Score
acc = accuracy_score(y_test, preds)
print("\n--- Q2 RESULTS ---")
print(f"Accuracy: {acc:.2%}")

# Display Confusion Matrix
print("\nConfusion Matrix (Actual vs Predicted):")
y_test_names = le_target.inverse_transform(y_test)
preds_names = le_target.inverse_transform(preds)

cm = pd.crosstab(y_test_names, preds_names, rownames=['Actual'], colnames=['Predicted'])
print(cm)

# Plot Feature Importance
plot_importance(model, max_num_features=10, title='Important Factors for Errors')
plt.show()