# ==============================
# 1. IMPORT LIBRARIES
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import os
# ==============================
# 2. LOAD DATASET
# ==============================



data = pd.read_csv("bank-additional-full.csv", sep=';')

print("\n First 5 Rows:")
print(data.head())

print("\n Dataset Shape:")
print(data.shape)

print("\n Dataset Info:")
print(data.info())

print("\n Summary Statistics:")
print(data.describe())

print("\n Target Distribution:")
print(data['y'].value_counts())
# ==============================
# 3. DATA CLEANING
# ==============================

# Check missing values
print("\n Missing Values:")
print(data.isnull().sum())

# Replace "unknown" with NaN
data.replace("unknown", np.nan, inplace=True)

# Check again
print("\n Missing After Replacing 'unknown':")
print(data.isnull().sum())

# Fill missing categorical values with mode
for col in data.select_dtypes(include='object').columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

print("\n Missing After Imputation:")
print(data.isnull().sum())

# ==============================
# 4. EXPLORATORY DATA ANALYSIS
# ==============================

# Plot target distribution
plt.figure()
sns.countplot(x='y', data=data)
plt.title("Target Distribution")
plt.show()

# Correlation heatmap (numeric only)
plt.figure(figsize=(10,8))
sns.heatmap(data.select_dtypes(include=np.number).corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ==============================
# 5. FEATURE ENGINEERING
# ==============================

# Convert target to binary
data['y'] = data['y'].map({'no':0, 'yes':1})

# One-Hot Encoding
data = pd.get_dummies(data, drop_first=True)

print("\n Dataset Shape After Encoding:")
print(data.shape)

# ==============================
# 6. TRAIN TEST SPLIT
# ==============================

# Create model directory
os.makedirs("model", exist_ok=True)

X = data.drop("y", axis=1)
y = data["y"]
# Save training column names for Streamlit alignment
joblib.dump(X.columns.tolist(), "model/train_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")

# ==============================
# 7. MODEL TRAINING
# ==============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, eval_metric='logloss')
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)

    }

    results.append(metrics)

    joblib.dump(model, f"model/{name}.pkl")


# ==============================
# 8. RESULTS TABLE
# ==============================

results_df = pd.DataFrame(results)
print("\n FINAL MODEL COMPARISON:")
print(results_df)

results_df.to_csv("model/model_comparison_results.csv", index=False)