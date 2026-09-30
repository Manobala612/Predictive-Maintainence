# ==============================
# Import Libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("/content/ai4i2020.csv")

# ==============================
# Basic Information
# ==============================
print("="*50)
print("First Five Rows")
print(df.head())

print("="*50)
print("Dataset Shape")
print(df.shape)

print("="*50)
print("Columns")
print(df.columns)

print("="*50)
print("Information")
print(df.info())

print("="*50)
print("Statistics")
print(df.describe())

# ==============================
# Missing Values
# ==============================
print("="*50)
print("Missing Values")
print(df.isnull().sum())

plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# ==============================
# Duplicate Values
# ==============================
print("="*50)
print("Duplicate Values :", df.duplicated().sum())

# ==============================
# Target Distribution
# ==============================
plt.figure(figsize=(6,4))
sns.countplot(x="Machine failure", data=df)
plt.title("Machine Failure Distribution")
plt.show()

# ==============================
# Encode Categorical Columns
# ==============================
encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ==============================
# Correlation Heatmap
# ==============================
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ==============================
# Feature Distribution
# ==============================
df.hist(figsize=(15,12))
plt.tight_layout()
plt.show()

# ==============================
# Features and Target
# ==============================
X = df.drop([
    "Machine failure",
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
], axis=1)

y = df["Machine failure"]

# ==============================
# Train Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=7,
    stratify=y
)

# ==============================
# Random Forest Model
# ==============================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=7,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ==============================
# Prediction
# ==============================
y_pred = model.predict(X_test)

# ==============================
# Accuracy
# ==============================
accuracy = accuracy_score(y_test, y_pred)

print("="*50)
print("Accuracy :", accuracy)

# ==============================
# Confusion Matrix
# ==============================
cm = confusion_matrix(y_test, y_pred)

print("="*50)
print("Confusion Matrix")
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No Failure','Failure'],
    yticklabels=['No Failure','Failure']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==============================
# Classification Report
# ==============================
print("="*50)
print("Classification Report")
print(classification_report(y_test, y_pred))

# ==============================
# Feature Importance
# ==============================
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,6))
sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)
plt.title("Feature Importance")
plt.show()

# ==============================
# ROC Curve
# ==============================
prob = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, prob)

auc = roc_auc_score(y_test, prob)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label="AUC = %.3f"%auc)
plt.plot([0,1],[0,1],'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ==============================
# Learning Curve
# ==============================
train_sizes, train_scores, test_scores = learning_curve(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_mean, marker='o', label="Training Accuracy")
plt.plot(train_sizes, test_mean, marker='o', label="Validation Accuracy")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.show()

# ==============================
# Accuracy Bar Graph
# ==============================
plt.figure(figsize=(5,5))
plt.bar(["Random Forest"], [accuracy], color="green")
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.show()

# ==============================
# Save Model
# ==============================
joblib.dump(model, "machine_failure_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("="*50)
print("Model Saved Successfully!")
print("Files Created:")
print("1. machine_failure_model.pkl")
print("2. label_encoders.pkl")
print("3. feature_names.pkl")
