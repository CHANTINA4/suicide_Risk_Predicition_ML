# ==============================
# CS1138 PROJECT - WEEK 11
# EDA + Preprocessing + Baseline Model
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# -----------------------------------
# 1️⃣ LOAD DATASET
# -----------------------------------
df = pd.read_csv("Final_SP_dataSet.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# -----------------------------------
# 2️⃣ HANDLE MISSING VALUES
# -----------------------------------
# Family Problem has many missing values (~37%)
# Instead of dropping rows, we replace missing with 'Unknown'

df['Family Problem'] = df['Family Problem'].fillna('Unknown')

print("\nMissing After Cleaning:\n", df.isnull().sum())

# -----------------------------------
# 3️⃣ ENCODE CATEGORICAL VARIABLES
# -----------------------------------
# Machine learning models need numeric data

le = LabelEncoder()

# Encode ALL non-numeric columns
for col in df.columns:
    if df[col].dtype != 'int64' and df[col].dtype != 'float64':
        df[col] = LabelEncoder().fit_transform(df[col])



print("\nData After Encoding:\n", df.head())
print("\nAfter Encoding Data Types:\n")
print(df.dtypes)


# -----------------------------------
# 4️⃣ CORRELATION ANALYSIS
# -----------------------------------
numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(12,8))
print(df.dtypes)
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# -----------------------------------
# 5️⃣ DEFINE FEATURES & TARGET
# -----------------------------------
# Target variable = Suicide Attempt

X = df.drop("Suicide Attempt", axis=1)
y = df["Suicide Attempt"]

# -----------------------------------
# 6️⃣ TRAIN-TEST SPLIT
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("Balanced Classes:\n", pd.Series(y_train).value_counts())

# -----------------------------------
# 7️⃣ FEATURE SCALING
# -----------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------
# 8️⃣ BASELINE MODEL - LOGISTIC REGRESSION
# -----------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----------------------------------
# 9️⃣ EVALUATION METRICS
# -----------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------------
# -----------------------------------
# 🔟 MULTICLASS ROC CURVE (FIXED)
# -----------------------------------

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Binarize the output
classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)

# Get predicted probabilities
y_prob = model.predict_proba(X_test)

plt.figure()

for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC = {roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend()
plt.show()