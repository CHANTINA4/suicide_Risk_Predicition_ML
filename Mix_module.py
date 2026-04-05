# ==========================================
# FINAL OPTIMIZED ML PIPELINE (FULL PROJECT)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------
# 1️⃣ LOAD DATA
# -----------------------------------
df = pd.read_csv("Final_MLdataSet.csv")

# -----------------------------------
# 2️⃣ HANDLE MISSING VALUES
# -----------------------------------
df['Family Problem'] = df['Family Problem'].fillna('Unknown')

# -----------------------------------
# 3️⃣ ENCODE FEATURES
# -----------------------------------
for col in df.columns:
    df[col] = df[col].astype(str)   # force all to string
    df[col] = df[col].str.strip()   # remove spaces
    df[col] = LabelEncoder().fit_transform(df[col])
print(df.select_dtypes(include=['object']).columns)

print(df.head())
print(df.dtypes)
# -----------------------------------
# 4️⃣ SPLIT
# -----------------------------------
X = df.drop("Suicide Attempt", axis=1)
y = df["Suicide Attempt"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------------
# 5️⃣ SCALING
# -----------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------
# STORE RESULTS
# -----------------------------------
results = []

# -----------------------------------
# FUNCTION TO EVALUATE MODEL
# -----------------------------------
def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    try:
        y_prob = model.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=np.unique(y_test))
        auc_score = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except:
        auc_score = 0

    results.append({
        "Model": name,
        "Accuracy": acc,
        "ROC_AUC": auc_score
    })

    print("\n==============================")
    print(f"MODEL: {name}")
    print("==============================")
    print("Accuracy:", acc)
    print("ROC AUC:", auc_score)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

# -----------------------------------
# 6️⃣ MODELS
# -----------------------------------
log_grid = GridSearchCV(LogisticRegression(max_iter=2000), {'C': [0.1, 1, 10]}, cv=5)

svm_grid = GridSearchCV(SVC(probability=True),
                        {'C': [1, 10], 'gamma': [0.1, 0.01], 'kernel': ['rbf']},
                        cv=5)

knn_grid = GridSearchCV(KNeighborsClassifier(),
                        {'n_neighbors': [3,5,7,9], 'weights': ['uniform','distance']},
                        cv=5)

dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                       {'max_depth': [3,5,7], 'min_samples_split': [2,5,10]},
                       cv=5)

# -----------------------------------
# 7️⃣ TRAIN + EVALUATE
# -----------------------------------
evaluate_model("Logistic Regression", log_grid)
evaluate_model("SVM", svm_grid)
evaluate_model("KNN", knn_grid)
evaluate_model("Decision Tree", dt_grid)

print("\nStored Results:", results)

# -----------------------------------
# 📊 8️⃣ FINAL COMPARISON GRAPH
# -----------------------------------
results_df = pd.DataFrame(results)
x = np.arange(len(results_df["Model"]))

plt.figure(figsize=(10,6))

plt.bar(x - 0.2, results_df["Accuracy"], width=0.4, label="Accuracy")
plt.bar(x + 0.2, results_df["ROC_AUC"], width=0.4, label="ROC AUC")

plt.xticks(x, results_df["Model"], rotation=25, ha='right')
plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Final Model Comparison")

for i, v in enumerate(results_df["Accuracy"]):
    plt.text(i - 0.2, v + 0.01, f"{v:.2f}", ha='center')

for i, v in enumerate(results_df["ROC_AUC"]):
    plt.text(i + 0.2, v + 0.01, f"{v:.2f}", ha='center')

plt.legend()
plt.ylim(0,1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# -----------------------------------
# 📊 9️⃣ INDIVIDUAL MODEL GRAPHS
# -----------------------------------
for res in results:
    plt.figure(figsize=(5,4))
    
    metrics = ["Accuracy", "ROC_AUC"]
    values = [res["Accuracy"], res["ROC_AUC"]]
    
    plt.bar(metrics, values)
    plt.title(f"{res['Model']} Performance")
    plt.ylabel("Score")
    plt.ylim(0,1)
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 🔟 CONFUSION MATRIX (VISUAL)
# ==========================================
models = [
    ("Logistic Regression", log_grid),
    ("SVM", svm_grid),
    ("KNN", knn_grid),
    ("Decision Tree", dt_grid)
]

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# ==========================================
# 1️⃣1️⃣ ROC CURVE
# ==========================================
plt.figure(figsize=(8,6))

for name, model in models:
    model.fit(X_train, y_train)
    
    try:
        y_prob = model.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=np.unique(y_test))
        
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    except:
        continue

plt.plot([0,1], [0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ==========================================
# 1️⃣2️⃣ FEATURE IMPORTANCE
# ==========================================
best_dt = dt_grid.best_estimator_

importances = best_dt.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,5))
plt.bar(feature_df["Feature"], feature_df["Importance"])

plt.xticks(rotation=45, ha='right')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance (Decision Tree)")

plt.tight_layout()
plt.show()
# update v1