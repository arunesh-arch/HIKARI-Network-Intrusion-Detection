
# ============================================================
# HIKARI 2021 - NETWORK TRAFFIC CLASSIFICATION
# Multiple Machine Learning Algorithms
# Automatic Best Model Selection
# Classification: Benign (0) vs Attack (1)
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = (
    r"C:\Users\arune\Desktop\Internships and projects"
    r"\AI_Sem1_Class_Project\Network_Traffic_Analysis"
    r"\HIKARI_Project\ALLFLOWMETER_HIKARI2021.csv"
    r"\ALLFLOWMETER_HIKARI2021.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "hikari_best_model.pkl"
)

COMPARISON_PATH = os.path.join(
    BASE_DIR,
    "model_comparison.csv"
)

# ============================================================
# 2. CHECK DATASET FILE
# ============================================================

print("\n==========================================")
print("HIKARI NETWORK TRAFFIC CLASSIFICATION")
print("==========================================")

print("\nChecking dataset file...")

if not os.path.isfile(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}"
    )

if not os.access(DATASET_PATH, os.R_OK):
    raise PermissionError(
        "Cannot read dataset. Check Windows permissions "
        "and close applications using the CSV file."
    )

print("Dataset file found!")

# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading HIKARI dataset...")

try:
    df = pd.read_csv(
        DATASET_PATH,
        low_memory=False
    )

except PermissionError:
    raise PermissionError(
        "Windows denied access to the dataset. "
        "Close Excel and check file permissions."
    )

print("Dataset loaded successfully!")

print("Original dataset shape:", df.shape)

# ============================================================
# 4. REMOVE UNNECESSARY COLUMNS
# ============================================================

print("\nRemoving unnecessary columns...")

drop_columns = [
    'uid',
    'originh',
    'originp',
    'responh',
    'responp',
    'traffic_category',
    'Unnamed: 0',
    'Unnamed: 0.1',
    'Unnamed: 0.2'
]

df.drop(
    columns=drop_columns,
    errors='ignore',
    inplace=True
)

print("Unnecessary columns removed!")

# ============================================================
# 5. DATA CLEANING
# ============================================================

print("\nCleaning dataset...")

if 'Label' not in df.columns:
    raise ValueError(
        "Target column 'Label' is missing!"
    )

# Replace infinite values
df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# Convert target to numeric
df['Label'] = pd.to_numeric(
    df['Label'],
    errors='coerce'
)

# Remove missing target rows
df.dropna(
    subset=['Label'],
    inplace=True
)

# Validate target labels
if not df['Label'].isin([0, 1]).all():
    raise ValueError(
        "Unexpected target labels. Expected only 0 and 1."
    )

df['Label'] = df['Label'].astype(int)

# Convert features to numeric
X = df.drop(
    columns=['Label']
)

X = X.apply(
    pd.to_numeric,
    errors='coerce'
)

# Remove completely empty features
X.dropna(
    axis=1,
    how='all',
    inplace=True
)

# Remove duplicate records using features and target
clean_df = X.copy()

clean_df['Label'] = df['Label'].values

before_duplicates = len(clean_df)

clean_df.drop_duplicates(
    inplace=True
)

after_duplicates = len(clean_df)

print(
    "Duplicate records removed:",
    before_duplicates - after_duplicates
)

# Separate features and target
X = clean_df.drop(
    columns=['Label']
)

y = clean_df['Label']

# Save feature names
feature_columns = X.columns.tolist()

print("\nCleaned dataset shape:", X.shape)

print("\nTarget distribution:")

print(
    y.value_counts()
)

print("\nTarget percentages:")

print(
    (y.value_counts(normalize=True) * 100).round(2)
)

# Validate target classes
if y.nunique() != 2:
    raise ValueError(
        "Dataset must contain both Benign and Attack classes."
    )

# ============================================================
# 6. SPLIT DATASET
# 70% TRAINING
# 15% VALIDATION
# 15% TESTING
# ============================================================

print("\nSplitting dataset...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

# ============================================================
# 7. DEFINE MACHINE LEARNING ALGORITHMS
# ============================================================

print("\nPreparing machine learning algorithms...")

models = {

    # Logistic Regression
    "Logistic Regression": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    # Decision Tree
    "Decision Tree": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=20,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    # Random Forest
    "Random Forest": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )
        )
    ]),

    # Extra Trees
    "Extra Trees": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )
        )
    ])
}

# ============================================================
# 8. TRAIN AND EVALUATE ALL ALGORITHMS
# ============================================================

print("\n==========================================")
print("TRAINING MACHINE LEARNING MODELS")
print("==========================================")

results = []

trained_models = {}

for name, model in models.items():

    print("\n------------------------------------------")
    print("Training:", name)
    print("------------------------------------------")

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Validation predictions
    predictions = model.predict(
        X_val
    )

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(
        y_val,
        predictions
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val,
        probabilities
    )

    pr_auc = average_precision_score(
        y_val,
        probabilities
    )

    # Store results
    results.append({

        "Algorithm": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc,

        "PR-AUC": pr_auc

    })

    trained_models[name] = model

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))
    print("PR-AUC   :", round(pr_auc, 4))

# ============================================================
# 9. COMPARE ALL ALGORITHMS
# ============================================================

print("\n==========================================")
print("MODEL PERFORMANCE COMPARISON")
print("==========================================")

results_df = pd.DataFrame(results)

results_df.sort_values(
    by="F1 Score",
    ascending=False,
    inplace=True
)

results_df.reset_index(
    drop=True,
    inplace=True
)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# Save comparison table
results_df.to_csv(
    COMPARISON_PATH,
    index=False
)

print("\nModel comparison saved successfully!")

# ============================================================
# 10. SELECT BEST MODEL
# ============================================================

print("\n==========================================")
print("SELECTING BEST MODEL")
print("==========================================")

best_algorithm = results_df.iloc[0]["Algorithm"]

best_model = trained_models[best_algorithm]

print("\nSelected Algorithm:", best_algorithm)

print(
    "Validation F1 Score:",
    round(
        results_df.iloc[0]["F1 Score"],
        4
    )
)

# ============================================================
# 11. FINAL TEST EVALUATION
# ============================================================

print("\n==========================================")
print("FINAL TEST PERFORMANCE")
print("==========================================")

test_predictions = best_model.predict(
    X_test
)

test_probabilities = best_model.predict_proba(
    X_test
)[:, 1]

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

test_pr_auc = average_precision_score(
    y_test,
    test_probabilities
)

print("\nAccuracy :", round(test_accuracy, 4))
print("Precision:", round(test_precision, 4))
print("Recall   :", round(test_recall, 4))
print("F1 Score :", round(test_f1, 4))
print("ROC-AUC  :", round(test_roc_auc, 4))
print("PR-AUC   :", round(test_pr_auc, 4))

# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Benign",
            "Attack"
        ],
        zero_division=0
    )
)

# Confusion matrix
print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)

# ============================================================
# 12. SAVE BEST MODEL AND FEATURE INFORMATION
# ============================================================

print("\nSaving best model...")

model_package = {

    "model": best_model,

    "features": feature_columns,

    "algorithm": best_algorithm,

    "validation_f1": float(
        results_df.iloc[0]["F1 Score"]
    ),

    "test_metrics": {

        "accuracy": float(test_accuracy),

        "precision": float(test_precision),

        "recall": float(test_recall),

        "f1_score": float(test_f1),

        "roc_auc": float(test_roc_auc),

        "pr_auc": float(test_pr_auc)

    }

}

joblib.dump(
    model_package,
    MODEL_PATH
)

print("\nModel saved successfully!")

# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n")
print("==========================================")
print("       HIKARI PROJECT COMPLETED")
print("==========================================")

print("\nDataset Records:", len(clean_df))

print("Number of Features:", len(feature_columns))

print("\nAlgorithms Compared:")

for algorithm in models:
    print("-", algorithm)

print("\nBest Algorithm:", best_algorithm)

print("Test Accuracy:", round(test_accuracy, 4))

print("Test Precision:", round(test_precision, 4))

print("Test Recall:", round(test_recall, 4))

print("Test F1 Score:", round(test_f1, 4))

print("\nGenerated Files:")

print("1. hikari_best_model.pkl")

print("2. model_comparison.csv")

print("\nModel Location:")
print(MODEL_PATH)

print("\n==========================================")
print("       TRAINING FINISHED SUCCESSFULLY")
print("==========================================")
