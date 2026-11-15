
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("processed/supply_chain_processed.csv")

print("Dataset shape:", df.shape)

X = df.drop(columns=["Late_delivery_risk"])
y = df["Late_delivery_risk"]

# Use the same train-test split as the original project
# Split dataset into training and testing sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

print("\nTraining set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Testing set:", X_test.shape)

# Train original Random Forest for comparison
print("\nTraining original Random Forest...")

original_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

original_model.fit(X_train_full, y_train_full)
original_pred = original_model.predict(X_test)

# Define parameters for tuning
param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [None, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.7],
    "class_weight": [None, "balanced", "balanced_subsample"]
}

# Cross-validation
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# Create Randomized Search
print("\nStarting Random Forest hyperparameter tuning...")
print("This may take some time.")

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=6,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=1,
    verbose=2
)

random_search.fit(X_train, y_train)

# Get best model
tuned_model = random_search.best_estimator_

print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(round(random_search.best_score_ * 100, 2), "%")

# Evaluate tuned model on validation data
val_pred = tuned_model.predict(X_val)

print("\n--- Validation Evaluation ---")
print("Accuracy:", round(accuracy_score(y_val, val_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_val, val_pred, zero_division=0) * 100, 2), "%")
print("Recall:", round(recall_score(y_val, val_pred, zero_division=0) * 100, 2), "%")
print("F1 Score:", round(f1_score(y_val, val_pred, zero_division=0) * 100, 2), "%")

# Find the best prediction threshold using validation data
print("\n--- Threshold Tuning ---")

val_probabilities = tuned_model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in [0.30, 0.32, 0.33, 0.35, 0.37, 0.40, 0.45, 0.50]:
    val_threshold_pred = (val_probabilities >= threshold).astype(int)

    precision = precision_score(y_val, val_threshold_pred, zero_division=0)
    recall = recall_score(y_val, val_threshold_pred, zero_division=0)
    f1 = f1_score(y_val, val_threshold_pred, zero_division=0)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

threshold_df = pd.DataFrame(threshold_results)

print(threshold_df.round(4))

# Select threshold with the highest validation F1-score
best_threshold = threshold_df.loc[
    threshold_df["F1-Score"].idxmax(), "Threshold"
]

print("\nBest Threshold:", best_threshold)
# Final evaluation on untouched test data
test_probabilities = tuned_model.predict_proba(X_test)[:, 1]

tuned_pred = (test_probabilities >= best_threshold).astype(int)

print("\n--- Final Tuned Random Forest Evaluation ---")
print("Selected Threshold:", best_threshold)
print("Accuracy:", round(accuracy_score(y_test, tuned_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, tuned_pred, zero_division=0) * 100, 2), "%")
print("Recall:", round(recall_score(y_test, tuned_pred, zero_division=0) * 100, 2), "%")
print("F1 Score:", round(f1_score(y_test, tuned_pred, zero_division=0) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, tuned_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tuned_pred))
# Compare original and tuned models
comparison = pd.DataFrame({
    "Model": ["Original Random Forest", "Tuned Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, original_pred),
        accuracy_score(y_test, tuned_pred)
    ],
    "Precision": [
        precision_score(y_test, original_pred, zero_division=0),
        precision_score(y_test, tuned_pred, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, original_pred, zero_division=0),
        recall_score(y_test, tuned_pred, zero_division=0)
    ],
    "F1-Score": [
        f1_score(y_test, original_pred, zero_division=0),
        f1_score(y_test, tuned_pred, zero_division=0)
    ]
})

print("\n--- Original vs Tuned Random Forest ---")
print(comparison.round(4))

# Save results separately
os.makedirs("models", exist_ok=True)

comparison.to_csv(
    "models/random_forest_tuning_comparison.csv",
    index=False
)

joblib.dump(
    tuned_model,
    "models/random_forest_tuned_model.pkl"
)

print("\nTuned model saved successfully!")
print("Location: models/random_forest_tuned_model.pkl")
print("Comparison saved successfully!")
