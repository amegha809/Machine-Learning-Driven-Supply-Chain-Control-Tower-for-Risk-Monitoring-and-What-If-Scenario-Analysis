
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,recall_score,precision_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import os 
import joblib

# Load the processed dataset
df = pd.read_csv("processed/supply_chain_processed.csv")

print("Loaded dataset shape:", df.shape)
print("Number of input rows:", len(df))
print("Target distribution:")
print(df["Late_delivery_risk"].value_counts())
# Separate input features and target
X = df.drop(columns=["Late_delivery_risk"])
y = df["Late_delivery_risk"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully!")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Create the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
print("Training Random Forest model...")

rf_model.fit(X_train, y_train)

print("Model training completed successfully!")

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Random Forest Model Evaluation ---")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Create folder for trained models
os.makedirs("models", exist_ok=True)

# Save the trained Random Forest model
joblib.dump(rf_model, "models/random_forest_model.pkl")

print("\nRandom Forest model saved successfully!")
print("Location: models/random_forest_model.pkl")



# Train Logistic Regression with feature scaling
lr_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        random_state=42
    )
)

print("\nTraining Logistic Regression model...")

lr_model.fit(X_train, y_train)

print("Logistic Regression training completed!")

# Make predictions
lr_pred = lr_model.predict(X_test)

# Evaluate the model
lr_accuracy = accuracy_score(y_test, lr_pred)

print("\n--- Logistic Regression Evaluation ---")
print("Accuracy:", round(lr_accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_pred))


# Compare both trained models

rf_pred = rf_model.predict(X_test)

comparison = pd.DataFrame({
    "Model": ["Random Forest", "Logistic Regression"],
    "Accuracy": [
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, lr_pred)
    ],
    "Precision": [
        precision_score(y_test, rf_pred),
        precision_score(y_test, lr_pred)
    ],
    "Recall": [
        recall_score(y_test, rf_pred),
        recall_score(y_test, lr_pred)
    ],
    "F1-Score": [
        f1_score(y_test, rf_pred),
        f1_score(y_test, lr_pred)
    ]
})

print("\n--- Model Comparison ---")
print(comparison.round(4))

# Save comparison results
comparison.to_csv("models/model_comparison.csv", index=False)

print("\nModel comparison saved successfully!")


# Feature Importance using Random Forest

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- Top 10 Important Features ---")
print(feature_importance.head(10).round(4))

# Save feature importance results
feature_importance.to_csv(
    "models/feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")


import matplotlib.pyplot as plt

# Select the top 10 important features
top_features = feature_importance.head(10).sort_values(
    by="Importance",
    ascending=True
)

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Features for Late Delivery Risk Prediction")
plt.tight_layout()

# Save the chart
plt.savefig("models/feature_importance.png", dpi=300)

print("\nFeature importance chart saved successfully!")

# Display the chart
plt.show()
