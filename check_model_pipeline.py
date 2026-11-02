import pandas as pd
import joblib

# Load the saved Random Forest model
model = joblib.load("models/random_forest_model.pkl")

# Load the processed dataset
df = pd.read_csv("processed/supply_chain_processed.csv")

# Separate features and target
X = df.drop(columns=["Late_delivery_risk"])
y = df["Late_delivery_risk"]

print("----- MODEL PIPELINE CHECK -----")

print("\nDataset shape:", df.shape)
print("Feature count:", X.shape[1])
print("Target distribution:")
print(y.value_counts())

print("\nModel type:", type(model).__name__)
print("Model expects:", model.n_features_in_, "features")

# Check the number of features
if X.shape[1] == model.n_features_in_:
    print("\nPASS: Feature count matches.")
else:
    print("\nFAIL: Feature count does not match.")

# Check feature names when available
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    dataset_features = list(X.columns)

    if model_features == dataset_features:
        print("PASS: Feature names and order match.")
    else:
        print("WARNING: Feature names or order differ.")
        print("\nModel features:")
        print(model_features)
        print("\nDataset features:")
        print(dataset_features)
else:
    print("\nModel does not store feature names.")
    print("Feature count alone does not prove correct feature order.")