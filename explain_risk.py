
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("processed/supply_chain_processed.csv")
X = df.drop(columns=["Late_delivery_risk"])

# Load trained Random Forest
rf_model = joblib.load("models/random_forest_model.pkl")

print("Random Forest model loaded successfully!")

# Select 20 sample orders
X_sample = X.sample(n=20, random_state=42)

# Explain the complete Random Forest model
print("Calculating SHAP values using the complete Random Forest...")

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample)

# Extract SHAP values for late-delivery class
if isinstance(shap_values, list):
    late_shap_values = shap_values[1]
elif len(shap_values.shape) == 3:
    late_shap_values = shap_values[:, :, 1]
else:
    late_shap_values = shap_values

print("SHAP values calculated successfully!")

# Generate summary plot
shap.summary_plot(
    late_shap_values,
    X_sample,
    show=False
)

plt.tight_layout()

os.makedirs("models", exist_ok=True)

plt.savefig(
    "models/shap_summary.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Chart saved to models/shap_summary.png")

# Calculate average absolute SHAP importance
importance = np.abs(late_shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean Absolute SHAP Value": importance
})

importance_df = importance_df.sort_values(
    by="Mean Absolute SHAP Value",
    ascending=False
)

# Save feature importance
importance_df.to_csv(
    "reports/shap_feature_importance.csv",
    index=False
)

print("\nTop 10 Important Features:")
print(importance_df.head(10).to_string(index=False))

print("\nFeature importance saved to reports/shap_feature_importance.csv")
