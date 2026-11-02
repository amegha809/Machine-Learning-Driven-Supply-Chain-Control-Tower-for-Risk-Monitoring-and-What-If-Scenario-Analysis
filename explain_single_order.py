
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

# Load dataset and trained model
df = pd.read_csv("processed/supply_chain_processed.csv")
X = df.drop(columns=["Late_delivery_risk"])

rf_model = joblib.load("models/random_forest_model.pkl")

# Select one order to explain
order = X.iloc[[0]]

# Predict risk
prediction = rf_model.predict(order)[0]
probabilities = rf_model.predict_proba(order)[0]

print("\n--- Individual Order Risk Explanation ---")
print("Predicted class:", prediction)
print("Risk category:", "HIGH" if prediction == 1 else "LOW")
print(f"Probability of low risk: {probabilities[0] * 100:.2f}%")
print(f"Probability of late delivery: {probabilities[1] * 100:.2f}%")

# Explain the complete Random Forest model
print("\nCalculating SHAP values for the complete Random Forest...")

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(order)

# Extract contributions for class 1 (late delivery)
if isinstance(shap_values, list):
    mean_shap_values = shap_values[1][0]
elif len(shap_values.shape) == 3:
    mean_shap_values = shap_values[0, :, 1]
else:
    mean_shap_values = shap_values[0]

# Create a table of feature contributions
explanation = pd.DataFrame({
    "Feature": X.columns,
    "Feature Value": order.iloc[0].values,
    "SHAP Contribution": mean_shap_values
})

explanation["Direction"] = np.where(
    explanation["SHAP Contribution"] > 0,
    "Pushes model output higher",
    "Pushes model output lower"
)

explanation["Absolute Contribution"] = (
    explanation["SHAP Contribution"].abs()
)

explanation = explanation.sort_values(
    by="Absolute Contribution",
    ascending=False
)

print("\nTop 10 feature contributions:")
print(explanation.head(10).to_string(index=False))

# Generate Simple Language Explanation

print("\n--- Simple Language Explanation ---")

# Select features that increase late-delivery risk
risk_increasing = explanation[
    explanation["SHAP Contribution"] > 0
].head(3)

# Select features that decrease late-delivery risk
risk_decreasing = explanation[
    explanation["SHAP Contribution"] < 0
].head(3)

# Convert feature names into simple language
feature_names = {
    "Shipping Mode_Standard Class": "Standard Class shipping",
    "Shipping Mode_Second Class": "Second Class shipping",
    "Shipping Mode_Same Day": "Same Day shipping",
    "Days for shipment (scheduled)": "scheduled shipping duration",
    "Type_TRANSFER": "transfer payment type",
    "Type_DEBIT": "debit payment type",
    "Type_PAYMENT": "payment transaction type",
    "Order Profit Per Order": "profit earned per order",
    "Benefit per order": "benefit received per order",
    "Order Item Discount": "discount given on the order",
    "Order Item Discount Rate": "discount percentage",
    "Sales per customer": "sales value per customer",
    "Order Item Total": "total value of the order",
    "Product Price": "product price",
    "Sales": "sales amount",
    "Order Item Quantity": "quantity of products ordered"
}

print("\nFactors contributing to HIGH late-delivery risk:")

if len(risk_increasing) == 0:
    print("No major factors increased the predicted late-delivery risk.")
else:
    for _, row in risk_increasing.iterrows():
        feature = feature_names.get(
            row["Feature"], row["Feature"]
        )
        print(f"- {feature} contributed towards higher late-delivery risk.")

print("\nFactors contributing to LOW late-delivery risk:")

if len(risk_decreasing) == 0:
    print("No major factors decreased the predicted late-delivery risk.")
else:
    for _, row in risk_decreasing.iterrows():
        feature = feature_names.get(
            row["Feature"], row["Feature"]
        )
        print(f"- {feature} contributed towards lower late-delivery risk.")
        
# Save the explanation
os.makedirs("reports", exist_ok=True)

explanation.to_csv(
    "reports/single_order_explanation.csv",
    index=False
)

print("\nExplanation saved to reports/single_order_explanation.csv")

# Plot the top 10 feature contributions
top_features = explanation.head(10).sort_values(
    by="SHAP Contribution"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["SHAP Contribution"]
)

plt.xlabel("SHAP Contribution")
plt.ylabel("Features")
plt.title("Top 10 Features Affecting Late-Delivery Risk")
plt.tight_layout()

plt.savefig(
    "reports/single_order_shap_chart.png",
    dpi=300
)

print("SHAP chart saved to reports/single_order_shap_chart.png")

plt.show()
