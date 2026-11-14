
import pandas as pd
import joblib
import os

# Load the trained model
model = joblib.load("models/random_forest_model.pkl")

# Load the processed dataset
df = pd.read_csv("processed/supply_chain_processed.csv")

# Separate input features and actual target
X = df.drop(columns=["Late_delivery_risk"])
y_actual = df["Late_delivery_risk"]

# Predict risk probabilities
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

# Create the risk report with original order details
risk_report = df.copy()

# Add prediction information
risk_report["Actual_Risk"] = y_actual
risk_report["Predicted_Risk"] = predictions
risk_report["High_Risk_Probability"] = (
    probabilities * 100
).round(2)

# Assign readable labels
risk_report["Predicted_Risk"] = (
    risk_report["Predicted_Risk"].map({
        0: "LOW",
        1: "HIGH"
    })
)

# Separate HIGH and LOW risk orders
high_risk_orders = risk_report[
    risk_report["Predicted_Risk"] == "HIGH"
].sort_values(
    by="High_Risk_Probability",
    ascending=False
).head(100)

low_risk_orders = risk_report[
    risk_report["Predicted_Risk"] == "LOW"
].sort_values(
    by="High_Risk_Probability",
    ascending=True
).head(100)

# Combine both categories
final_report = pd.concat(
    [high_risk_orders, low_risk_orders]
)

# Save the report
os.makedirs("reports", exist_ok=True)

final_report.to_csv(
    "reports/high_risk_orders.csv",
    index=False
)

print("RISK MONITORING REPORT")
print("----------------------")
print("Total orders analysed:", len(risk_report))
print("HIGH-risk orders included:", len(high_risk_orders))
print("LOW-risk orders included:", len(low_risk_orders))
print("Total orders in report:", len(final_report))

print("\nPrediction distribution:")
print(final_report["Predicted_Risk"].value_counts())

print("\nReport columns:")
print(final_report.columns.tolist())

print("\nSample HIGH-risk orders:")
print(high_risk_orders.head(5).to_string(index=False))

print("\nSample LOW-risk orders:")
print(low_risk_orders.head(5).to_string(index=False))

print("\nReport saved to reports/high_risk_orders.csv")
