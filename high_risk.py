
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

# Create the risk report
risk_report = pd.DataFrame({
    "Actual_Risk": y_actual,
    "Predicted_Risk": predictions,
    "High_Risk_Probability": (probabilities * 100).round(2)
})

# Assign readable labels
risk_report["Predicted_Risk"] = risk_report["Predicted_Risk"].map({
    0: "LOW",
    1: "HIGH"
})

# Rank orders by predicted high-risk probability
risk_report = risk_report.sort_values(
    by="High_Risk_Probability",
    ascending=False
)

# Keep the 100 highest-risk orders
top_risk_orders = risk_report.head(100).copy()

# Save the report
os.makedirs("reports", exist_ok=True)

top_risk_orders.to_csv(
    "reports/high_risk_orders.csv",
    index=False
)

print("HIGH-RISK ORDER MONITORING")
print("--------------------------")
print("Total orders analysed:", len(risk_report))
print("Top orders included in report:", len(top_risk_orders))
print("\nTop 10 highest-risk orders:")
print(top_risk_orders.head(10).to_string(index=False))

print("\nReport saved to reports/high_risk_orders.csv")
