
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("models/random_forest_model.pkl")

# Load processed data to get the expected feature columns
df = pd.read_csv("processed/supply_chain_processed.csv")
X = df.drop(columns=["Late_delivery_risk"])

# Create a sample order
order = {
    "Days for shipment (scheduled)": 4,
    "Benefit per order": 25.0,
    "Sales per customer": 150.0,
    "Order Item Discount": 10.0,
    "Order Item Discount Rate": 0.10,
    "Order Item Quantity": 2,
    "Sales": 150.0,
    "Order Item Total": 140.0,
    "Order Profit Per Order": 25.0,
    "Product Price": 75.0,
    "Type": "DEBIT",
    "Customer Segment": "Consumer",
    "Market": "Africa",
    "Shipping Mode": "Standard Class"
}

def predict_risk(order_details):
    order_df = pd.DataFrame([order_details])

    
# Use the same encoding reference as the training dataset

    encoded = pd.get_dummies(
        order_df,
        columns=[
            "Type",
            "Customer Segment",
            "Market",
            "Shipping Mode"
        ],
        drop_first=True,
        dtype="int8"
    )

    # Align columns with the model's training features
    encoded = encoded.reindex(columns=X.columns, fill_value=0)

    # Correct the shipping mode columns explicitly
    shipping_columns = [
        "Shipping Mode_Same Day",
        "Shipping Mode_Second Class",
        "Shipping Mode_Standard Class"
    ]

    for col in shipping_columns:
        if col in encoded.columns:
            encoded[col] = 0

    if order_details["Shipping Mode"] == "Same Day":
        encoded["Shipping Mode_Same Day"] = 1
    elif order_details["Shipping Mode"] == "Second Class":
        encoded["Shipping Mode_Second Class"] = 1
    elif order_details["Shipping Mode"] == "Standard Class":
        encoded["Shipping Mode_Standard Class"] = 1

    print(
        "Encoded shipping features:",
        encoded.filter(like="Shipping Mode_").to_dict(orient="records")[0]
    )

    prediction = model.predict(encoded)[0]
    probabilities = model.predict_proba(encoded)[0]

    risk = "HIGH" if prediction == 1 else "LOW"
    high_probability = probabilities[1] * 100

    return risk, high_probability



# Compare shipping options
print("WHAT-IF SHIPPING SCENARIO ANALYSIS")
print("----------------------------------")


results = []

for mode in [
    "Standard Class",
    "Second Class",
    "First Class",
    "Same Day"
]:
    scenario = order.copy()
    scenario["Shipping Mode"] = mode

    risk, probability = predict_risk(scenario)

    results.append({
        "Shipping Mode": mode,
        "Predicted Risk": risk,
        "High-Risk Probability (%)": round(probability, 2)
    })

results_df = pd.DataFrame(results)

print("\n--- Scenario Comparison ---")
print(results_df.to_string(index=False))

os.makedirs("reports", exist_ok=True)
results_df.to_csv(
    "reports/shipping_scenario_comparison.csv",
    index=False
)

print("\nScenario comparison saved successfully!")


# Plot scenario comparison
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Shipping Mode"],
    results_df["High-Risk Probability (%)"]
)

plt.xlabel("Shipping Mode")
plt.ylabel("High-Risk Probability (%)")
plt.title("What-If Analysis: Delivery Risk by Shipping Mode")
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig(
    "reports/shipping_scenario_chart.png",
    dpi=300
)

print("Scenario chart saved to reports/shipping_scenario_chart.png")

plt.show()
