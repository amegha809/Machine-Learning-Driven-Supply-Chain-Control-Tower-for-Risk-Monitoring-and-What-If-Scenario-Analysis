
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import shap
import numpy as np

# Load the trained model
model = joblib.load("models/random_forest_model.pkl")

# Load processed data to get the exact training feature columns
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

categorical_columns = [
    "Type",
    "Customer Segment",
    "Market",
    "Shipping Mode"
]


def encode_order(order_details):
    """Encode an order using the model's training feature columns."""

    encoded = pd.DataFrame(0.0, index=[0], columns=X.columns)

    # Copy numerical values
    for column, value in order_details.items():
        if column not in categorical_columns and column in encoded.columns:
            encoded.loc[0, column] = value

    # Encode categorical values
    for column in categorical_columns:
        value = order_details.get(column)

        if value is None:
            continue

        dummy_column = f"{column}_{value}"

        if dummy_column in encoded.columns:
            encoded.loc[0, dummy_column] = 1

    assert list(encoded.columns) == list(X.columns)
    assert encoded.isna().sum().sum() == 0

    return encoded[X.columns]


def predict_risk(order_details=None):
    """Predict risk using the original order and supplied changes."""

    scenario_order = order.copy()

    if order_details is not None:
        scenario_order.update(order_details)

    encoded = encode_order(scenario_order)

    prediction = model.predict(encoded)[0]
    probabilities = model.predict_proba(encoded)[0]

    risk = "HIGH" if prediction == 1 else "LOW"
    high_probability = probabilities[1] * 100

    return risk, high_probability


def explain_order_risk(order_details):
    """Generate a simple-language explanation using SHAP."""

    encoded = encode_order(order_details)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(encoded)

    # Extract contributions for late-delivery class
    if isinstance(shap_values, list):
        contributions = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        contributions = shap_values[0, :, 1]
    else:
        contributions = shap_values[0]

    # Create explanation table
    explanation = pd.DataFrame({
        "Feature": X.columns,
        "SHAP Contribution": contributions,
        "Feature Value": encoded.iloc[0].values
    })

    explanation["Absolute Contribution"] = (
        explanation["SHAP Contribution"].abs()
    )

    explanation = explanation.sort_values(
        by="Absolute Contribution",
        ascending=False
    )

    # Remove inactive categorical features
    inactive_features = [
        "Shipping Mode_Standard Class",
        "Shipping Mode_Second Class",
        "Shipping Mode_Same Day",
        "Type_DEBIT",
        "Type_PAYMENT",
        "Type_TRANSFER",
        "Customer Segment_Corporate",
        "Customer Segment_Home Office",
        "Market_Europe",
        "Market_LATAM",
        "Market_Pacific Asia",
        "Market_USCA"
    ]

    explanation = explanation[
        ~(
            explanation["Feature"].isin(inactive_features)
            & (explanation["Feature Value"] == 0)
        )
    ].copy()

    # Convert feature names into simple language
    feature_names = {
        "Shipping Mode_Standard Class": "Standard Class shipping",
        "Shipping Mode_Second Class": "Second Class shipping",
        "Shipping Mode_Same Day": "Same Day shipping",
        "Days for shipment (scheduled)": "Scheduled shipping days",
        "Type_TRANSFER": "Transfer payment",
        "Type_DEBIT": "Debit payment",
        "Type_PAYMENT": "Payment transaction",
        "Customer Segment_Corporate": "Corporate customer segment",
        "Customer Segment_Home Office": "Home Office customer segment",
        "Market_Europe": "Europe market",
        "Market_LATAM": "LATAM market",
        "Market_Pacific Asia": "Pacific Asia market",
        "Market_USCA": "USCA market",
        "Order Profit Per Order": "Profit per order",
        "Benefit per order": "Benefit per order",
        "Order Item Discount": "Order discount",
        "Order Item Discount Rate": "Discount percentage",
        "Sales per customer": "Sales per customer",
        "Order Item Total": "Total order value",
        "Product Price": "Product price",
        "Sales": "Sales amount",
        "Order Item Quantity": "Product quantity"
    }

    explanation["Feature"] = explanation["Feature"].map(
        lambda name: feature_names.get(name, name)
    )

    # Select top three factors
    risk_increasing = explanation[
        explanation["SHAP Contribution"] > 0
    ].head(3)

    risk_decreasing = explanation[
        explanation["SHAP Contribution"] < 0
    ].head(3)

    print("\n--- Why Did the Model Predict This Risk? ---")

    print("\nFactors increasing late-delivery risk:")

    if risk_increasing.empty:
        print("- No active features increased the model output.")
    else:
        for _, row in risk_increasing.iterrows():
            print(
                f"- {row['Feature']} (value: {row['Feature Value']}) "
                "contributed towards higher risk."
            )

    print("\nFactors decreasing late-delivery risk:")

    if risk_decreasing.empty:
        print("- No active features decreased the model output.")
    else:
        for _, row in risk_decreasing.iterrows():
            print(
                f"- {row['Feature']} (value: {row['Feature Value']}) "
                "contributed towards lower risk."
            )

    # Save explanation
    os.makedirs("reports", exist_ok=True)

    explanation.to_csv(
        "reports/interactive_order_explanation.csv",
        index=False
    )

    print(
        "\nInteractive SHAP explanation saved to "
        "reports/interactive_order_explanation.csv"
    )

    return explanation


def compare_order_risk(changed_details):
    """Compare original order risk with modified order risk."""

    # Predict original order risk
    original_risk, original_probability = predict_risk()

    # Predict modified order risk
    updated_risk, updated_probability = predict_risk(changed_details)

    # Generate explanation for the modified order
    explain_order_risk({
        **order,
        **changed_details
    })
    # Calculate probability difference
    difference = updated_probability - original_probability

    print("\n--- Interactive What-If Risk Analysis ---")

    print("\nOriginal Order:")
    print("Shipping Mode:", order["Shipping Mode"])
    print("Scheduled Days:", order["Days for shipment (scheduled)"])
    print("Risk:", original_risk)
    print(f"Late-Delivery Probability: {original_probability:.2f}%")

    print("\nModified Order:")
    for feature, value in changed_details.items():
        print(f"{feature}: {value}")

    print("Updated Risk:", updated_risk)
    print(f"Updated Late-Delivery Probability: {updated_probability:.2f}%")

    print("\n--- Risk Comparison ---")
    print(f"Original Probability: {original_probability:.2f}%")
    print(f"Updated Probability: {updated_probability:.2f}%")
    print(f"Probability Change: {difference:+.2f} percentage points")

    if difference > 0:
        print("Result: Late-delivery risk probability increased.")
    elif difference < 0:
        print("Result: Late-delivery risk probability decreased.")
    else:
        print("Result: Late-delivery risk probability remained unchanged.")

    return {
        "original_risk": original_risk,
        "original_probability": original_probability,
        "updated_risk": updated_risk,
        "updated_probability": updated_probability,
        "difference": difference
    }

def run_scenario_analysis():

    print("WHAT-IF SCENARIO ANALYSIS")
    print("-------------------------")

    original_risk, original_probability = predict_risk()

    print("\nOriginal Order")
    print("Shipping Mode:", order["Shipping Mode"])
    print("Scheduled Days:", order["Days for shipment (scheduled)"])
    print("Predicted Risk:", original_risk)
    print(f"Late-Delivery Probability: {original_probability:.2f}%")

    results = []

    # Test different shipping modes
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
            "Scenario Type": "Shipping Mode",
            "Shipping Mode": mode,
            "Scheduled Days": scenario["Days for shipment (scheduled)"],
            "Predicted Risk": risk,
            "Late-Delivery Probability (%)": round(probability, 2),
            "Change from Original (%)": round(
                probability - original_probability, 2
            )
        })

    # Test different scheduled shipment durations
    for days in [1, 2, 3, 4, 5, 6]:
        scenario = order.copy()
        scenario["Days for shipment (scheduled)"] = days

        risk, probability = predict_risk(scenario)

        results.append({
            "Scenario Type": "Scheduled Days",
            "Shipping Mode": scenario["Shipping Mode"],
            "Scheduled Days": days,
            "Predicted Risk": risk,
            "Late-Delivery Probability (%)": round(probability, 2),
            "Change from Original (%)": round(
                probability - original_probability, 2
            )
        })

    # Display and save results
    results_df = pd.DataFrame(results)

    print("\n--- Scenario Comparison ---")
    print(results_df.to_string(index=False))

    os.makedirs("reports", exist_ok=True)

    results_df.to_csv(
        "reports/shipping_scenario_comparison.csv",
        index=False
    )

    print("\nScenario results saved to reports/shipping_scenario_comparison.csv")

    # Plot shipping mode comparison
    shipping_results = results_df[
        results_df["Scenario Type"] == "Shipping Mode"
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(
        shipping_results["Shipping Mode"],
        shipping_results["Late-Delivery Probability (%)"]
    )

    plt.xlabel("Shipping Mode")
    plt.ylabel("Late-Delivery Probability (%)")
    plt.title("What-If Analysis: Shipping Mode Comparison")
    plt.ylim(0, 100)
    plt.tight_layout()

    plt.savefig(
        "reports/shipping_scenario_chart.png",
        dpi=300
    )

    print("Shipping chart saved to reports/shipping_scenario_chart.png")

    plt.show()

    # Plot scheduled days comparison
    days_results = results_df[
        results_df["Scenario Type"] == "Scheduled Days"
    ]

    plt.figure(figsize=(9, 5))

    plt.plot(
        days_results["Scheduled Days"],
        days_results["Late-Delivery Probability (%)"],
        marker="o"
    )

    plt.xlabel("Scheduled Shipment Days")
    plt.ylabel("Late-Delivery Probability (%)")
    plt.title("What-If Analysis: Scheduled Days Comparison")
    plt.xticks([1, 2, 3, 4, 5, 6])
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "reports/scheduled_days_scenario_chart.png",
        dpi=300
    )

    print("Scheduled days chart saved to reports/scheduled_days_scenario_chart.png")

    plt.show()



if __name__ == "__main__":

    compare_order_risk({
        "Shipping Mode": "First Class",
        "Days for shipment (scheduled)": 2,
        "Sales": 200.0,
        "Order Item Quantity": 3
    })