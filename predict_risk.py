
import pandas as pd
import joblib

# Load the trained Random Forest model
model = joblib.load("models/random_forest_model.pkl")

# Load processed data to obtain the expected feature names
df = pd.read_csv("processed/supply_chain_processed.csv")


original_df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="latin1")

print("Discount Rate Range:")
print(original_df["Order Item Discount Rate"].describe())

print("\nOrder Item Quantity Range:")
print(original_df["Order Item Quantity"].describe())

X = df.drop(columns=["Late_delivery_risk"])

print("Supply Chain Risk Prediction System")
print("-----------------------------------")
print("Model loaded successfully!")
print("Number of input features:", len(X.columns))
print("\nExpected input features:")
print(list(X.columns))


# # Prepare a sample order for prediction

# sample_order = {
#     "Days for shipment (scheduled)": 4,
#     "Benefit per order": 25.0,
#     "Sales per customer": 150.0,
#     "Order Item Discount": 10.0,
#     "Order Item Discount Rate": 0.1,
#     "Order Item Quantity": 2,
#     "Sales": 150.0,
#     "Order Item Total": 140.0,
#     "Order Profit Per Order": 25.0,
#     "Product Price": 75.0,
#     "Type": "DEBIT",
#     "Customer Segment": "Consumer",
#     "Market": "Pacific Asia",
#     "Shipping Mode": "Standard Class"
# }

# # Convert the sample into a DataFrame
# sample_df = pd.DataFrame([sample_order])

# # Apply the same one-hot encoding used during training
# sample_encoded = pd.get_dummies(
#     sample_df,
#     columns=[
#         "Type",
#         "Customer Segment",
#         "Market",
#         "Shipping Mode"
#     ],
#     drop_first=True,
#     dtype="int8"
# )

# # Match the training feature columns
# sample_encoded = sample_encoded.reindex(
#     columns=X.columns,
#     fill_value=0
# )

# print("\nSample order prepared successfully!")
# print("Input shape:", sample_encoded.shape)
# print("Expected shape: (1,", len(X.columns), ")")


# # Predict late-delivery risk
# prediction = model.predict(sample_encoded)[0]

# print("\n--- Prediction Result ---")

# if prediction == 1:
#     print("Predicted Risk: HIGH")
#     print("The order is predicted to have late-delivery risk.")
# else:
#     print("Predicted Risk: LOW")
#     print("The order is predicted to have no late-delivery risk.")


# # Calculate prediction probabilities
# probabilities = model.predict_proba(sample_encoded)[0]

# low_risk_probability = probabilities[0] * 100
# high_risk_probability = probabilities[1] * 100

# print("\n--- Risk Probability ---")
# print(f"Low Risk Probability: {low_risk_probability:.2f}%")
# print(f"High Risk Probability: {high_risk_probability:.2f}%")


# # Save prediction report

# prediction_report = sample_df.copy()

# prediction_report["Predicted_Risk"] = (
#     "HIGH" if prediction == 1 else "LOW"
# )

# prediction_report["Low_Risk_Probability"] = round(
#     low_risk_probability, 2
# )

# prediction_report["High_Risk_Probability"] = round(
#     high_risk_probability, 2
# )

# import os
# os.makedirs("reports", exist_ok=True)

# prediction_report.to_csv(
#     "reports/sample_risk_prediction.csv",
#     index=False
# )

# print("\nPrediction report saved successfully!")
# print("Location: reports/sample_risk_prediction.csv")


# Take order details from the user

print("\nEnter the order details:")

scheduled_days = int(
    input("Days for shipment (scheduled): ")
)

benefit = float(input("Benefit per order: "))
sales_customer = float(input("Sales per customer: "))
discount = float(input("Order Item Discount: "))
discount_rate = float(input("Order Item Discount Rate: "))
quantity = int(input("Order Item Quantity: "))
sales = float(input("Sales: "))
item_total = float(input("Order Item Total: "))
profit = float(input("Order Profit Per Order: "))
price = float(input("Product Price: "))

order_type = input("Type (DEBIT, TRANSFER, PAYMENT): ").strip()
segment = input("Customer Segment (Consumer, Corporate, Home Office): ").strip()
market = input("Market (Africa, Europe, LATAM, Pacific Asia, USCA): ").strip()
shipping_mode = input(
    "Shipping Mode (Standard Class, Second Class, First Class, Same Day): "
).strip()

sample_order = {
    "Days for shipment (scheduled)": scheduled_days,
    "Benefit per order": benefit,
    "Sales per customer": sales_customer,
    "Order Item Discount": discount,
    "Order Item Discount Rate": discount_rate,
    "Order Item Quantity": quantity,
    "Sales": sales,
    "Order Item Total": item_total,
    "Order Profit Per Order": profit,
    "Product Price": price,
    "Type": order_type,
    "Customer Segment": segment,
    "Market": market,
    "Shipping Mode": shipping_mode
}

sample_df = pd.DataFrame([sample_order])

sample_encoded = pd.get_dummies(
    sample_df,
    columns=[
        "Type",
        "Customer Segment",
        "Market",
        "Shipping Mode"
    ],
    drop_first=True,
    dtype="int8"
)

sample_encoded = sample_encoded.reindex(
    columns=X.columns,
    fill_value=0
)


# Validate numeric inputs
if not 0 <= discount_rate <= 0.25:
    print("Invalid Discount Rate. Enter a value between 0 and 0.25.")
    exit()

if not 1 <= quantity <= 5:
    print("Invalid Quantity. Enter a whole number between 1 and 5.")
    exit()

if scheduled_days <= 0:
    print("Scheduled shipping days must be positive.")
    exit()

if discount < 0 or sales < 0 or item_total < 0 or price < 0:
    print("Discount, Sales, Item Total, and Product Price cannot be negative.")
    exit()

print("\nOrder details entered successfully!")


# Validate categorical inputs
valid_types = ["DEBIT", "TRANSFER", "PAYMENT"]
valid_segments = ["Consumer", "Corporate", "Home Office"]
valid_markets = ["Africa", "Europe", "LATAM", "Pacific Asia", "USCA"]
valid_shipping_modes = [
    "Standard Class", "Second Class", "First Class", "Same Day"
]

if order_type not in valid_types:
    print("Invalid Type. Please enter a valid order type.")
    exit()

if segment not in valid_segments:
    print("Invalid Customer Segment.")
    exit()

if market not in valid_markets:
    print("Invalid Market. Please check the spelling.")
    exit()

if shipping_mode not in valid_shipping_modes:
    print("Invalid Shipping Mode.")
    exit()


# Predict late-delivery risk
prediction = model.predict(sample_encoded)[0]

# Calculate risk probabilities
probabilities = model.predict_proba(sample_encoded)[0]

low_risk_probability = probabilities[0] * 100
high_risk_probability = probabilities[1] * 100

print("\n--- Prediction Result ---")


if prediction == 1:
    print("Predicted Risk: HIGH")
    print("Recommendation: Review this order for possible delivery delays.")
    print("Suggested Action: Check the shipping schedule and order status.")
else:
    print("Predicted Risk: LOW")
    print("Recommendation: No high late-delivery risk was detected by the model.")
    print("Suggested Action: Continue monitoring the order.")


print(f"Low Risk Probability: {low_risk_probability:.2f}%")
print(f"High Risk Probability: {high_risk_probability:.2f}%")

# Save the prediction report
import os

prediction_report = sample_df.copy()

prediction_report["Predicted_Risk"] = (
    "HIGH" if prediction == 1 else "LOW"
)
prediction_report["Low_Risk_Probability"] = round(
    low_risk_probability, 2
)
prediction_report["High_Risk_Probability"] = round(
    high_risk_probability, 2
)

os.makedirs("reports", exist_ok=True)

prediction_report.to_csv(
    "reports/sample_risk_prediction_2.csv",
    index=False
)

print("\nPrediction report saved successfully!")
print("Location: reports/sample_risk_prediction_2.csv")
