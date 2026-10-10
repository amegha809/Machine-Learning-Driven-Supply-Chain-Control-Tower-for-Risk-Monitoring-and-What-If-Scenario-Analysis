import pandas as pd
from sklearn.model_selection import train_test_split
import os
#dataframe
df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="latin1")#latin1 encoding utf-8 cant read all the characters
print("printing head 5 rows")
print(df.head(5))
print("rows and columns")
print(df.shape)
print("column names")
print(df.columns.tolist())
print("datatypes")
print(df.dtypes)
print("missing values")
print(df.isnull().sum())
print("duplicate rows")
print(df.duplicated().sum())
#deleting inrrelevant columns
columns_to_remove = [
    "Customer Email",
    "Customer Password",
    "Product Description",
    "Product Image"
]
df = df.drop(columns=columns_to_remove)
print("Updated dataset shape:")
print(df.shape)
#checking unique values in order zipcode #609
print("Unique Order Zipcode values:")
print(df["Order Zipcode"].nunique())
#deleting zipcode column 
df = df.drop(columns=["Order Zipcode"])
print("Updated dataset shape:")
print(df.shape)
#inspecting date columns datatypes,object
print(df[["order date (DateOrders)", "shipping date (DateOrders)"]].dtypes)
#converting them to datetime
df["order date (DateOrders)"] = pd.to_datetime(
    df["order date (DateOrders)"]
)

df["shipping date (DateOrders)"] = pd.to_datetime(
    df["shipping date (DateOrders)"]
)

print(df[["order date (DateOrders)", "shipping date (DateOrders)"]].dtypes)
#calculationg actual days between shipping and ordering
df["Actual_Shipping_Days"] = (
    df["shipping date (DateOrders)"] -
    df["order date (DateOrders)"]
).dt.days
print(df["Actual_Shipping_Days"].head())
#check the new column
print(df["Actual_Shipping_Days"].describe())
#shipment delay
df["Shipping_Delay_Days"] = (
    df["Days for shipping (real)"] -
    df["Days for shipment (scheduled)"]
)
print(df["Shipping_Delay_Days"].head())
#check the new column
print(df["Shipping_Delay_Days"].describe())
#check categorical values
print(df["Late_delivery_risk"].value_counts())
print(df["Late_delivery_risk"].isnull().sum())
#check data leakage 
print("Delivery Status Values:")
print(df["Delivery Status"].value_counts())
print("Delivery Status vs Late Delivery Risk:")

print(pd.crosstab(
    df["Delivery Status"],
    df["Late_delivery_risk"]
))
#shipping related columns
print("Shipping-related columns:")

print(df[[
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Actual_Shipping_Days",
    "Shipping_Delay_Days",
    "Late_delivery_risk"
]].head())
#check target columns 
print("Target distribution:")
print(df["Late_delivery_risk"].value_counts())

print("\nMissing target values:")
print(df["Late_delivery_risk"].isnull().sum())

#selecting input features and target variable 

# Select a smaller set of useful input features
features = [
    "Type",
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Customer Segment",
    "Market",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Product Price",
    "Shipping Mode"
]

target = "Late_delivery_risk"

# Create input features and target
X = df[features].copy()
y = df[target].copy()

# One-hot encode only the selected categorical columns
X_encoded = pd.get_dummies(
    X,
    columns=[
        "Type",
        "Customer Segment",
        "Market",
        "Shipping Mode"
    ],
    drop_first=True,
    dtype="int8"
)

print("Original dataset rows:", len(df))
print("Encoded feature shape:", X_encoded.shape)
print("Target shape:", y.shape)

print("\nFirst five rows after encoding:")
print(X_encoded.head())

print("\ncheck numerical columns")
print(X_encoded.dtypes.unique())

# Prepare input features and target variable
X = X_encoded
y = df["Late_delivery_risk"]

print("Input features shape:", X.shape)
print("Target variable shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# Create folder for processed data

import os

# Create a processed folder inside the main project folder
os.makedirs("processed", exist_ok=True)

# Combine encoded features and target
processed_df = X_encoded.copy()
processed_df["Late_delivery_risk"] = y

# Save the processed dataset inside the processed folder
processed_df.to_csv(
    "processed/supply_chain_processed.csv",
    index=False
)

print("Processed dataset saved successfully!")
print("Saved dataset shape:", processed_df.shape)
