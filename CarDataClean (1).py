import numpy as np
import pandas as pd
from pathlib import Path

# Load the messy car dataset
# Keep this .py file and the CSV file in the same folder.
input_file = "car_data_messy_for_cleaning.csv"
output_file = "cleaned_car_data.csv"

# If your downloaded CSV has (1) in its name, use that file instead.
if not Path(input_file).exists() and Path("car_data_messy_for_cleaning(1).csv").exists():
    input_file = "car_data_messy_for_cleaning(1).csv"

df = pd.read_csv(input_file)
print("First 10 rows of messy data:")
print(df.head(10))

# 1. Remove duplicate rows
df = df.drop_duplicates(keep="first").copy()

# 2. Clean Car_Name (remove extra spaces and standardize text)
df["Car_Name"] = df["Car_Name"].fillna("Unknown")
df["Car_Name"] = df["Car_Name"].astype(str).str.strip().str.title()
df.loc[df["Car_Name"].isin(["", "Nan", "None"]), "Car_Name"] = "Unknown"

# 3. Clean Year and keep only years from 2005 to 2025
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df.loc[(df["Year"] < 2005) | (df["Year"] > 2025), "Year"] = np.nan

median_year = df["Year"].median()
df["Year"] = df["Year"].fillna(median_year).round().astype(int)

# 4. Clean Selling_Price, Present_Price and Kms_Driven
# Remove currency symbols, commas and extra spaces before conversion.
numeric_columns = ["Selling_Price", "Present_Price", "Kms_Driven"]

for column in numeric_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("nan", "", regex=False)
    )
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Prices and kilometres cannot be negative.
for column in ["Selling_Price", "Present_Price", "Kms_Driven"]:
    df.loc[df[column] < 0, column] = np.nan

# Fill missing numeric values with each column's median.
# For a prediction project, rows with missing Selling_Price (the target)
# can instead be removed rather than filled with a guessed price.
for column in numeric_columns:
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)

# 5. Clean and standardize Fuel_Type
df["Fuel_Type"] = df["Fuel_Type"].astype("string").str.strip().str.upper()
fuel_map = {
    "PETROL": "Petrol",
    "DIESEL": "Diesel",
    "CNG": "CNG"
}
df["Fuel_Type"] = df["Fuel_Type"].map(fuel_map).fillna("Unknown")

# 6. Clean and standardize Seller_Type
df["Seller_Type"] = df["Seller_Type"].astype("string").str.strip().str.upper()
seller_map = {
    "DEALER": "Dealer",
    "INDIVIDUAL": "Individual"
}
df["Seller_Type"] = df["Seller_Type"].map(seller_map).fillna("Unknown")

# 7. Clean and standardize Transmission
df["Transmission"] = df["Transmission"].astype("string").str.strip().str.upper()
transmission_map = {
    "MANUAL": "Manual",
    "AUTOMATIC": "Automatic"
}
df["Transmission"] = df["Transmission"].map(transmission_map).fillna("Unknown")

# 8. Clean Owner (number of previous owners)
df["Owner"] = pd.to_numeric(df["Owner"], errors="coerce")
df.loc[(df["Owner"] < 0) | (df["Owner"] > 3), "Owner"] = np.nan

median_owner = df["Owner"].median()
if pd.isna(median_owner):
    median_owner = 0
df["Owner"] = df["Owner"].fillna(median_owner).round().astype(int)

# 9. Save the cleaned dataset
df.to_csv(output_file, index=False)

print("\nCleaning completed successfully!")
print("Original messy rows may include duplicates and invalid values.")
print("Cleaned dataset shape:", df.shape)
print("Remaining missing values:")
print(df.isnull().sum())
print("\nCleaned data saved as:", output_file)
print("\nFirst 10 rows of cleaned data:")
print(df.head(10))
