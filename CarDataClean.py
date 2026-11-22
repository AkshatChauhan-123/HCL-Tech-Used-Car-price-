import numpy as np
import pandas as pd

# Load the messy car dataset
df = pd.read_csv("car_data_messy_for_cleaning.csv")

print("Original data:")
print(df.head(10))
print("\nOriginal shape:", df.shape)

# 1. Clean column names (remove extra spaces)
df.columns = df.columns.astype(str).str.strip()

# 2. Clean Car Name (remove extra spaces and standardize capitalization)
df["Car Name"] = (
    df["Car Name"]
    .astype("string")
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.title()
)
df["Car Name"] = df["Car Name"].replace("", pd.NA).fillna("Unknown")

# 3. Clean Year
# Convert values such as "09" to 2009 and reject impossible years.
year = pd.to_numeric(df["year"], errors="coerce")
year = year.where(year >= 1000, year + 2000)
year = year.where(year.between(1950, pd.Timestamp.today().year))
df["year"] = year

# 4. Clean Selling Price and Present Price
# These price columns are assumed to be in lakh, as in the source dataset.
def clean_price(series):
    values = (
        series.astype("string")
        .str.strip()
        .str.replace(r"(?i)rs\.?|₹|,", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )
    values = pd.to_numeric(values, errors="coerce")

    # Negative, zero, and extremely large values are treated as invalid.
    values = values.where(values.gt(0) & values.le(100))
    return values

df["Selling Price"] = clean_price(df["Selling Price"])
df["PRESENT_PRICE"] = clean_price(df["PRESENT_PRICE"])

# 5. Clean kilometers driven
df["kms driven"] = pd.to_numeric(
    df["kms driven"].astype("string").str.strip()
      .str.replace(",", "", regex=False),
    errors="coerce"
)
df.loc[~df["kms driven"].between(0, 1_000_000), "kms driven"] = np.nan

# 6. Clean Fuel Type
fuel = df["Fuel-Type"].astype("string").str.strip().str.upper()
fuel_map = {
    "PETROL": "Petrol", "PERTOL": "Petrol", "PETRL": "Petrol", "P": "Petrol",
    "DIESEL": "Diesel", "D": "Diesel",
    "CNG": "CNG"
}
df["Fuel-Type"] = fuel.map(fuel_map).fillna("Unknown")

# 7. Clean Seller Type
seller = df["seller type"].astype("string").str.strip().str.upper()
seller_map = {
    "DEALER": "Dealer", "DEALERSHIP": "Dealer", "DELER": "Dealer",
    "INDIVIDUAL": "Individual", "INDIVIDUL": "Individual",
    "INDIV": "Individual", "OWNER": "Individual"
}
df["seller type"] = seller.map(seller_map).fillna("Unknown")

# 8. Clean Transmission
transmission = df["Transmission"].astype("string").str.strip().str.upper()
transmission_map = {
    "MANUAL": "Manual", "MAN": "Manual", "MANUL": "Manual", "M": "Manual",
    "AUTOMATIC": "Automatic", "AUTO": "Automatic",
    "AUTOMATC": "Automatic", "A": "Automatic"
}
df["Transmission"] = transmission.map(transmission_map).fillna("Unknown")

# 9. Clean Owner count
owner_text = (
    df["OWNER"].astype("string").str.strip().str.lower()
    .replace({"zero": "0", "first": "1", "one": "1"})
)
df["OWNER"] = pd.to_numeric(owner_text, errors="coerce")
df.loc[~df["OWNER"].between(0, 4), "OWNER"] = np.nan

# 10. Fill missing numeric values with the median
numeric_columns = [
    "year", "Selling Price", "PRESENT_PRICE", "kms driven", "OWNER"
]
for column in numeric_columns:
    median_value = df[column].median()
    if pd.notna(median_value):
        df[column] = df[column].fillna(median_value)

# Use integer types for year, kilometers and owner count
for column in ["year", "kms driven", "OWNER"]:
    if df[column].notna().all():
        df[column] = df[column].round().astype("int64")

# 11. Remove completely duplicated rows
df = df.drop_duplicates()

# 12. Save the cleaned dataset
df.to_csv("cleaned_car_data.csv", index=False)

print("\nCleaned data:")
print(df.head(10))
print("\nCleaned shape:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nCleaned file saved as cleaned_car_data.csv")
