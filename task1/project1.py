import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# 1. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs("output", exist_ok=True)


# ============================================================
# 2. LOAD EXCEL DATASET
# ============================================================

file_path = "/Users/sahilpatil/Desktop/task1/dataset/Dataset for Data Analytics copy.xlsx"

df = pd.read_excel(file_path)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# ============================================================
# 3. BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATA INFORMATION")
print("=" * 60)

print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES BEFORE IMPUTATION")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal missing values:",
      df.isnull().sum().sum())


# Save missing value report
missing_values.to_csv(
    "output/missing_values_before.csv"
)


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("HANDLING MISSING VALUES")
print("=" * 60)


# CouponCode is categorical.
# Missing coupon means no coupon was used.
df["CouponCode"] = df["CouponCode"].fillna("No Coupon")


print("\nMissing values after imputation:")

print(df.isnull().sum())

print("\nTotal missing values after imputation:",
      df.isnull().sum().sum())


# ============================================================
# 6. CONVERT DATE COLUMN
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

print("\nDate column converted successfully.")


# ============================================================
# 7. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)


print("\nProduct-wise order count:")

print(
    df["Product"]
    .value_counts()
    .head(10)
)


print("\nPayment Method Distribution:")

print(
    df["PaymentMethod"]
    .value_counts()
)


print("\nOrder Status Distribution:")

print(
    df["OrderStatus"]
    .value_counts()
)


# ============================================================
# 8. OUTLIER DETECTION USING IQR
# ============================================================

print("\n" + "=" * 60)
print("OUTLIER DETECTION USING IQR")
print("=" * 60)


numeric_columns = [
    "Quantity",
    "UnitPrice",
    "ItemsInCart",
    "TotalPrice"
]


outlier_report = []


for column in numeric_columns:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR

    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print("\nColumn:", column)

    print("Q1:", Q1)

    print("Q3:", Q3)

    print("IQR:", IQR)

    print("Lower Limit:", lower_limit)

    print("Upper Limit:", upper_limit)

    print("Number of Outliers:",
          len(outliers))


    outlier_report.append({
        "Column": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower_Limit": lower_limit,
        "Upper_Limit": upper_limit,
        "Number_of_Outliers": len(outliers)
    })


# Convert report to DataFrame

outlier_report_df = pd.DataFrame(
    outlier_report
)


# Save report

outlier_report_df.to_csv(
    "output/outlier_report.csv",
    index=False
)


# ============================================================
# 9. NEUTRALIZE OUTLIERS USING IQR
# ============================================================

print("\n" + "=" * 60)
print("NEUTRALIZING OUTLIERS")
print("=" * 60)


for column in numeric_columns:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR

    upper_limit = Q3 + 1.5 * IQR


    # Cap values below lower limit

    df[column] = np.where(
        df[column] < lower_limit,
        lower_limit,
        df[column]
    )


    # Cap values above upper limit

    df[column] = np.where(
        df[column] > upper_limit,
        upper_limit,
        df[column]
    )


print("Outliers successfully neutralized using IQR.")


# ============================================================
# 10. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)


# ------------------------------------------------------------
# FEATURE 1: Price Per Item
# ------------------------------------------------------------

df["PricePerItem"] = (
    df["TotalPrice"] /
    df["Quantity"].replace(0, np.nan)
)


# ------------------------------------------------------------
# FEATURE 2: Cart Utilization
# ------------------------------------------------------------

df["CartUtilization"] = (
    df["Quantity"] /
    df["ItemsInCart"].replace(0, np.nan)
)


# ------------------------------------------------------------
# FEATURE 3: Coupon Used
# ------------------------------------------------------------

df["CouponUsed"] = np.where(
    df["CouponCode"] == "No Coupon",
    0,
    1
)


# ------------------------------------------------------------
# FEATURE 4: Order Year
# ------------------------------------------------------------

df["OrderYear"] = df["Date"].dt.year


# ------------------------------------------------------------
# FEATURE 5: Order Month
# ------------------------------------------------------------

df["OrderMonth"] = df["Date"].dt.month


# ------------------------------------------------------------
# FEATURE 6: Order Day of Week
# ------------------------------------------------------------

df["OrderDayOfWeek"] = (
    df["Date"].dt.dayofweek
)


# Replace infinite values

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# Fill any remaining missing numeric values

df["PricePerItem"] = (
    df["PricePerItem"].fillna(0)
)

df["CartUtilization"] = (
    df["CartUtilization"].fillna(0)
)


print("\nNew features created:")

print("1. PricePerItem")

print("2. CartUtilization")

print("3. CouponUsed")

print("4. OrderYear")

print("5. OrderMonth")

print("6. OrderDayOfWeek")


# ============================================================
# 11. CHECK FINAL DATASET
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)


print("\nFinal Shape:")

print(df.shape)


print("\nFinal Missing Values:")

print(df.isnull().sum())


# ============================================================
# 12. SAVE CLEAN DATASET
# ============================================================

output_file = (
    "output/Dataset_Cleaned_Feature_Engineered.csv"
)


df.to_csv(
    output_file,
    index=False
)


print("\nClean dataset saved to:")

print(output_file)


# ============================================================
# 13. VISUALIZATION - TOTAL PRICE
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["TotalPrice"],
    kde=True
)

plt.title("Distribution of Total Price")

plt.xlabel("Total Price")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "output/total_price_distribution.png"
)

plt.show()


# ============================================================
# 14. VISUALIZATION - PAYMENT METHOD
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="PaymentMethod"
)

plt.title("Orders by Payment Method")

plt.xlabel("Payment Method")

plt.ylabel("Number of Orders")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "output/payment_method_distribution.png"
)

plt.show()


# ============================================================
# 15. VISUALIZATION - ORDER STATUS
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="OrderStatus"
)

plt.title("Order Status Distribution")

plt.xlabel("Order Status")

plt.ylabel("Number of Orders")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "output/order_status_distribution.png"
)

plt.show()


# ============================================================
# 16. CORRELATION HEATMAP
# ============================================================

numeric_df = df.select_dtypes(
    include=np.number
)


correlation = numeric_df.corr()


plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "output/correlation_heatmap.png"
)

plt.show()


# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("PROJECT 1 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nDataset cleaned.")

print("Missing values handled.")

print("Outliers detected using IQR.")

print("Outliers neutralized.")

print("Six new features created.")

print("EDA visualizations generated.")

print("\nCheck the 'output' folder for results.")