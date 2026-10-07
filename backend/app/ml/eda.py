import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "kyoto_real_estate_english.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 1: Loading processed English dataset")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: Dataset information")
print("=" * 70)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 4. TARGET VARIABLE
# ============================================================

target = "total_transaction_price"

print("\n" + "=" * 70)
print("STEP 3: Target variable analysis")
print("=" * 70)

print(df[target].describe())

print("\nMedian price:")
print(f"¥{df[target].median():,.0f}")

print("\nMean price:")
print(f"¥{df[target].mean():,.0f}")

print("\nMinimum price:")
print(f"¥{df[target].min():,.0f}")

print("\nMaximum price:")
print(f"¥{df[target].max():,.0f}")


# ============================================================
# 5. HISTOGRAM — TRANSACTION PRICE
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df[target].dropna(),
    bins=40
)

plt.xlabel("Transaction Price (JPY)")
plt.ylabel("Number of Properties")
plt.title("Kyoto House Transaction Price Distribution")

plt.ticklabel_format(
    style="plain",
    axis="x"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ============================================================
# 6. LAND AREA DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["land_area_sqm"].dropna(),
    bins=40
)

plt.xlabel("Land Area (m²)")
plt.ylabel("Number of Properties")
plt.title("Land Area Distribution")

plt.tight_layout()

plt.show()


# ============================================================
# 7. TOTAL FLOOR AREA DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["total_floor_area_sqm"].dropna(),
    bins=40
)

plt.xlabel("Total Floor Area (m²)")
plt.ylabel("Number of Properties")
plt.title("Total Floor Area Distribution")

plt.tight_layout()

plt.show()


# ============================================================
# 8. PRICE VS LAND AREA
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["land_area_sqm"],
    df[target],
    alpha=0.6
)

plt.xlabel("Land Area (m²)")
plt.ylabel("Transaction Price (JPY)")
plt.title("Transaction Price vs Land Area")

plt.ticklabel_format(
    style="plain",
    axis="y"
)

plt.tight_layout()

plt.show()


# ============================================================
# 9. PRICE VS FLOOR AREA
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["total_floor_area_sqm"],
    df[target],
    alpha=0.6
)

plt.xlabel("Total Floor Area (m²)")
plt.ylabel("Transaction Price (JPY)")
plt.title("Transaction Price vs Total Floor Area")

plt.ticklabel_format(
    style="plain",
    axis="y"
)

plt.tight_layout()

plt.show()


# ============================================================
# 10. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: Correlation with transaction price")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include="number"
).columns

correlations = (
    df[numeric_columns]
    .corr()[target]
    .sort_values(ascending=False)
)

print(correlations)


# ============================================================
# 11. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 8))

correlation_matrix = df[numeric_columns].corr()

plt.imshow(
    correlation_matrix,
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Numeric Feature Correlation Matrix")

plt.tight_layout()

plt.show()


# ============================================================
# 12. CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: Categorical feature analysis")
print("=" * 70)

categorical_columns = [
    "building_structure",
    "property_use",
    "front_road_direction",
    "front_road_type",
    "city_planning"
]

for column in categorical_columns:

    print("\n" + "-" * 60)
    print(column)

    print(
        df[column]
        .value_counts(dropna=False)
        .head(15)
    )


# ============================================================
# 13. BUILDING STRUCTURE VS PRICE
# ============================================================

if "building_structure" in df.columns:

    structure_prices = (
        df.groupby("building_structure")[target]
        .median()
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 70)
    print("Median price by building structure")
    print("=" * 70)

    print(structure_prices)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

