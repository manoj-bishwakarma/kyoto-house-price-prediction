import pandas as pd
import re
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "kyoto_real_estate.csv"
)

PROCESSED_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "kyoto_real_estate_english.csv"
)

PROCESSED_DATA_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD ORIGINAL JAPANESE DATA
# ============================================================

print("=" * 70)
print("STEP 1: Loading original Japanese dataset")
print("=" * 70)

df = pd.read_csv(
    RAW_DATA_PATH,
    encoding="cp932"
)

print(f"Original shape: {df.shape}")


# ============================================================
# 3. SELECT LAND + BUILDING TRANSACTIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: Selecting land + building properties")
print("=" * 70)

df = df[
    df["種類"] == "宅地(土地と建物)"
].copy()

print(f"Rows after property-type filtering: {len(df)}")


# ============================================================
# 4. SELECT FEATURES
# ============================================================

columns_to_keep = [
    "種類",
    "面積（㎡）",
    "延床面積（㎡）",
    "建築年",
    "最寄駅：距離（分）",
    "間口",
    "前面道路：幅員（ｍ）",
    "建ぺい率（％）",
    "容積率（％）",
    "建物の構造",
    "用途",
    "今後の利用目的",
    "前面道路：方位",
    "前面道路：種類",
    "都市計画",
    "取引時期",
    "取引価格（総額）"
]

df = df[columns_to_keep].copy()

# ============================================================
# 5. RENAME COLUMNS TO ENGLISH
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: Translating column names")
print("=" * 70)

column_translation = {

    "種類":
        "property_type",

    "面積（㎡）":
        "land_area_sqm",

    "延床面積（㎡）":
        "total_floor_area_sqm",

    "建築年":
        "year_built",

    "最寄駅：距離（分）":
        "station_distance_minutes",

    "間口":
        "frontage_m",

    "前面道路：幅員（ｍ）":
        "front_road_width_m",

    "建ぺい率（％）":
        "building_coverage_ratio",

    "容積率（％）":
        "floor_area_ratio",

    "建物の構造":
        "building_structure",

    "用途":
        "property_use",

    "今後の利用目的":
        "future_use",

    "前面道路：方位":
        "front_road_direction",

    "前面道路：種類":
        "front_road_type",

    "都市計画":
        "city_planning",

    "取引時期":
        "transaction_period",

    "取引価格（総額）":
        "total_transaction_price"
}

df.rename(
    columns=column_translation,
    inplace=True
)

# ============================================================
# 6. CONVERT NUMERIC COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: Converting numeric columns")
print("=" * 70)

numeric_columns = [
    "land_area_sqm",
    "total_floor_area_sqm",
    "station_distance_minutes",
    "frontage_m",
    "front_road_width_m",
    "building_coverage_ratio",
    "floor_area_ratio",
    "total_transaction_price"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 7. CONVERT JAPANESE ERA YEAR → WESTERN YEAR
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: Converting Japanese construction years")
print("=" * 70)


def convert_japanese_year(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    match = re.search(r"(\d{4})", value)

    if (
        match
        and not any(
            era in value
            for era in ["昭和", "平成", "令和"]
        )
    ):
        year = int(match.group(1))

        if 1800 <= year <= 2100:
            return year

    match = re.search(
        r"昭和\s*(\d+|元)",
        value
    )

    if match:

        era_year = (
            1
            if match.group(1) == "元"
            else int(match.group(1))
        )

        return 1925 + era_year

    match = re.search(
        r"平成\s*(\d+|元)",
        value
    )

    if match:

        era_year = (
            1
            if match.group(1) == "元"
            else int(match.group(1))
        )

        return 1988 + era_year

    match = re.search(
        r"令和\s*(\d+|元)",
        value
    )

    if match:

        era_year = (
            1
            if match.group(1) == "元"
            else int(match.group(1))
        )

        return 2018 + era_year

    return None

df["year_built"] = (
    df["year_built"]
    .apply(convert_japanese_year)
)

df["year_built"] = pd.to_numeric(
    df["year_built"],
    errors="coerce"
)

print(
    "Year conversion completed."
)

print(
    f"Available year_built values: "
    f"{df['year_built'].notna().sum()}"
)


# ============================================================
# 8. PROPERTY TYPE
# ============================================================

property_type_map = {

    "宅地(土地と建物)":
        "land_and_building",

    "宅地(土地)":
        "land_only"
}

df["property_type"] = (
    df["property_type"]
    .map(property_type_map)
    .fillna(df["property_type"])
)
# ============================================================
# 9. BUILDING STRUCTURE
# ============================================================

building_structure_map = {

    "木造":
        "wooden",

    "軽量鉄骨造":
        "light_steel_frame",

    "鉄骨造":
        "steel_frame",

    "ＲＣ":
        "reinforced_concrete",

    "RC":
        "reinforced_concrete",

    "鉄筋コンクリート造":
        "reinforced_concrete",

    "ＳＲＣ":
        "steel_reinforced_concrete",

    "SRC":
        "steel_reinforced_concrete",

    "鉄骨鉄筋コンクリート造":
        "steel_reinforced_concrete",

    "ブロック造":
        "block_construction",

    "コンクリートブロック造":
        "concrete_block",

    "その他":
        "other"
}


def translate_building_structure(value):

    if pd.isna(value):
        return value

    value = str(value).strip()

    # Exact match
    if value in building_structure_map:
        return building_structure_map[value]

    # Mixed construction
    if "鉄骨造" in value and "木造" in value:
        return "steel_frame_and_wooden"

    if "木造" in value:
        return "wooden"

    if "軽量鉄骨造" in value:
        return "light_steel_frame"

    if "鉄筋コンクリート" in value:
        return "reinforced_concrete"

    if "ＳＲＣ" in value or "SRC" in value:
        return "steel_reinforced_concrete"

    if "鉄骨造" in value:
        return "steel_frame"

    return value


df["building_structure"] = (
    df["building_structure"]
    .apply(translate_building_structure)
)


# ============================================================
# 10. ROAD DIRECTION
# ============================================================

road_direction_map = {

    "北":
        "north",

    "南":
        "south",

    "東":
        "east",

    "西":
        "west",

    "北東":
        "northeast",

    "北西":
        "northwest",

    "南東":
        "southeast",

    "南西":
        "southwest"
}

df["front_road_direction"] = (
    df["front_road_direction"]
    .map(road_direction_map)
    .fillna(df["front_road_direction"])
)

# ============================================================
# 11. ROAD TYPE
# ============================================================

road_type_map = {

    "国道":
        "national_road",

    "府道":
        "prefectural_road",

    "都道府県道":
        "prefectural_road",

    "市道":
        "municipal_road",

    "区道":
        "ward_road",

    "私道":
        "private_road",

    "公道":
        "public_road",

    "道路":
        "road",

    "区画街路":
        "district_road",

    "その他":
        "other"
}

df["front_road_type"] = (
    df["front_road_type"]
    .map(road_type_map)
    .fillna(df["front_road_type"])
)


# ============================================================
# 12. CITY PLANNING / ZONING
# ============================================================

city_planning_map = {

    "１低住専":
        "category_1_low_rise_residential",

    "1低住専":
        "category_1_low_rise_residential",

    "２低住専":
        "category_2_low_rise_residential",

    "2低住専":
        "category_2_low_rise_residential",

    "１中住専":
        "category_1_mid_high_rise_residential",

    "1中住専":
        "category_1_mid_high_rise_residential",

    "２中住専":
        "category_2_mid_high_rise_residential",

    "2中住専":
        "category_2_mid_high_rise_residential",

    "１種住居":
        "category_1_residential",

    "1種住居":
        "category_1_residential",

    "２種住居":
        "category_2_residential",

    "2種住居":
        "category_2_residential",

    "準住居":
        "quasi_residential",

    "近隣商業":
        "neighborhood_commercial",

    "商業":
        "commercial",

    "商業地域":
        "commercial",

    "準工業":
        "quasi_industrial",

    "工業":
        "industrial",

    "工業地域":
        "industrial",

    "工業専用":
        "industrial_exclusive",

    "調整区域":
        "urbanization_control_area",

    "市街化区域":
        "urbanization_area",

    "市街化調整区域":
        "urbanization_control_area",

    "非線引都市計画区域":
        "non_delineated_urban_planning_area",

    "都市計画区域外":
        "outside_urban_planning_area"
}

df["city_planning"] = (
    df["city_planning"]
    .map(city_planning_map)
    .fillna(df["city_planning"])
)

property_use_map = {

    "住宅":
        "residential",

    "共同住宅":
        "apartment",

    "店舗":
        "shop",

    "事務所":
        "office",

    "工場":
        "factory",

    "倉庫":
        "warehouse",

    "旅館":
        "hotel_or_inn",

    "駐車場":
        "parking",

    "作業場":
        "workshop",

    "その他":
        "other"
}

def translate_multi_category(value, mapping):

    if pd.isna(value):
        return value

    value = str(value).strip()

    if value == "":
        return value

    # Exact match
    if value in mapping:
        return mapping[value]

    # Multiple categories separated by Japanese comma
    parts = value.split("、")

    translated_parts = []

    for part in parts:

        part = part.strip()

        translated_parts.append(
            mapping.get(part, part)
        )

    return "_and_".join(translated_parts)


df["property_use"] = (
    df["property_use"]
    .apply(
        lambda x:
        translate_multi_category(
            x,
            property_use_map
        )
    )
)

# ============================================================
# 14. FUTURE USE
# ============================================================

future_use_map = {

    "住宅":
        "residential",

    "共同住宅":
        "apartment",

    "店舗":
        "shop",

    "事務所":
        "office",

    "工場":
        "factory",

    "倉庫":
        "warehouse",

    "駐車場":
        "parking",

    "作業場":
        "workshop",

    "その他":
        "other"
}

df["future_use"] = (
    df["future_use"]
    .apply(
        lambda x:
        translate_multi_category(
            x,
            future_use_map
        )
    )
)


# ============================================================
# 15. TRANSACTION PERIOD
# ============================================================

def translate_transaction_period(value):

    if pd.isna(value):
        return value

    value = str(value).strip()

    replacements = {

        "第１四半期": "Q1",
        "第２四半期": "Q2",
        "第３四半期": "Q3",
        "第４四半期": "Q4",

        "第1四半期": "Q1",
        "第2四半期": "Q2",
        "第3四半期": "Q3",
        "第4四半期": "Q4",

        "四半期": ""
    }

    for japanese, english in replacements.items():

        value = value.replace(
            japanese,
            english
        )


    value = value.replace(
        "年",
        " "
    )

    value = " ".join(
        value.split()
    )

    return value

df["transaction_period"] = (
    df["transaction_period"]
    .apply(translate_transaction_period)
)
# ============================================================
# 16. CLEAN TEXT COLUMNS
# ============================================================

text_columns = [

    "property_type",
    "building_structure",
    "property_use",
    "future_use",
    "front_road_direction",
    "front_road_type",
    "city_planning",
    "transaction_period"
]

for column in text_columns:

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
    )

# ============================================================
# 17. REMOVE DUPLICATES AGAIN
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: Removing duplicate processed records")
print("=" * 70)

duplicates = df.duplicated().sum()

print(
    f"Duplicate processed rows found: {duplicates}"
)

df = df.drop_duplicates().copy()

print(
    f"Rows after duplicate removal: {len(df)}"
)


# ============================================================
# 18. CHECK REMAINING JAPANESE VALUES
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: Checking categorical translations")
print("=" * 70)

categorical_columns = [

    "building_structure",
    "property_use",
    "future_use",
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
        .to_string()
    )

# ============================================================
# 19. SAVE PROCESSED DATASET
# ============================================================

print("\n" + "=" * 70)
print("STEP 8: Saving English dataset")
print("=" * 70)

df.to_csv(
    PROCESSED_DATA_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"Saved to:\n{PROCESSED_DATA_PATH}"
)
# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(
    f"Rows:    {df.shape[0]}"
)

print(
    f"Columns: {df.shape[1]}"
)

print("\nColumns:")

for column in df.columns:
    print(f"  - {column}")


# ============================================================
# 21. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_summary = pd.DataFrame({

    "missing_count":
        df.isnull().sum(),

    "missing_percentage":
        (df.isnull().mean() * 100).round(2)
})

print(
    missing_summary[
        missing_summary["missing_count"] > 0
    ]
    .sort_values(
        by="missing_count",
        ascending=False
    )
)

# ============================================================
# 22. TARGET STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("TARGET STATISTICS")
print("=" * 70)

target = "total_transaction_price"

print(
    df[target].describe()
)

print(
    f"\nMedian price: "
    f"¥{df[target].median():,.0f}"
)

print(
    f"Mean price: "
    f"¥{df[target].mean():,.0f}"
)

print(
    f"Minimum price: "
    f"¥{df[target].min():,.0f}"
)

print(
    f"Maximum price: "
    f"¥{df[target].max():,.0f}"
)
# ============================================================
# 23. FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nOriginal raw data was NOT modified."
)

print(
    "\nProcessed English file:"
)

print(
    PROCESSED_DATA_PATH
)