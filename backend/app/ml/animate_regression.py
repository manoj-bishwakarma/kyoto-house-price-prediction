# -*- coding: utf-8 -*-

"""
Kyoto House Price Prediction
----------------------------

This script:

1. Loads Kyoto real-estate transaction data
2. Selects land + building transactions
3. Extracts land area and transaction price
4. Applies log transformation to price
5. Splits data into training and testing sets
6. Trains a Linear Regression model
7. Evaluates the model using:
      - MAE
      - RMSE
      - R²
8. Creates an educational animation
9. Saves the animation directly as MP4

Output:
    outputs/kyoto_linear_regression_animated.mp4
"""


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd

import matplotlib

# Required when generating a video without opening a GUI window
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# Current file:
#
# backend/
# └── app/
#     └── ml/
#         └── model.py
#
# parents[0] = ml
# parents[1] = app
# parents[2] = backend

BASE_DIR = Path(__file__).resolve().parents[2]


# Input CSV
DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "kyoto_real_estate.csv"
)


# Output directory
OUTPUT_DIR = BASE_DIR / "outputs"

# Create outputs folder automatically
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# MP4 output
OUT_MP4 = (
    OUTPUT_DIR
    / "kyoto_linear_regression_animated.mp4"
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 60)
print("KYOTO HOUSE PRICE PREDICTION")
print("=" * 60)

print("\nLoading Kyoto real estate dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="cp932"
)

print("\nDataset loaded successfully!")

print(
    f"Original dataset shape: {df.shape}"
)


# ============================================================
# 4. SELECT HOUSE + BUILDING TRANSACTIONS
# ============================================================

print("\nSelecting land + building transactions...")

df = df[
    df["種類"] == "宅地(土地と建物)"
].copy()

print(
    f"After filtering: {df.shape}"
)


# ============================================================
# 5. CONVERT LAND AREA TO NUMERIC
# ============================================================

"""
Japanese column:

面積（㎡）
めんせき
Area in square meters

For these transactions, this represents the land area.

The original CSV stores this field as text, so we convert
it into a numerical feature.
"""

print("\nConverting land area...")

df["area_m2"] = (
    df["面積（㎡）"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.extract(r"([\d.]+)")[0]
    .astype(float)
)


# ============================================================
# 6. CONVERT TARGET PRICE
# ============================================================

"""
Japanese column:

取引価格（総額）
とりひきかかく（そうがく）
Total transaction price

This is our target variable.
"""

print("Converting transaction price...")

df["price"] = pd.to_numeric(
    df["取引価格（総額）"],
    errors="coerce"
)


# ============================================================
# 7. REMOVE INVALID DATA
# ============================================================

print("\nRemoving invalid records...")

df = df.dropna(
    subset=[
        "area_m2",
        "price"
    ]
)

# Area and price must be positive
df = df[
    (df["area_m2"] > 0)
    &
    (df["price"] > 0)
].copy()


print(
    f"Usable records: {len(df)}"
)


# ============================================================
# 8. DISPLAY BASIC DATA INFORMATION
# ============================================================

print("\nData summary")
print("-" * 60)

print(
    f"Minimum land area : {df['area_m2'].min():.1f} m²"
)

print(
    f"Maximum land area : {df['area_m2'].max():.1f} m²"
)

print(
    f"Minimum price     : ¥{df['price'].min():,.0f}"
)

print(
    f"Median price      : ¥{df['price'].median():,.0f}"
)

print(
    f"Maximum price     : ¥{df['price'].max():,.0f}"
)


# ============================================================
# 9. LOG TRANSFORMATION
# ============================================================

"""
Real-estate prices are strongly right-skewed.

Example:

¥20 million
¥30 million
¥50 million
...
¥1.4 billion

A few extremely expensive properties can strongly influence
ordinary Linear Regression.

Therefore we train the model on:

log(1 + price)

After prediction, we convert the result back to yen.
"""

print("\nApplying log transformation to transaction price...")

df["log_price"] = np.log1p(
    df["price"]
)


# ============================================================
# 10. CREATE FEATURES AND TARGET
# ============================================================

X = df[
    ["area_m2"]
]

y = df[
    "log_price"
]


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ============================================================
# 12. TRAIN LINEAR REGRESSION
# ============================================================

print("\nTraining Linear Regression...")

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ============================================================
# 13. MODEL COEFFICIENTS
# ============================================================

coefficient = model.coef_[0]

intercept = model.intercept_


print("\nModel equation")
print("-" * 60)

print(
    f"log(price) = "
    f"{coefficient:.6f} × area + "
    f"{intercept:.6f}"
)


# ============================================================
# 14. PREDICT TEST DATA
# ============================================================

print("\nGenerating predictions...")

y_pred_log = model.predict(
    X_test
)


# Convert predictions back to yen
y_pred_price = np.expm1(
    y_pred_log
)

# Convert actual test values back to yen
y_test_price = np.expm1(
    y_test
)


# ============================================================
# 15. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test_price,
    y_pred_price
)


rmse = np.sqrt(
    mean_squared_error(
        y_test_price,
        y_pred_price
    )
)


r2 = r2_score(
    y_test_price,
    y_pred_price
)


# ============================================================
# 16. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"MAE  : ¥{mae:,.0f}"
)

print(
    f"RMSE : ¥{rmse:,.0f}"
)

print(
    f"R²   : {r2:.4f}"
)

print("=" * 60)


# ============================================================
# 17. PREPARE DATA FOR ANIMATION
# ============================================================

train_x = X_train[
    "area_m2"
].to_numpy()

train_y = np.expm1(
    y_train.to_numpy()
)


test_x = X_test[
    "area_m2"
].to_numpy()

test_y = y_test_price.to_numpy()


# Random order for point appearance
rng = np.random.default_rng(42)

train_order = rng.permutation(
    len(train_x)
)

test_order = rng.permutation(
    len(test_x)
)


# ============================================================
# 18. CREATE REGRESSION LINE
# ============================================================

x_min = df[
    "area_m2"
].min()

x_max = df[
    "area_m2"
].max()


x_line = np.linspace(
    x_min,
    x_max,
    200
)


# Model predicts LOG price
log_prediction = model.predict(
    x_line.reshape(-1, 1)
)


# Convert back to actual price
price_line = np.expm1(
    log_prediction
)


# ============================================================
# 19. ANIMATION SETTINGS
# ============================================================

# Number of frames each point remains visible
POINT_HOLD = 2

# Pause before regression line starts
PAUSE_BEFORE_LINE = 15

# Number of frames used to draw regression line
LINE_DRAW_FRAMES = 35

# Final screen hold
FINAL_HOLD = 50


# ============================================================
# 20. BUILD ANIMATION FRAME SEQUENCE
# ============================================================

frames = []


# ------------------------------------------------------------
# Stage 1: Training points appear
# ------------------------------------------------------------

for i in range(
    1,
    len(train_x) + 1
):

    frames += [
        (
            "train",
            i,
            0.0
        )
    ] * POINT_HOLD


# ------------------------------------------------------------
# Stage 2: Pause
# ------------------------------------------------------------

frames += [
    (
        "pause",
        len(train_x),
        0.0
    )
] * PAUSE_BEFORE_LINE


# ------------------------------------------------------------
# Stage 3: Regression line appears
# ------------------------------------------------------------

for f in range(
    1,
    LINE_DRAW_FRAMES + 1
):

    fraction = (
        f /
        LINE_DRAW_FRAMES
    )

    frames += [
        (
            "line",
            len(train_x),
            fraction
        )
    ]


# ------------------------------------------------------------
# Stage 4: Test points appear
# ------------------------------------------------------------

for i in range(
    1,
    len(test_x) + 1
):

    frames += [
        (
            "test",
            i,
            1.0
        )
    ]


# ------------------------------------------------------------
# Stage 5: Final result
# ------------------------------------------------------------

frames += [
    (
        "final",
        len(test_x),
        1.0
    )
] * FINAL_HOLD


print(
    f"\nTotal animation frames: {len(frames)}"
)


# ============================================================
# 21. CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6.5)
)


# ------------------------------------------------------------
# Y-axis uses logarithmic scale
# ------------------------------------------------------------

ax.set_yscale(
    "log"
)


# ------------------------------------------------------------
# X-axis
# ------------------------------------------------------------

ax.set_xlim(
    x_min * 0.9,
    x_max * 1.05
)


# ------------------------------------------------------------
# Y-axis
# ------------------------------------------------------------

price_min = df[
    "price"
].min()

price_max = df[
    "price"
].max()


ax.set_ylim(
    price_min * 0.8,
    price_max * 1.2
)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

ax.set_xlabel(
    "Land Area (m²)",
    fontsize=12
)

ax.set_ylabel(
    "Transaction Price (JPY, log scale)",
    fontsize=12
)


# ------------------------------------------------------------
# Main title
# ------------------------------------------------------------

ax.set_title(
    "Kyoto House Price Prediction",
    fontsize=16,
    fontweight="bold"
)


# ============================================================
# 22. CREATE SCATTER OBJECTS
# ============================================================

# Training data
train_scatter = ax.scatter(
    [],
    [],
    s=35,
    alpha=0.65,
    label="Training data"
)


# Test data
test_scatter = ax.scatter(
    [],
    [],
    s=50,
    alpha=0.90,
    marker="x",
    label="Test data"
)


# Regression line
fit_line, = ax.plot(
    [],
    [],
    linewidth=3,
    label="Linear Regression"
)


# Legend
ax.legend(
    loc="upper left"
)


# ============================================================
# 23. ANIMATION TITLE
# ============================================================

title = ax.set_title(
    "Loading Kyoto real estate data...",
    fontsize=15,
    fontweight="bold"
)


# ============================================================
# 24. ANIMATION UPDATE FUNCTION
# ============================================================

def update(frame_number):

    stage, count, fraction = (
        frames[frame_number]
    )


    # ========================================================
    # TRAINING DATA
    # ========================================================

    if stage in [
        "train",
        "pause",
        "line",
        "test",
        "final"
    ]:

        idx = train_order[
            :count
        ]

        if len(idx) > 0:

            train_scatter.set_offsets(
                np.column_stack(
                    [
                        train_x[idx],
                        train_y[idx]
                    ]
                )
            )


    # ========================================================
    # REGRESSION LINE
    # ========================================================

    if stage in [
        "line",
        "test",
        "final"
    ]:

        line_count = max(
            2,
            int(
                len(x_line)
                * fraction
            )
        )

        fit_line.set_data(
            x_line[:line_count],
            price_line[:line_count]
        )


    # ========================================================
    # TEST DATA
    # ========================================================

    if stage in [
        "test",
        "final"
    ]:

        idx = test_order[
            :count
        ]

        if len(idx) > 0:

            test_scatter.set_offsets(
                np.column_stack(
                    [
                        test_x[idx],
                        test_y[idx]
                    ]
                )
            )


    # ========================================================
    # TITLES
    # ========================================================

    if stage == "train":

        title.set_text(
            "Kyoto real-estate training data "
            f"({count}/{len(train_x)})"
        )


    elif stage == "pause":

        title.set_text(
            "Learning the relationship between "
            "land area and transaction price..."
        )


    elif stage == "line":

        title.set_text(
            "Fitting Linear Regression..."
        )


    elif stage == "test":

        title.set_text(
            "Testing the model..."
        )


    elif stage == "final":

        title.set_text(
            "Kyoto House Price Prediction | "
            f"MAE: ¥{mae / 1_000_000:.1f}M | "
            f"RMSE: ¥{rmse / 1_000_000:.1f}M | "
            f"R²: {r2:.3f}"
        )


    return (
        train_scatter,
        test_scatter,
        fit_line,
        title
    )


# ============================================================
# 25. CREATE ANIMATION
# ============================================================

print("\nCreating animation...")

anim = animation.FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=50,
    blit=False
)


# ============================================================
# 26. LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# 27. CREATE MP4 WRITER
# ============================================================

print("\nPreparing MP4 encoder...")

writer = animation.FFMpegWriter(
    fps=20,
    metadata={
        "title": "Kyoto House Price Prediction",
        "artist": "Manoj",
        "comment": (
            "Linear Regression on Kyoto "
            "real-estate transaction data"
        )
    },
    bitrate=4000
)


# ============================================================
# 28. SAVE MP4
# ============================================================

print("\nSaving MP4...")

anim.save(
    OUT_MP4,
    writer=writer,
    dpi=150
)


# ============================================================
# 29. CLEAN UP
# ============================================================

plt.close()


# ============================================================
# 30. FINISHED
# ============================================================

print("\n")
print("=" * 60)
print("ANIMATION COMPLETE")
print("=" * 60)

print(
    f"MP4 saved to:\n{OUT_MP4}"
)

print(
    f"\nTotal frames: {len(frames)}"
)

print(
    "\nYou can now open the MP4 and upload it to LinkedIn."
)

print("=" * 60)