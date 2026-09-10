"""Exercise 3 - Preparing Real-World Data for a Neural Network (Spaceship Titanic).

Reads train.csv, explores it, splits it (stratified, no leakage), preprocesses
it (imputation, encoding, log transform, scaling) and produces Figure 6.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FIG_DIR = "../figures"
rng = np.random.default_rng(42)
pd.set_option("future.no_silent_downcasting", True)

df = pd.read_csv("../data/train.csv")

# ---------------------------------------------------------------
# Item A - get to know the data
# ---------------------------------------------------------------
print("Shape:", df.shape)
print("\nTransported value counts:")
print(df["Transported"].value_counts(normalize=True))

numeric_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
categorical_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

print("\nNumeric features:", numeric_cols)
print("Categorical features:", categorical_cols)

missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_table = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_table = missing_table[missing_table["missing_count"] > 0]
print("\nMissing values:\n", missing_table)

print("\nSpend columns stats (mean / median / max), full data:")
print(df[spend_cols].agg(["mean", "median", "max"]))

# ---------------------------------------------------------------
# Item B - split before transforming (avoid data leakage)
# ---------------------------------------------------------------
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["Transported"], random_state=42
)
print(f"\nTrain shape: {train_df.shape}, Test shape: {test_df.shape}")
print("\nSpend columns stats on TRAIN ONLY (pre-transformation):")
print(train_df[spend_cols].agg(["mean", "median", "max"]))

# ---------------------------------------------------------------
# Item C - preprocessing (fit on train, apply on test)
# ---------------------------------------------------------------
def preprocess(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # --- missing data ---
    # numeric: fill with train median
    for col in numeric_cols:
        median = train_df[col].median()
        train_df[col] = train_df[col].fillna(median)
        test_df[col] = test_df[col].fillna(median)

    # categorical: fill with train mode
    for col in categorical_cols:
        mode = train_df[col].mode()[0]
        train_df[col] = train_df[col].fillna(mode)
        test_df[col] = test_df[col].fillna(mode)

    # --- feature engineering ---
    for d in (train_df, test_df):
        d["TotalSpend"] = d[spend_cols].sum(axis=1)
        d.drop(columns=["Cabin", "Name", "PassengerId"], inplace=True)

    # --- log(1+x) on spend columns (reduces heavy right tail for tanh) ---
    for col in spend_cols + ["TotalSpend"]:
        train_df[col] = np.log1p(train_df[col])
        test_df[col] = np.log1p(test_df[col])

    # --- one-hot encoding of categoricals ---
    # categories fit on train only; unseen test categories are dropped by reindex
    train_cat = pd.get_dummies(train_df[categorical_cols], dummy_na=False)
    test_cat = pd.get_dummies(test_df[categorical_cols], dummy_na=False)
    test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0)

    feature_cols = numeric_cols + ["TotalSpend"]
    X_train = pd.concat([train_df[feature_cols].reset_index(drop=True), train_cat.reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_df[feature_cols].reset_index(drop=True), test_cat.reset_index(drop=True)], axis=1)

    # --- scaling: standardization fit on train only ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    y_train = train_df["Transported"].astype(int).reset_index(drop=True)
    y_test = test_df["Transported"].astype(int).reset_index(drop=True)

    return X_train_scaled, X_test_scaled, y_train, y_test, train_df, test_df


before_foodcourt = train_df["FoodCourt"].copy()
X_train, X_test, y_train, y_test, train_df_p, test_df_p = preprocess(train_df, test_df)
after_foodcourt = np.log1p(before_foodcourt)

# ---------------------------------------------------------------
# Item D - Figure 6: before/after log transform + final checks
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(before_foodcourt, bins=40, color="tab:blue")
axes[0].set_title("FoodCourt - before log1p (train)")
axes[0].set_xlabel("FoodCourt")
axes[0].set_ylabel("Count")

axes[1].hist(after_foodcourt, bins=40, color="tab:orange")
axes[1].set_title("FoodCourt - after log1p (train)")
axes[1].set_xlabel("log(1 + FoodCourt)")
axes[1].set_ylabel("Count")

fig.suptitle("Figure 6 - Effect of log(1+x) on a heavy-tailed feature")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/figure6.png", dpi=150)
plt.close(fig)

print("\nFinal checks:")
print("NaNs in X_train:", X_train.isna().sum().sum())
print("NaNs in X_test:", X_test.isna().sum().sum())
print("X_train shape:", X_train.shape)
print("X_train min/max:", X_train.values.min(), X_train.values.max())
print("X_test min/max:", X_test.values.min(), X_test.values.max())

print("\nDone. Figures saved to", FIG_DIR)
