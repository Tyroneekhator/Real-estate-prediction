import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def outliers(df: pd.DataFrame, ft: str):
    q1 = df[ft].quantile(0.25)
    q3 = df[ft].quantile(0.75)
    iqr = q3 - q1
    min_val = q1 - 1.5 * iqr
    max_val = q3 + 1.5 * iqr
    ls = df.index[(df[ft] < min_val) | (df[ft] > max_val)]
    return ls


def remove(df: pd.DataFrame, ls):
    ls = sorted(set(ls))
    return df.drop(ls)


def remove_pps_outliers(df: pd.DataFrame):
    df_out = pd.DataFrame()
    for _, subdf in df.groupby("state"):
        m = np.mean(subdf.price_per_room)
        st = np.std(subdf.price_per_room)
        reduced_df = subdf[
            (subdf.price_per_room > (m - st)) &
            (subdf.price_per_room <= (m + st))
        ]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


def clean_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop toilets column if present
    if "toilets" in df.columns:
        df = df.drop(["toilets"], axis=1)

    # Remove outliers
    index_list = []
    for col in ["bedrooms", "bathrooms", "parking_space", "price"]:
        if col in df.columns:
            index_list.extend(outliers(df, col))

    df2 = remove(df, index_list).copy()

    # Rename title -> house_type
    if "title" in df2.columns:
        df2 = df2.rename(columns={"title": "house_type"})

    # Feature engineering
    df2["total_rooms"] = df2["bedrooms"] + df2["bathrooms"]
    df2["price_per_room"] = df2["price"] / df2["total_rooms"].replace(0, np.nan)
    df2 = df2.dropna().copy()

    # Remove price-per-room outliers
    df3 = remove_pps_outliers(df2).copy()

    # Remove rare states/towns
    few_state_records = df3["state"].value_counts()
    few_town_records = df3["town"].value_counts()

    df4 = df3[
        ~df3["state"].isin(few_state_records[few_state_records < 10].index)
        & ~df3["town"].isin(few_town_records[few_town_records < 10].index)
    ].copy()

    # Drop parking_space if present
    if "parking_space" in df4.columns:
        df4 = df4.drop(["parking_space"], axis=1)

    return df4


def train_and_save_model():
    csv_path = "nigeria_houses_data.csv"
    model_path = "model.joblib"

    df = clean_data(csv_path)

    # Final training columns
    X = df[["bedrooms", "bathrooms", "house_type", "town", "state"]].copy()
    y = df["price"].copy()

    # Encode categorical columns
    le_house_type = LabelEncoder()
    le_town = LabelEncoder()
    le_state = LabelEncoder()

    X["house_type"] = le_house_type.fit_transform(X["house_type"])
    X["town"] = le_town.fit_transform(X["town"])
    X["state"] = le_state.fit_transform(X["state"])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Training complete.")
    print(f"Rows used: {len(df)}")
    print(f"MAE: {mae:,.2f}")
    print(f"R2 Score: {r2:.4f}")

    # Save everything needed for prediction
    model_bundle = {
        "model": model,
        "le_house_type": le_house_type,
        "le_town": le_town,
        "le_state": le_state,
        "states": sorted(df["state"].unique().tolist()),
        "towns_by_state": {
            state: sorted(df[df["state"] == state]["town"].unique().tolist())
            for state in sorted(df["state"].unique())
        },
        "house_types": sorted(df["house_type"].unique().tolist()),
    }

    joblib.dump(model_bundle, model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    train_and_save_model()