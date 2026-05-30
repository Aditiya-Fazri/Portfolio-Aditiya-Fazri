import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


# 12 fitur yang dipake buat training
FITUR = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd",
    "TotRmsAbvGrd",
    "Fireplaces",
    "LotArea",
]


def prepare_features(df):
    # ambil 12 kolom yang dipake
    data = df[FITUR].copy()

    # isi missing pake median tiap kolom
    for kolom in FITUR:
        if data[kolom].isnull().sum() > 0:
            data[kolom] = data[kolom].fillna(data[kolom].median())

    # bikin fitur baru TotalSF
    data["TotalSF"] = data["TotalBsmtSF"] + data["GrLivArea"]
    return data


def train():
    print("baca data...")
    df = pd.read_csv("data_house.csv")
    print(f"data: {df.shape[0]} baris, {df.shape[1]} kolom")

    X = prepare_features(df)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("training random forest...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # evaluasi
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE: {mae:,.2f}")
    print(f"R2: {r2:.4f}")

    # simpan model + metric ke joblib
    artifact = {
        "model": model,
        "features": list(X.columns),
        "metrics": {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        },
        "feature_importance": dict(
            zip(X.columns, model.feature_importances_.tolist())
        ),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }
    joblib.dump(artifact, "house_price_model.joblib")
    print("model disimpan ke house_price_model.joblib")


if __name__ == "__main__":
    train()
