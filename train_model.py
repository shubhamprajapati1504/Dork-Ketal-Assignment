"""Train the XGBoost pipeline once and save it for the FastAPI service."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from main import FEATURE_COLUMNS, add_lag_features


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "processed_csv_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"

CATEGORICAL_FEATURES = [
    "Store ID", "Product ID", "Category", "Region", "Weather Condition", "Seasonality"
]
NUMERIC_FEATURES = [feature for feature in FEATURE_COLUMNS if feature not in CATEGORICAL_FEATURES]
LAG_FEATURES = [
    "demand_lag_1", "demand_lag_7", "demand_lag_14", "demand_lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
]


def train_and_save_model() -> None:
    data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    model_data = add_lag_features(data).dropna(subset=LAG_FEATURES)

    # The last 20% of dates are held out, preserving the time order.
    split_date = sorted(model_data["Date"].unique())[int(model_data["Date"].nunique() * 0.8)]
    train_data = model_data.loc[model_data["Date"] < split_date]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                objective="reg:squarederror", random_state=42, n_jobs=-1,
            )),
        ]
    )
    model.fit(train_data[FEATURE_COLUMNS], train_data["Demand"])

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
