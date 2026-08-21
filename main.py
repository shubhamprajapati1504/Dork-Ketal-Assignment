"""FastAPI service for product demand forecasts and inventory recommendations."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DATA_PATH = BASE_DIR / "Data" / "processed_csv_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
SAFETY_STOCK_RATE = 0.20

FEATURE_COLUMNS = [
    "Store ID", "Product ID", "Category", "Region", "Weather Condition",
    "Seasonality", "Price", "Discount", "Promotion", "Competitor Pricing",
    "Epidemic", "demand_lag_1", "demand_lag_7", "demand_lag_14",
    "demand_lag_28", "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "day_of_week", "month", "day", "week_of_year", "is_weekend",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
]


class PredictionRequest(BaseModel):
    product_id: str = Field(..., examples=["P0001"])
    current_inventory: int = Field(..., ge=0, examples=[300])


class PredictionResponse(BaseModel):
    product_id: str
    forecast_demand: int
    recommended_order: int
    stockout_risk: bool


class ExplanationRequest(PredictionResponse):
    current_inventory: int = Field(..., ge=0)


class ExplanationResponse(BaseModel):
    explanation: str


def add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the historical demand features used when the model was trained."""
    data = data.sort_values(["Store ID", "Product ID", "Date"]).copy()
    demand_by_store_product = data.groupby(["Store ID", "Product ID"])["Demand"]

    for lag in (1, 7, 14, 28):
        data[f"demand_lag_{lag}"] = demand_by_store_product.shift(lag)

    data["rolling_mean_7"] = demand_by_store_product.transform(
        lambda values: values.shift(1).rolling(7).mean()
    )
    data["rolling_mean_14"] = demand_by_store_product.transform(
        lambda values: values.shift(1).rolling(14).mean()
    )
    data["rolling_std_7"] = demand_by_store_product.transform(
        lambda values: values.shift(1).rolling(7).std()
    )
    return data


def build_next_day_features(history: pd.DataFrame, product_id: str) -> pd.DataFrame:
    """Build one next-day feature row for every store that sells a product."""
    product_history = history.loc[history["Product ID"] == product_id].copy()
    if product_history.empty:
        raise ValueError(f"Unknown product_id: {product_id}")

    latest_rows = product_history.groupby("Store ID", as_index=False).tail(1).copy()
    next_date = pd.to_datetime(latest_rows["Date"]) + pd.Timedelta(days=1)

    latest_rows["day_of_week"] = next_date.dt.dayofweek
    latest_rows["month"] = next_date.dt.month
    latest_rows["day"] = next_date.dt.day
    latest_rows["week_of_year"] = next_date.dt.isocalendar().week.astype(int)
    latest_rows["is_weekend"] = (latest_rows["day_of_week"] >= 5).astype(int)
    latest_rows["dow_sin"] = np.sin(2 * np.pi * latest_rows["day_of_week"] / 7)
    latest_rows["dow_cos"] = np.cos(2 * np.pi * latest_rows["day_of_week"] / 7)
    latest_rows["month_sin"] = np.sin(2 * np.pi * latest_rows["month"] / 12)
    latest_rows["month_cos"] = np.cos(2 * np.pi * latest_rows["month"] / 12)
    return latest_rows[FEATURE_COLUMNS]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the already-trained model and its supporting historical data once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found at {MODEL_PATH}")

    history = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    app.state.history = add_lag_features(history)
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Inventory Forecast API",
    description="Forecasts next-day product demand and recommends an inventory order.",
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        features = build_next_day_features(app.state.history, request.product_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    forecast = max(0, round(float(app.state.model.predict(features).sum())))
    safety_stock = round(forecast * SAFETY_STOCK_RATE)
    recommended_order = max(0, forecast + safety_stock - request.current_inventory)

    return {
        "product_id": request.product_id,
        "forecast_demand": forecast,
        "recommended_order": recommended_order,
        "stockout_risk": request.current_inventory < forecast,
    }


@app.post("/explain", response_model=ExplanationResponse)
def explain_forecast(request: ExplanationRequest) -> dict[str, str]:
    """Use an LLM only to explain the model output; it never calculates forecasts."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail="AI helper is not configured. Add GEMINI_API_KEY to your .env file.",
        )

    prompt = f"""Explain this inventory forecast in one simple, useful sentence.
Use the numbers exactly as supplied. Do not calculate, change, or invent any values.

Product ID: {request.product_id}
Forecast demand: {request.forecast_demand}
Current inventory: {request.current_inventory}
Recommended order: {request.recommended_order}
Stockout risk: {request.stockout_risk}
    """

    try:
        from google import genai

        client = genai.Client(api_key=gemini_api_key)
        response = client.interactions.create(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
            input=(
                "You are an inventory assistant. Explain only the supplied forecast data "
                "in plain English. Keep the response under 100 words.\n\n"
                + prompt
            ),
        )
    except ImportError as error:
        raise HTTPException(
            status_code=503,
            detail="Gemini package is not installed. Run: pip install -r requirements.txt",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="Gemini explanation could not be generated. Check the API key, model, and Gemini API quota.",
        ) from error

    return {"explanation": response.output_text.strip()}
