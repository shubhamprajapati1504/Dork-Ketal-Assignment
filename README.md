# Inventory Demand Forecasting

This project forecasts next-day product demand for retail stores and turns that forecast into a simple inventory-order recommendation. It includes a FastAPI backend, a React dashboard, a trained XGBoost model, automated API tests, and Docker support.

## What is in the project

- `main.py` - FastAPI service with health, forecast, and explanation endpoints.
- `train_model.py` - trains and saves the XGBoost demand model.
- `frontend/` - React/Vite dashboard for requesting forecasts.
- `tests/test_api.py` - API tests.
- `Data/model_comparison.csv` - recorded model-evaluation results.
- `database.py` and `queries.sql` - optional SQLite database setup and example analysis queries.

## Dataset

The project uses Kaggle's [Retail Store Inventory and Demand Forecasting dataset](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting?resource=download).

The copy used here is `Data/sales_data.csv`. It contains 76,000 daily records from 1 January 2022 to 30 January 2024, covering 5 stores, 20 products, 5 product categories, and 4 regions. Key columns include demand, inventory level, units sold and ordered, price and discount, promotion, weather, seasonality, competitor pricing, and epidemic status.

`Data/processed_csv_dataset.csv` is the version used by the model. It adds calendar features such as weekday, month, and week of year.

## Assumptions

- Each API forecast is for the next day.
- The most recent row for each store/product pair represents the information available at prediction time.
- Demand is forecast separately for every store that carries the requested product, then summed into one product-level forecast.
- A product is considered at stockout risk when current inventory is less than the forecast demand.
- The model treats the supplied data as representative of future conditions. In production, data freshness and data quality checks would be needed.

## Model choice

I chose **XGBoost** because it performed best in the evaluation and works well with the mix of numeric, categorical, calendar, and lagged-demand features in this dataset.

The training pipeline uses:

- One-hot encoding for store, product, category, region, weather condition, and seasonality.
- Numeric business inputs such as price, discount, promotion, competitor pricing, and epidemic status.
- Demand lags of 1, 7, 14, and 28 days plus rolling 7- and 14-day demand statistics.
- A time-based split: the first 80% of dates are used for training, keeping the latest 20% for evaluation.

## Evaluation results

Lower is better for MAE, RMSE, and MAPE.

| Model | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| **XGBoost** | **16.54** | **22.38** | **24.24%** |
| Random Forest | 24.35 | 31.24 | 38.31% |
| Linear Regression | 25.37 | 32.73 | 36.77% |

XGBoost had the lowest error on every recorded metric, so it is the model served by the API.

## Inventory rule

The API applies a 20% safety-stock buffer to the forecast:

```text
recommended order = max(0, forecast demand + round(forecast demand * 0.20) - current inventory)
```

For example, if forecast demand is 100 units and current inventory is 50 units, the safety stock is 20 units and the recommended order is 70 units.

## Run locally

Prerequisite: Python 3.12 or later.

In PowerShell, from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
uvicorn main:app --reload
```

The API is then available at <http://localhost:8000>. Interactive API documentation is available at <http://localhost:8000/docs>.

To check that it is running:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

To request a forecast:

```powershell
Invoke-RestMethod http://localhost:8000/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"product_id":"P0001","current_inventory":0}'
```

### Optional Gemini explanation

The forecast itself never requires Gemini. To enable the `/explain` endpoint, create `.env` from `.env.example` and add a valid `GEMINI_API_KEY`:

```powershell
Copy-Item .env.example .env
```

## Run the dashboard

Keep the backend running, then open another PowerShell window:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, normally <http://localhost:5173>.

## Run with Docker

Prerequisite: Docker Desktop is installed and running.

From the project folder:

```powershell
docker compose up --build
```

The first build downloads the Python image and dependencies, trains a fresh model, and starts the API on port 8000. In a second PowerShell window, test it with:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Stop the service with `Ctrl+C`, then run:

```powershell
docker compose down
```

After the first build, use `docker compose up` unless you changed code, dependencies, the Dockerfile, or training data.

## Run tests

First train the model if you have not already done so, then run:

```powershell
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m pytest -q
```

The test suite checks the health endpoint, a successful inventory recommendation, unknown-product validation, and behavior when the Gemini key is absent.

## Optional database setup

Create the SQLite database from the processed data:

```powershell
.\.venv\Scripts\python.exe database.py
```

This creates `database/inventory.db`. Example SQL analysis queries are in `queries.sql`.

## With more time

- Experiment further with LSTM models using longer demand sequences for each store-product pair, then compare their accuracy and training time against XGBoost.
- Tune the LSTM sequence length, number of layers, hidden units, dropout, batch size, and number of training epochs to see whether it can improve next-day forecasts.
- Test hybrid features for the LSTM, combining past demand with price, discount, promotion, weather, seasonality, competitor pricing, and epidemic information.
- Analyse how promotions, discounts, weather conditions, seasons, and regions affect demand for individual product categories.
- Build product- and store-level dashboards to identify fast-moving products, slow-moving products, frequent stockout risks, and stores with unusually high demand.
- Compare inventory recommendations for different safety-stock percentages using the available inventory-level and demand columns.
- Add weekly and monthly forecast views, using the daily dataset to help with longer-term ordering and stock planning.
