import os

import pytest
from fastapi.testclient import TestClient

from main import SAFETY_STOCK_RATE, app


@pytest.fixture(scope="module")
def client():
    """Start the app once so the saved model loads only once for this test module."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_inventory_recommendation(client):
    response = client.post(
        "/predict",
        json={"product_id": "P0001", "current_inventory": 0},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["product_id"] == "P0001"
    assert result["forecast_demand"] >= 0
    assert result["stockout_risk"] is True
    assert result["recommended_order"] == result["forecast_demand"] + round(
        result["forecast_demand"] * SAFETY_STOCK_RATE
    )


def test_predict_rejects_unknown_product(client):
    response = client.post(
        "/predict",
        json={"product_id": "NOT-A-PRODUCT", "current_inventory": 20},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown product_id: NOT-A-PRODUCT"


def test_explain_requires_gemini_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post(
        "/explain",
        json={
            "product_id": "P0001",
            "forecast_demand": 100,
            "recommended_order": 120,
            "stockout_risk": True,
            "current_inventory": 0,
        },
    )

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def teardown_module():
    """Ensure the test process does not retain a modified environment variable."""
    os.environ.pop("GEMINI_API_KEY", None)
