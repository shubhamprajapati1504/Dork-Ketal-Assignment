import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export default function App() {
  const [productId, setProductId] = useState("P0001");
  const [inventory, setInventory] = useState(300);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [insight, setInsight] = useState("");
  const [insightError, setInsightError] = useState("");
  const [explaining, setExplaining] = useState(false);

  async function getPrediction(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    setInsight("");
    setInsightError("");

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          product_id: productId.toUpperCase(),
          current_inventory: Number(inventory),
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Could not get a forecast.");
      setResult(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function explainForecast() {
    if (!result) return;
    setExplaining(true);
    setInsightError("");
    try {
      const response = await fetch(`${API_URL}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...result, current_inventory: Number(inventory) }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Could not generate AI insight.");
      setInsight(data.explanation);
    } catch (requestError) {
      setInsightError(requestError.message);
    } finally {
      setExplaining(false);
    }
  }

  return (
    <main className="page-shell">
      <nav className="topbar">
        <div className="brand"><span className="brand-mark">◈</span> Stockwise</div>
        <span className="status"><i /> Forecast API online</span>
      </nav>

      <section className="hero">
        <div>
          <p className="eyebrow">INVENTORY INTELLIGENCE</p>
          <h1>Order with confidence.</h1>
          <p className="subtitle">Use your demand model to plan tomorrow’s stock before shelves run empty.</p>
        </div>
        <div className="hero-orb" aria-hidden="true"><span>↗</span></div>
      </section>

      <section className="card forecast-card">
        <div className="card-heading">
          <div>
            <p className="eyebrow">NEW FORECAST</p>
            <h2>Check a product</h2>
          </div>
          <p className="helper">Try products P0001–P0020</p>
        </div>
        <form onSubmit={getPrediction}>
          <label>
            Product ID
            <input value={productId} onChange={(event) => setProductId(event.target.value)} placeholder="e.g. P0001" required />
          </label>
          <label>
            Current inventory
            <input type="number" min="0" value={inventory} onChange={(event) => setInventory(event.target.value)} required />
          </label>
          <button disabled={loading}>{loading ? "Calculating..." : "Generate forecast →"}</button>
        </form>

        {error && <p className="error">⚠ {error}</p>}
        {result && (
          <div className="result-panel">
            <div className="result-heading">
              <span>Forecast for {result.product_id}</span>
              <span className="next-day">Next-day estimate</span>
            </div>
            <div className="results">
              <Metric label="Forecast demand" value={result.forecast_demand.toLocaleString()} accent="purple" />
              <Metric label="Recommended order" value={result.recommended_order.toLocaleString()} accent="blue" />
              <Metric label="Current inventory" value={Number(inventory).toLocaleString()} accent="slate" />
            </div>
            <div className={`risk ${result.stockout_risk ? "yes" : "no"}`}>
              <span>{result.stockout_risk ? "!" : "✓"}</span>
              <div><strong>{result.stockout_risk ? "Stockout risk detected" : "Stock level looks healthy"}</strong><small>{result.stockout_risk ? "Current inventory is below forecast demand." : "Current inventory covers the forecast demand."}</small></div>
            </div>
            <div className="insight">
              <div>
                <p className="insight-label">✦ AI INVENTORY INSIGHT</p>
                <p>{insight || "Ask the AI helper to turn this model output into a quick recommendation."}</p>
              </div>
              <button className="insight-button" type="button" onClick={explainForecast} disabled={explaining}>
                {explaining ? "Writing..." : "Explain forecast"}
              </button>
            </div>
            {insightError && <p className="error">⚠ {insightError}</p>}
          </div>
        )}
      </section>

      <p className="footnote">Forecasts are generated from the trained XGBoost demand model.</p>
    </main>
  );
}

function Metric({ label, value, accent }) {
  return <div className={`metric ${accent}`}><span>{label}</span><strong>{value}</strong></div>;
}
