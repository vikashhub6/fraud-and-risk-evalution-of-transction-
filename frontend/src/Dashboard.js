import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  RadialBarChart,
  RadialBar,
  PolarAngleAxis,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

const API_BASE = process.env.REACT_APP_API_BASE;

if (!API_BASE) {
  // eslint-disable-next-line no-console
  console.error(
    "REACT_APP_API_BASE is not set. Copy frontend/.env.example to frontend/.env " +
      "and set it to your backend URL, then restart `npm start`."
  );
}

const RISK_COLORS = {
  Safe: "#22c55e",
  Suspect: "#facc15",
  "High Risk": "#ff2d55",
};

const initialForm = {
  transaction_amount: 1200,
  location_divergence_score: 30,
  device_trust_score: 70,
  transaction_velocity: 3,
};

const FIELD_META = [
  {
    key: "transaction_amount",
    label: "Transaction Amount",
    unit: "USD",
    min: 0,
    max: 20000,
    step: 50,
  },
  {
    key: "location_divergence_score",
    label: "Location Divergence Score",
    unit: "/ 100",
    min: 0,
    max: 100,
    step: 1,
  },
  {
    key: "device_trust_score",
    label: "Device Trust Score",
    unit: "/ 100",
    min: 0,
    max: 100,
    step: 1,
  },
  {
    key: "transaction_velocity",
    label: "Transaction Velocity",
    unit: "txns / hr",
    min: 0,
    max: 30,
    step: 1,
  },
];

function StatusBadge({ status }) {
  const color = RISK_COLORS[status] || "#94a3b8";
  return (
    <span
      className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-semibold tracking-wide"
      style={{
        backgroundColor: `${color}22`,
        color,
        border: `1px solid ${color}55`,
      }}
    >
      <span
        className="w-2 h-2 rounded-full"
        style={{ backgroundColor: color, boxShadow: `0 0 8px ${color}` }}
      />
      {status}
    </span>
  );
}

function TelemetryGauge({ label, value, color }) {
  const pct = Math.round(value * 100);
  const data = [{ name: label, value: pct, fill: color }];
  return (
    <div className="flex flex-col items-center bg-slate-900/60 border border-slate-800 rounded-xl p-4">
      <div className="w-full h-40">
        <ResponsiveContainer>
          <RadialBarChart
            innerRadius="70%"
            outerRadius="100%"
            data={data}
            startAngle={90}
            endAngle={-270}
          >
            <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
            <RadialBar dataKey="value" cornerRadius={12} background={{ fill: "#1e293b" }} />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <div className="-mt-24 text-3xl font-bold" style={{ color }}>
        {pct}%
      </div>
      <div className="mt-16 text-sm text-slate-400 uppercase tracking-widest">{label}</div>
    </div>
  );
}

export default function Dashboard() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: Number(value) }));
  };

  const evaluateRisk = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/evaluate-fraud`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed with status ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const importanceData = result
    ? Object.entries(result.risk_breakdown.feature_importance).map(([name, value]) => ({
        name: name
          .split("_")
          .map((w) => w[0].toUpperCase() + w.slice(1))
          .join(" "),
        value: Number((value * 100).toFixed(1)),
      }))
    : [];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-8 py-5 flex items-center justify-between">
        <div>
          <Link to="/" className="text-xs text-cyan-400 hover:underline mb-1 inline-block">
            ← Back to overview
          </Link>
          <h1 className="text-xl font-bold tracking-tight">
            Fraud &amp; Risk Operations Console
          </h1>
          <p className="text-sm text-slate-500">
            Multimodal Financial Fraud Detection &amp; Risk Analytics Pipeline
          </p>
        </div>
        <div className="text-xs text-slate-500 font-mono">
          ML: XGBoost &middot; DL: PyTorch MLP
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-8 py-10 grid grid-cols-1 lg:grid-cols-5 gap-8">
        {/* Left: Transaction Risk Simulator Form */}
        <section className="lg:col-span-2 bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
          <h2 className="text-lg font-semibold mb-1">Transaction Risk Simulator</h2>
          <p className="text-sm text-slate-500 mb-6">
            Adjust live transaction signals to simulate an evaluation.
          </p>

          <div className="space-y-6">
            {FIELD_META.map((field) => (
              <div key={field.key}>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-sm text-slate-300">{field.label}</label>
                  <span className="text-sm font-mono text-cyan-400">
                    {form[field.key]} {field.unit}
                  </span>
                </div>
                <input
                  type="range"
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  value={form[field.key]}
                  onChange={(e) => handleChange(field.key, e.target.value)}
                  className="w-full accent-cyan-400"
                />
              </div>
            ))}
          </div>

          <button
            onClick={evaluateRisk}
            disabled={loading}
            className="mt-8 w-full py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 disabled:bg-slate-700 disabled:text-slate-400 text-slate-950 font-semibold transition-colors"
          >
            {loading ? "Evaluating..." : "Evaluate Risk"}
          </button>

          {error && (
            <p className="mt-4 text-sm text-rose-400 border border-rose-900 bg-rose-950/40 rounded-lg p-3">
              {error}
            </p>
          )}
        </section>

        {/* Right: Live Risk Telemetry Dashboard */}
        <section className="lg:col-span-3 space-y-6">
          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold">Live Risk Telemetry</h2>
              {result && <StatusBadge status={result.risk_status} />}
            </div>

            {!result ? (
              <div className="h-48 flex items-center justify-center text-slate-600 text-sm">
                Run an evaluation to see live telemetry.
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-4">
                <TelemetryGauge
                  label="ML Anomaly Score"
                  value={result.risk_breakdown.ml_anomaly_score}
                  color="#38bdf8"
                />
                <TelemetryGauge
                  label="DL Fraud Probability"
                  value={result.risk_breakdown.dl_fraud_probability}
                  color="#a78bfa"
                />
                <TelemetryGauge
                  label="Combined Risk Score"
                  value={result.risk_breakdown.combined_risk_score}
                  color={RISK_COLORS[result.risk_status] || "#ff2d55"}
                />
              </div>
            )}
          </div>

          {/* Decision Matrix Breakdown */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
            <h2 className="text-lg font-semibold mb-4">Decision Matrix Breakdown</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-sm text-slate-400 mb-3 uppercase tracking-widest">
                  ML Metadata Score
                </h3>
                {result ? (
                  <div className="space-y-2">
                    <MetricRow
                      label="XGBoost Anomaly Signal"
                      value={`${(result.risk_breakdown.ml_anomaly_score * 100).toFixed(1)}%`}
                    />
                    <MetricRow
                      label="Transaction Amount"
                      value={`$${form.transaction_amount.toLocaleString()}`}
                    />
                    <MetricRow
                      label="Location Divergence"
                      value={form.location_divergence_score}
                    />
                    <MetricRow label="Device Trust" value={form.device_trust_score} />
                    <MetricRow label="Velocity" value={form.transaction_velocity} />
                  </div>
                ) : (
                  <p className="text-sm text-slate-600">No data yet.</p>
                )}
              </div>

              <div>
                <h3 className="text-sm text-slate-400 mb-3 uppercase tracking-widest">
                  Neural Network Probability
                </h3>
                {result ? (
                  <div className="h-48">
                    <ResponsiveContainer>
                      <BarChart data={importanceData} layout="vertical" margin={{ left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis type="number" domain={[0, 100]} tick={{ fill: "#64748b", fontSize: 11 }} />
                        <YAxis
                          type="category"
                          dataKey="name"
                          width={140}
                          tick={{ fill: "#94a3b8", fontSize: 11 }}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "#0f172a",
                            border: "1px solid #1e293b",
                            borderRadius: 8,
                          }}
                        />
                        <Bar dataKey="value" fill="#38bdf8" radius={[0, 6, 6, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="text-sm text-slate-600">No data yet.</p>
                )}
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function MetricRow({ label, value }) {
  return (
    <div className="flex items-center justify-between text-sm border-b border-slate-800 pb-2">
      <span className="text-slate-400">{label}</span>
      <span className="font-mono text-slate-200">{value}</span>
    </div>
  );
}
