import React from "react";
import { useNavigate } from "react-router-dom";

const pipelineSteps = [
  {
    step: "Step 01",
    title: "Feature Scaling",
    desc: "StandardScaler normalizes raw transaction metadata — amount, location divergence, device trust, velocity.",
  },
  {
    step: "Step 02",
    title: "XGBoost Classifier",
    desc: "Tabular ML engine produces a baseline anomaly probability from the scaled feature vector.",
  },
  {
    step: "Step 03",
    title: "PyTorch MLP",
    desc: "2 hidden layers, ReLU activations, Dropout regularization — ingests ML signal + features, outputs fraud probability.",
  },
  {
    step: "Step 04",
    title: "Risk Classification",
    desc: "Final probability is bucketed into Safe, Suspect, or High Risk and returned via REST API.",
  },
];

const features = [
  { icon: "⚙", color: "cyan", title: "Tabular ML Engine", desc: "XGBoost Classifier trained on transaction metadata vectors, wrapped with strict Pydantic V2 input validation." },
  { icon: "◆", color: "rose", title: "Deep Learning Risk Layer", desc: "Custom PyTorch nn.Module — Linear → ReLU → Dropout ×2 → Sigmoid, computing the final fraud probability." },
  { icon: "▤", color: "green", title: "Unified Inference API", desc: "FastAPI gateway at /api/v1/evaluate-fraud running the full ML→DL pipeline sequentially per request." },
  { icon: "◧", color: "amber", title: "Live Risk Telemetry", desc: "React + Recharts dashboard rendering radial gauges for ML anomaly score and DL fraud probability in real time." },
  { icon: "▥", color: "cyan", title: "Decision Matrix Breakdown", desc: "Side-by-side panel splitting the ML metadata score from the neural network's feature-importance driven probability." },
  { icon: "⬤", color: "rose", title: "Risk Simulator", desc: "Slider-driven transaction simulator lets you test Amount, Velocity, Location Divergence, and Device Trust live." },
];

const colorMap = {
  cyan: { bg: "rgba(34,211,238,0.12)", fg: "#22d3ee" },
  rose: { bg: "rgba(255,45,85,0.12)", fg: "#ff2d55" },
  green: { bg: "rgba(34,197,94,0.12)", fg: "#22c55e" },
  amber: { bg: "rgba(250,204,21,0.12)", fg: "#facc15" },
};

const stack = ["FastAPI", "Scikit-Learn", "XGBoost", "PyTorch", "Pydantic V2", "React", "Tailwind CSS", "Recharts"];

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 relative overflow-hidden">
      {/* background grid + glow */}
      <div
        className="pointer-events-none fixed inset-0 z-0"
        style={{
          backgroundImage:
            "linear-gradient(to right, rgba(148,163,184,0.06) 1px, transparent 1px), linear-gradient(to bottom, rgba(148,163,184,0.06) 1px, transparent 1px)",
          backgroundSize: "40px 40px",
          maskImage: "radial-gradient(ellipse at 50% 0%, black 40%, transparent 75%)",
        }}
      />
      <div
        className="pointer-events-none fixed top-[-200px] left-1/2 -translate-x-1/2 w-[900px] h-[500px] z-0 blur-md"
        style={{ background: "radial-gradient(circle, rgba(34,211,238,0.18), transparent 70%)" }}
      />

      {/* Nav */}
      <nav className="sticky top-0 z-10 backdrop-blur-md bg-slate-950/75 border-b border-slate-800">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2 font-bold tracking-tight">
            <span
              className="w-2.5 h-2.5 rounded-sm bg-cyan-400"
              style={{ boxShadow: "0 0 12px #22d3ee" }}
            />
            FraudGuard AI
          </div>
          <div className="flex items-center gap-6 text-sm">
            <a href="#pipeline" className="text-slate-500 hover:text-slate-100 transition-colors">Pipeline</a>
            <a href="#features" className="text-slate-500 hover:text-slate-100 transition-colors">Features</a>
            <a href="#stack" className="text-slate-500 hover:text-slate-100 transition-colors">Stack</a>
            <button
              onClick={() => navigate("/dashboard")}
              className="px-4 py-2 rounded-lg font-semibold text-sm bg-cyan-400 text-slate-950 hover:shadow-[0_0_20px_rgba(34,211,238,0.5)] transition-shadow"
            >
              Enter Dashboard →
            </button>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <header className="relative z-[1] max-w-3xl mx-auto text-center px-6 pt-24 pb-16">
        <div className="inline-flex items-center gap-2 text-xs tracking-widest uppercase text-cyan-400 border border-cyan-400/30 bg-cyan-400/10 px-4 py-1.5 rounded-full mb-7">
          ML + Deep Learning &middot; No LLMs, No Chatbots
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold leading-tight tracking-tight mb-5 bg-gradient-to-b from-slate-50 to-slate-400 bg-clip-text text-transparent">
          Multimodal Financial Fraud
          <br />
          Detection &amp; Risk Analytics Pipeline
        </h1>
        <p className="text-slate-400 text-lg leading-relaxed mb-10 max-w-xl mx-auto">
          A hybrid sequential architecture where a Scikit-Learn / XGBoost tabular
          engine feeds its anomaly signal directly into a PyTorch neural network,
          producing real-time fraud probability and risk classification for live
          transactions.
        </p>
        <div className="flex gap-4 justify-center flex-wrap">
          <button
            onClick={() => navigate("/dashboard")}
            className="px-7 py-3.5 rounded-xl font-semibold bg-cyan-400 text-slate-950 hover:shadow-[0_0_32px_rgba(34,211,238,0.55)] transition-shadow"
          >
            Enter Dashboard →
          </button>
          <a
            href="#stack"
            className="px-7 py-3.5 rounded-xl font-semibold bg-slate-800/40 border border-slate-800 text-slate-100 hover:border-slate-600 transition-colors"
          >
            See Tech Stack
          </a>
        </div>
      </header>

      {/* Pipeline */}
      <div id="pipeline" className="relative z-[1] max-w-5xl mx-auto px-6 pb-20">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3.5">
          {pipelineSteps.map((s, i) => (
            <div key={i} className="bg-slate-900 border border-slate-800 rounded-2xl p-5 relative">
              <div className="text-[11px] text-slate-500 uppercase tracking-wider mb-2">{s.step}</div>
              <h4 className="text-sm font-semibold mb-2">{s.title}</h4>
              <p className="text-[13px] text-slate-500 leading-relaxed">{s.desc}</p>
              {i < pipelineSteps.length - 1 && (
                <span className="hidden md:block absolute -right-4 top-1/2 -translate-y-1/2 text-cyan-400 text-lg">→</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Features */}
      <section id="features" className="relative z-[1] border-t border-slate-800 py-16">
        <div className="max-w-5xl mx-auto px-6">
          <div className="text-center mb-11">
            <div className="text-cyan-400 text-xs tracking-widest uppercase font-bold">Architecture</div>
            <h2 className="text-3xl font-bold mt-2.5 mb-3 tracking-tight">Two models, one sequential decision</h2>
            <p className="text-slate-500 max-w-md mx-auto text-sm">
              Not an ensemble vote — a true pipeline where the ML stage's output becomes a DL input feature.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {features.map((f, i) => {
              const c = colorMap[f.color];
              return (
                <div
                  key={i}
                  className="bg-slate-900 border border-slate-800 rounded-2xl p-6 hover:border-slate-600 hover:-translate-y-0.5 transition-all"
                >
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center mb-4 text-lg"
                    style={{ backgroundColor: c.bg, color: c.fg }}
                  >
                    {f.icon}
                  </div>
                  <h3 className="text-[16.5px] font-semibold mb-2">{f.title}</h3>
                  <p className="text-sm text-slate-500 leading-relaxed">{f.desc}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Stack */}
      <section id="stack" className="relative z-[1] border-t border-slate-800 py-16">
        <div className="max-w-5xl mx-auto px-6 text-center">
          <div className="text-cyan-400 text-xs tracking-widest uppercase font-bold">Tech Stack</div>
          <h2 className="text-3xl font-bold mt-2.5 mb-9 tracking-tight">Built with</h2>
          <div className="flex flex-wrap gap-2.5 justify-center">
            {stack.map((s, i) => (
              <span
                key={i}
                className="flex items-center gap-2 px-4 py-2 rounded-full border border-slate-800 bg-slate-900 text-sm"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                {s}
              </span>
            ))}
          </div>
          <button
            onClick={() => navigate("/dashboard")}
            className="mt-10 px-7 py-3.5 rounded-xl font-semibold bg-cyan-400 text-slate-950 hover:shadow-[0_0_32px_rgba(34,211,238,0.55)] transition-shadow"
          >
            Enter Dashboard →
          </button>
        </div>
      </section>

      <footer className="relative z-[1] text-center text-slate-500 text-xs py-10">
        Multimodal Financial Fraud Detection &amp; Risk Analytics Pipeline — Major Project
      </footer>
    </div>
  );
}
