import { useEffect, useMemo, useState } from "react";

import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Link2,
  MessageSquareText,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  TrendingUp,
} from "lucide-react";

import {
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

import {
  analyzeSms,
  analyzeUrl,
  getHealth,
  getHistory,
} from "./api";

import "./App.css";


function RiskBadge({ band }) {
  const normalized =
    (band || "UNKNOWN").toLowerCase();

  return (
    <span
      className={`risk-badge risk-${normalized}`}
    >
      {band}
    </span>
  );
}


function MetricCard({
  icon: Icon,
  label,
  value,
  description,
}) {
  return (
    <div className="metric-card">
      <div className="metric-icon">
        <Icon size={19} />
      </div>

      <div>
        <div className="metric-label">
          {label}
        </div>

        <div className="metric-value">
          {value}
        </div>

        {description && (
          <div className="metric-description">
            {description}
          </div>
        )}
      </div>
    </div>
  );
}


function App() {
  const [mode, setMode] =
    useState("url");

  const [input, setInput] =
    useState("");

  const [result, setResult] =
    useState(null);

  const [history, setHistory] =
    useState([]);

  const [loading, setLoading] =
    useState(false);

  const [historyLoading, setHistoryLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  const [apiOnline, setApiOnline] =
    useState(false);


  async function refreshHistory() {
    try {
      const data =
        await getHistory(50);

      setHistory(
        data.items || [],
      );
    } catch {
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  }


  useEffect(() => {
    async function initialize() {
      try {
        await getHealth();
        setApiOnline(true);
      } catch {
        setApiOnline(false);
      }

      await refreshHistory();
    }

    initialize();
  }, []);


  async function handleAnalyze() {
    const value =
      input.trim();

    if (!value) {
      setError(
        mode === "url"
          ? "Enter a URL to analyze."
          : "Enter a message to analyze.",
      );
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const analysis =
        mode === "url"
          ? await analyzeUrl(value)
          : await analyzeSms(value);

      setResult(analysis);
      setInput("");

      await refreshHistory();
    } catch (err) {
      setError(
        err.message ||
        "Analysis failed.",
      );
    } finally {
      setLoading(false);
    }
  }


  function handleKeyDown(event) {
    if (
      event.key === "Enter" &&
      (event.ctrlKey || event.metaKey)
    ) {
      handleAnalyze();
    }
  }


  const threatDistribution =
    useMemo(() => {
      const phishing =
        history.filter(
          (item) =>
            item.prediction === "phishing",
        ).length;

      const scam =
        history.filter(
          (item) =>
            item.prediction === "scam",
        ).length;

      const legitimate =
        history.filter(
          (item) =>
            item.prediction === "legitimate",
        ).length;

      return [
        {
          name: "Phishing",
          value: phishing,
        },
        {
          name: "Scam",
          value: scam,
        },
        {
          name: "Legitimate",
          value: legitimate,
        },
      ].filter(
        (item) => item.value > 0,
      );
    }, [history]);


  const riskDistribution =
    useMemo(() => {
      return [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
      ]
        .map(
          (band) => ({
            name: band,
            value: history.filter(
              (item) =>
                item.risk_band === band,
            ).length,
          }),
        )
        .filter(
          (item) => item.value > 0,
        );
    }, [history]);


  const totalAnalyses =
    history.length;

  const threatsDetected =
    history.filter(
      (item) =>
        item.prediction === "phishing" ||
        item.prediction === "scam",
    ).length;

  const criticalCount =
    history.filter(
      (item) =>
        item.risk_band === "CRITICAL",
    ).length;


  return (
    <div className="app-shell">

      <header className="topbar">

        <div className="brand">

          <div className="brand-mark">
            <Shield size={22} />
          </div>

          <div>
            <div className="brand-name">
              ThreatGuard
            </div>

            <div className="brand-subtitle">
              AI Digital Threat Intelligence
            </div>
          </div>

        </div>


        <div className="system-status">

          <span
            className={
              apiOnline
                ? "status-dot online"
                : "status-dot offline"
            }
          />

          {apiOnline
            ? "API ONLINE"
            : "API OFFLINE"}

        </div>

      </header>


      <main className="main-content">

        <section className="hero">

          <div className="hero-copy">

            <div className="eyebrow">
              <Sparkles size={15} />
              REAL-TIME AI ANALYSIS
            </div>

            <h1>
              Detect digital threats
              <span>
                before they become incidents.
              </span>
            </h1>

            <p>
              Analyze suspicious URLs and
              messages using ThreatGuard's
              machine-learning detection engines.
            </p>

          </div>

        </section>


        <section className="analyzer-card">

          <div className="mode-tabs">

            <button
              className={
                mode === "url"
                  ? "mode-tab active"
                  : "mode-tab"
              }
              onClick={() => {
                setMode("url");
                setResult(null);
                setError("");
              }}
            >
              <Link2 size={17} />
              URL Analysis
            </button>


            <button
              className={
                mode === "sms"
                  ? "mode-tab active"
                  : "mode-tab"
              }
              onClick={() => {
                setMode("sms");
                setResult(null);
                setError("");
              }}
            >
              <MessageSquareText size={17} />
              SMS Analysis
            </button>

          </div>


          <div className="input-header">

            <div>

              <h2>
                {mode === "url"
                  ? "Analyze a suspicious URL"
                  : "Analyze a suspicious message"}
              </h2>

              <p>
                {mode === "url"
                  ? "Enter the complete URL for lexical threat analysis."
                  : "Paste the suspicious SMS or message content."}
              </p>

            </div>

          </div>


          <textarea
            className="threat-input"
            value={input}
            onChange={(event) =>
              setInput(event.target.value)
            }
            onKeyDown={handleKeyDown}
            placeholder={
              mode === "url"
                ? "https://example.com/account/verify"
                : "URGENT! Your account has been suspended..."
            }
            rows={5}
          />


          <div className="analyzer-footer">

            <span className="shortcut">
              Ctrl + Enter to analyze
            </span>

            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading
                ? "Analyzing..."
                : "Analyze Threat"}

              {!loading && (
                <TrendingUp size={17} />
              )}
            </button>

          </div>


          {error && (
            <div className="error-banner">
              <AlertTriangle size={18} />
              {error}
            </div>
          )}

        </section>


        {result && (
          <section className="result-card">

            <div className="result-header">

              <div>

                <div className="result-kicker">
                  ANALYSIS COMPLETE
                </div>

                <h2>
                  Threat assessment
                </h2>

              </div>

              <RiskBadge
                band={result.risk_band}
              />

            </div>


            <div className="result-grid">

              <div className="score-panel">

                <div className="score-label">
                  RISK SCORE
                </div>

                <div className="score-value">
                  {result.risk_score}
                  <span>/100</span>
                </div>

                <div className="score-track">

                  <div
                    className={`score-fill risk-${result.risk_band.toLowerCase()}`}
                    style={{
                      width: `${Math.min(
                        100,
                        Math.max(
                          0,
                          result.risk_score,
                        ),
                      )}%`,
                    }}
                  />

                </div>

              </div>


              <div className="classification-panel">

                <div className="classification-icon">

                  {result.prediction ===
                    "legitimate" ? (
                    <ShieldCheck size={30} />
                  ) : (
                    <ShieldAlert size={30} />
                  )}

                </div>


                <div>

                  <div className="score-label">
                    CLASSIFICATION
                  </div>

                  <div className="classification-value">
                    {result.prediction}
                  </div>

                  <div className="confidence">
                    Confidence:{" "}
                    {(
                      result.confidence * 100
                    ).toFixed(2)}
                    %
                  </div>

                </div>

              </div>


              <div className="reasons-panel">

                <div className="score-label">
                  DETECTION SIGNALS
                </div>

                {result.reasons?.length ? (
                  <ul>
                    {result.reasons.map(
                      (reason, index) => (
                        <li key={index}>
                          <AlertTriangle
                            size={14}
                          />
                          {reason}
                        </li>
                      ),
                    )}
                  </ul>
                ) : (
                  <div className="no-signals">
                    No major heuristic
                    signals detected.
                  </div>
                )}

              </div>

            </div>


            <div className="model-footer">

              <span>
                Model:{" "}
                <strong>
                  {result.model?.name}
                </strong>
              </span>

              <span>
                Version:{" "}
                <strong>
                  {result.model?.version}
                </strong>
              </span>

              <span>
                Threshold:{" "}
                <strong>
                  {result.threshold}
                </strong>
              </span>

            </div>

          </section>
        )}


        <section className="metrics-grid">

          <MetricCard
            icon={TrendingUp}
            label="TOTAL ANALYSES"
            value={totalAnalyses}
            description="Stored threat assessments"
          />

          <MetricCard
            icon={ShieldAlert}
            label="THREATS DETECTED"
            value={threatsDetected}
            description="Phishing and scam classifications"
          />

          <MetricCard
            icon={AlertTriangle}
            label="CRITICAL"
            value={criticalCount}
            description="Highest risk assessments"
          />

          <MetricCard
            icon={CheckCircle2}
            label="ENGINE STATUS"
            value={
              apiOnline
                ? "Operational"
                : "Offline"
            }
            description="FastAPI inference service"
          />

        </section>


        <section className="charts-grid">

          <div className="dashboard-card">

            <div className="card-heading">

              <div>
                <h3>
                  Threat Distribution
                </h3>

                <p>
                  Classification history
                </p>
              </div>

            </div>


            {threatDistribution.length ? (
              <div className="chart-container">

                <ResponsiveContainer
                  width="100%"
                  height={240}
                >
                  <PieChart>

                    <Pie
                      data={
                        threatDistribution
                      }
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={82}
                      innerRadius={48}
                      paddingAngle={3}
                    >

                      {threatDistribution.map(
                        (_, index) => (
                          <Cell
                            key={index}
                          />
                        ),
                      )}

                    </Pie>

                    <Tooltip />

                  </PieChart>
                </ResponsiveContainer>

              </div>
            ) : (
              <div className="empty-chart">
                No analyses yet.
              </div>
            )}

          </div>


          <div className="dashboard-card">

            <div className="card-heading">

              <div>
                <h3>
                  Risk Distribution
                </h3>

                <p>
                  Severity across analyses
                </p>
              </div>

            </div>


            {riskDistribution.length ? (
              <div className="chart-container">

                <ResponsiveContainer
                  width="100%"
                  height={240}
                >
                  <PieChart>

                    <Pie
                      data={
                        riskDistribution
                      }
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={82}
                      innerRadius={48}
                      paddingAngle={3}
                    >

                      {riskDistribution.map(
                        (_, index) => (
                          <Cell
                            key={index}
                          />
                        ),
                      )}

                    </Pie>

                    <Tooltip />

                  </PieChart>
                </ResponsiveContainer>

              </div>
            ) : (
              <div className="empty-chart">
                No analyses yet.
              </div>
            )}

          </div>

        </section>


        <section className="dashboard-card history-card">

          <div className="card-heading">

            <div>
              <h3>
                Analysis History
              </h3>

              <p>
                Recent threat assessments
              </p>
            </div>

            <div className="history-count">
              {history.length} records
            </div>

          </div>


          {historyLoading ? (
            <div className="history-empty">
              Loading history...
            </div>
          ) : history.length === 0 ? (
            <div className="history-empty">
              <Clock3 size={24} />
              No analyses have been recorded yet.
            </div>
          ) : (
            <div className="history-table-wrap">

              <table className="history-table">

                <thead>
                  <tr>
                    <th>TYPE</th>
                    <th>INPUT</th>
                    <th>RESULT</th>
                    <th>RISK</th>
                    <th>SCORE</th>
                    <th>TIME</th>
                  </tr>
                </thead>


                <tbody>

                  {history.map(
                    (item) => (
                      <tr key={item.id}>

                        <td>

                          <span className="type-chip">

                            {item.input_type ===
                            "url" ? (
                              <Link2 size={13} />
                            ) : (
                              <MessageSquareText
                                size={13}
                              />
                            )}

                            {item.input_type}

                          </span>

                        </td>


                        <td
                          className="history-input"
                          title={
                            item.input_value
                          }
                        >
                          {item.input_value}
                        </td>


                        <td>

                          <span
                            className={
                              item.prediction ===
                              "legitimate"
                                ? "result-positive"
                                : "result-negative"
                            }
                          >
                            {item.prediction}
                          </span>

                        </td>


                        <td>

                          <RiskBadge
                            band={
                              item.risk_band
                            }
                          />

                        </td>


                        <td>
                          <strong>
                            {item.risk_score}
                          </strong>
                        </td>


                        <td className="time-cell">
                          {new Date(
                            item.created_at,
                          ).toLocaleString()}
                        </td>

                      </tr>
                    ),
                  )}

                </tbody>

              </table>

            </div>
          )}

        </section>

      </main>


      <footer className="footer">

        <div>
          <Shield size={15} />
          ThreatGuard
        </div>

        <span>
          AI-assisted threat detection ·
          Risk signal, not proof of safety
        </span>

      </footer>

    </div>
  );
}


export default App;
