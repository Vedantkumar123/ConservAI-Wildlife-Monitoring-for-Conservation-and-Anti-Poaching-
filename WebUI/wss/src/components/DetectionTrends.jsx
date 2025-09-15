import React, { useEffect, useState } from 'react';
import { Link } from "react-router-dom";
import '../templates/DetectionTrends.css';

// Utility: get current date + time in "YYYY-MM-DD HH:mm:ss"
function getCurrentDateTime() {
  const now = new Date();
  const date = now.toISOString().split("T")[0]; // "YYYY-MM-DD"
  const time = now.toLocaleTimeString('en-GB', { hour12: false }); // "HH:MM:SS"
  return `${date} ${time}`;
}

export default function DetectionTrends() {
  const [alerts, setAlerts] = useState(() => {
    const saved = localStorage.getItem("alerts");
    return saved ? JSON.parse(saved).slice(0, 3) : [];
  });

  const [detections, setDetections] = useState(() => {
    const saved = localStorage.getItem("detections");
    return saved ? JSON.parse(saved).slice(-5) : [];
  });

  // Save alerts & detections in localStorage whenever they change
  useEffect(() => {
    localStorage.setItem("alerts", JSON.stringify(alerts));
  }, [alerts]);

  useEffect(() => {
    localStorage.setItem("detections", JSON.stringify(detections));
  }, [detections]);

  // Poll for new detections every 2 seconds
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:8000/predict-latest");
        if (!res.ok) {
          throw new Error('Failed to fetch latest detections');
        }
        const data = await res.json();

        // Process new detections from the backend
        if (data.status === "ok" && data.detections.length > 0) {
          const now = getCurrentDateTime();
          const newRows = [];

          // Group detections by species (from label) and camera
          const grouped = {};
          data.detections.forEach(det => {
            const key = `${det.label}-${det.camera}`;
            if (!grouped[key]) {
              grouped[key] = {
                time: now,
                camera: `Camera ${det.camera}`,
                species: det.label,
                count: 0,
                confidence: Math.round(det.confidence * 100),
              };
            }
            grouped[key].count += 1;
          });

          // Add new rows only if they're different from the last entry
          Object.values(grouped).forEach(entry => {
            const lastRow = detections[detections.length - 1];
            if (
              !lastRow ||
              lastRow.species !== entry.species ||
              lastRow.camera !== entry.camera ||
              lastRow.count !== entry.count ||
              lastRow.confidence !== entry.confidence
            ) {
              newRows.push(entry);

              // If a poacher is detected, create and push an alert
              if (entry.species.toLowerCase() === "poacher") {
                setAlerts(prev => {
                  const updated = [
                    {
                      title: "🚨 Poacher Detected!",
                      subtitle: `${entry.camera}`,
                      time: now
                    },
                    ...prev
                  ];
                  return updated.slice(0, 3); // keep only last 3 alerts
                });
              }
            }
          });

          if (newRows.length > 0) {
            setDetections(prev => {
              const updated = [...prev, ...newRows];
              return updated.slice(-5); // keep only last 5 detections
            });
          }
        }
      } catch (err) {
        console.error("Detection poll error:", err);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [detections]);

  return (
    <div className="detection-trends-root">
      {/* Navbar */}
      <div className='navbar'>
        <Link to="/" className="btn">Home</Link>
      </div>

      {/* Top: 40vh */}
      <div className="dt-top">
        <div className="dt-left-card">
          <div className="card-header">
            <h2 className="card-title">Detection Trends</h2>
            <div className="card-sub">Daily detections overview</div>
          </div>
          <div className="powerbi-embed chart-placeholder">
            <div className="placeholder-msg">Chart / Power BI embed placeholder</div>
          </div>
        </div>

        <div className="dt-right-card">
          <div className="card-header">
            <h3 className="card-title small">Recent Alerts</h3>
            <div className="card-sub">Latest activity</div>
          </div>

          <div className="alerts-list">
            {alerts.length === 0 && (
              <div className="placeholder-msg">No alerts yet</div>
            )}
            {alerts.slice(0, 3).map((a, idx) => (
              <div className="alert-item" key={idx}>
                <div className="alert-side" aria-hidden="true" />
                <div className="alert-content">
                  <div className="alert-title">{a.title}</div>
                  <div className="alert-sub">{a.subtitle}</div>
                </div>
                <div className="alert-time">{a.time}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom: 60vh */}
      <div className="dt-bottom">
        <div className="card-header card-header-bottom">
          <h2 className="card-title">Recent Detection</h2>
          <div className="card-sub">Latest detections list / Power BI integration placeholder</div>
        </div>

        <div className="recent-detections-card">
          <div className="table-wrapper">
            <table className="detections-table" role="table" aria-label="Recent detections table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Camera</th>
                  <th>Species</th>
                  <th>Count</th>
                  <th>Confidence</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {detections.length === 0 && (
                  <tr>
                    <td colSpan="6" className="placeholder-msg">No detections yet</td>
                  </tr>
                )}
                {detections.map((d, idx) => (
                  <tr key={idx}>
                    <td>{d.time}</td>
                    <td>{d.camera}</td>
                    <td><span className="species-pill">{d.species}</span></td>
                    <td>{d.count}</td>
                    <td>
                      <div className="conf-bar">
                        <div className="conf-fill" style={{ width: `${d.confidence}%` }} />
                        <span className="conf-text">{d.confidence}%</span>
                      </div>
                    </td>
                    <td><button className="view-btn">View</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}