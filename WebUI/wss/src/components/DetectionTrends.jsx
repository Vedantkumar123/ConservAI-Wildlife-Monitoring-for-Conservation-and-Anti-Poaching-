import React from 'react';
import { Link } from "react-router-dom";
import '../templates/DetectionTrends.css';

const sampleAlerts = [
  { id: 1, title: 'New species detected', subtitle: 'Trail cam - Camera 3', time: '10 minutes ago' },
  { id: 2, title: 'High activity detected', subtitle: 'Water Point - Camera 2', time: '30 minutes ago' },
  { id: 3, title: 'Camera maintenance', subtitle: 'Camera 1 - Connection lost', time: '2 hours ago' },
];

const sampleDetections = [
  { time: '01:15:08', camera: 'Camera 3 - East Plains', species: 'Leopard', count: 1, confidence: 86 },
  { time: '01:07:27', camera: 'Camera 2 - Water Point', species: 'Zebra', count: 1, confidence: 82 },
  { time: '23:33:40', camera: 'Camera 2 - Water Point', species: 'Zebra', count: 4, confidence: 81 },
  { time: '22:33:47', camera: 'Camera 4 - South Ridge', species: 'Red Fox', count: 2, confidence: 84 },
];

export default function DetectionTrends() {
  return (
    <div className="detection-trends-root">
      {/* Top: 40vh */}


      <div className='navbar'><Link to="/" className="btn">Home</Link></div>


      <div className="dt-top">
        <div className="dt-left-card">
          <div className="card-header">
            <h2 className="card-title">Detection Trends</h2>
            <div className="card-sub">Daily detections overview</div>
          </div>

          {/* Chart / Power BI placeholder */}
          <div className="powerbi-embed chart-placeholder" role="region" aria-label="Detection trends chart placeholder">
            {/* Replace this div's contents with Power BI embed or chart component */}
            <div className="placeholder-msg">Chart / Power BI embed placeholder</div>
          </div>
        </div>

        <div className="dt-right-card">
          <div className="card-header">
            <h3 className="card-title small">Recent Alerts</h3>
            <div className="card-sub">Latest activity</div>
          </div>

          <div className="alerts-list">
            {sampleAlerts.map((a) => (
              <div className="alert-item" key={a.id}>
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
          {/* Optional: if you want to embed Power BI instead of table, replace this table area */}
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
                {sampleDetections.map((d, idx) => (
                  <tr key={idx}>
                    <td>{d.time}</td>
                    <td>{d.camera}</td>
                    <td>
                      <span className="species-pill">{d.species}</span>
                    </td>
                    <td>{d.count}</td>
                    <td>
                      <div className="conf-bar">
                        <div className="conf-fill" style={{ width: `${d.confidence}%` }} />
                        <span className="conf-text">{d.confidence}%</span>
                      </div>
                    </td>
                    <td>
                      <button className="view-btn">View</button>
                    </td>
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