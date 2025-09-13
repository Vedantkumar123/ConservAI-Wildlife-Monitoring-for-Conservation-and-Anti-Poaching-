import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import "../templates/Homepage.css";

// Import your five video files
import videoSource1 from "D:/WORK/project 1/WebUI/wss/src/assets/videos/video1.mp4";
import videoSource2 from "D:/WORK/project 1/WebUI/wss/src/assets/videos/video2.mp4";
import videoSource3 from "D:/WORK/project 1/WebUI/wss/src/assets/videos/video3.mp4";
import videoSource4 from "D:/WORK/project 1/WebUI/wss/src/assets/videos/video4.mp4";
import videoSource5 from "D:/WORK/project 1/WebUI/wss/src/assets/videos/video5.mp4";
// The URL for your backend prediction service
const PREDICTION_API_URL = "http://localhost:8000/predict";

export default function Homepage() {
  const videoRefs = useRef([]);
  videoRefs.current = videoRefs.current.slice(0, 5);
  const workersRef = useRef([]);

  const [fullscreenIndex, setFullscreenIndex] = useState(null);
  const [annotatedSrcs, setAnnotatedSrcs] = useState({});

  const videoSources = [videoSource1, videoSource2, videoSource3, videoSource4, videoSource5];

  const setVideoRef = (el, idx) => {
    videoRefs.current[idx] = el;
  };

  useEffect(() => {
    // Initialize one worker per video source
    workersRef.current = videoSources.map(() => new Worker('/detection.worker.js'));

    workersRef.current.forEach((worker, index) => {
      worker.onmessage = (event) => {
        const { annotatedSrc, index: workerIndex } = event.data;
        setAnnotatedSrcs(prevSrcs => ({
          ...prevSrcs,
          [workerIndex]: annotatedSrc,
        }));
      };
    });

    return () => {
      workersRef.current.forEach(worker => worker.terminate());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const fps = 1;
    const interval = setInterval(() => {
      videoRefs.current.forEach(async (video, index) => {
        if (video && !video.paused && video.readyState >= 3) {
          try {
            const imageBitmap = await createImageBitmap(video);
            workersRef.current[index].postMessage(
              { imageData: imageBitmap, index: index, apiUrl: PREDICTION_API_URL },
              [imageBitmap]
            );
          } catch (err) {
            console.error(`Error processing frame for video ${index}:`, err);
          }
        }
      });
    }, 1000 / fps);
    return () => clearInterval(interval);
  }, []);

  // --- Fullscreen logic (unchanged) ---
  useEffect(() => {
    function onFullScreenChange() {
      const elem = document.fullscreenElement || document.webkitFullscreenElement;
      if (!elem) {
        setFullscreenIndex(null);
        videoRefs.current.forEach((v) => v?.play().catch(() => {}));
        document.body.style.overflow = "";
      } else {
        const foundIndex = videoRefs.current.findIndex((v) => v && elem.contains(v));
        if (foundIndex >= 0) {
          setFullscreenIndex(foundIndex);
          videoRefs.current.forEach((v) => v?.play().catch(() => {}));
          document.body.style.overflow = "hidden";
        } else {
          setFullscreenIndex(null);
        }
      }
    }
    document.addEventListener("fullscreenchange", onFullScreenChange);
    document.addEventListener("webkitfullscreenchange", onFullScreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", onFullScreenChange);
      document.removeEventListener("webkitfullscreenchange", onFullScreenChange);
    };
  }, []);
  const enterFullscreen = async (idx) => {
    const v = videoRefs.current[idx];
    const wrapper = v?.parentElement;
    if (!wrapper) return;
    try {
      if (wrapper.requestFullscreen) await wrapper.requestFullscreen();
      else if (wrapper.webkitRequestFullscreen) wrapper.webkitRequestFullscreen();
      videoRefs.current.forEach((ov) => ov?.play().catch(() => {}));
    } catch (err) { console.error("Fullscreen request failed:", err); }
  };
  const exitFullscreen = async () => {
    try {
      if (document.fullscreenElement || document.webkitFullscreenElement) {
        if (document.exitFullscreen) await document.exitFullscreen();
        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
      }
    } catch (err) { console.error("Error exiting fullscreen:", err); }
  };

  return (
    <div className="homepage-root">
      <div className="page-top">
        <nav className="top-navbar">
          <h1 className="brand">Wildlife Surveillance System</h1>
          <div className="nav-actions">
            <Link to="/detection-trends" className="btn">
              Detection Trends
            </Link>
            <Link to="/faq" className="btn outline">
              FAQ
            </Link>
          </div>
        </nav>
        <main className="main-content">
          <section className="camera-section">
            <div className="camera-left card">
              <div className="camera-label">Camera Feed</div>
              <div className="video-wrapper" onClick={() => enterFullscreen(0)}>
                <video
                  ref={(el) => setVideoRef(el, 0)}
                  src={videoSources[0]}
                  className="video-large clickable"
                  muted playsInline autoPlay loop
                />
                {annotatedSrcs[0] && (
                  <img src={annotatedSrcs[0]} alt="annotated" className="annotation-overlay" />
                )}
                <button
                  className="close-btn" onClick={(e) => { e.stopPropagation(); exitFullscreen(); }}
                  aria-hidden={fullscreenIndex !== 0} title="Close fullscreen"
                >✕</button>
              </div>
            </div>
            <div className="camera-right">
              <div className="grid-2x2">
                {videoSources.slice(1).map((videoSrc, i) => {
                  const index = i + 1;
                  return (
                    <div key={index} className="small-feed card">
                      <div className="video-wrapper" onClick={() => enterFullscreen(index)}>
                        <video
                          ref={(el) => setVideoRef(el, index)}
                          src={videoSrc} className="video-small clickable"
                          muted playsInline autoPlay loop
                        />
                        {annotatedSrcs[index] && (
                          <img src={annotatedSrcs[index]} alt="annotated" className="annotation-overlay" />
                        )}
                        <button
                          className="close-btn" onClick={(e) => { e.stopPropagation(); exitFullscreen(); }}
                          aria-hidden={fullscreenIndex !== index} title="Close fullscreen"
                        >✕</button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </section>
          <section className="powerbi-section">
            <div className="powerbi-grid">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="powerbi-card">
                  <div className="card-placeholder">Power BI Card {i}</div>
                </div>
              ))}
            </div>
          </section>
        </main>
      </div>
      <div className="page-bottom">
        <div className="analytics-section card">
          <div className="analytics-left">
            <div className="section-header">
              <h3 className="section-title">Species Distribution</h3>
              <div className="btn-group">
                <button className="small-pill">Daily</button>
                <button className="small-pill">Weekly</button>
                <button className="small-pill">Monthly</button>
              </div>
            </div>
            <div className="analytics-placeholder">
              <div className="powerbi-placeholder">Species distribution Power BI / Chart placeholder</div>
            </div>
          </div>
          <div className="analytics-right">
            <div className="section-header">
              <h3 className="section-title">Activity Timeline</h3>
            </div>
            <div className="analytics-placeholder">
              <div className="powerbi-placeholder">Activity timeline Power BI / Chart placeholder</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}