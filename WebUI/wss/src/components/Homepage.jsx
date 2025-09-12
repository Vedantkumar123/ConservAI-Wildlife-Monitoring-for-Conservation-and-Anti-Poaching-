import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import "../templates/Homepage.css";


export default function Homepage() {
  // refs for five video elements: index 0 = left large, 1-4 = small feeds
  const videoRefs = useRef([]);
  videoRefs.current = videoRefs.current.slice(0, 5);

  const [webcamStream, setWebcamStream] = useState(null);
  const [error, setError] = useState(null);
  const [fullscreenIndex, setFullscreenIndex] = useState(null);

  // ✅ New state to hold annotated YOLO output
  const [annotatedSrc, setAnnotatedSrc] = useState(null);

  // helper to collect refs
  const setVideoRef = (el, idx) => {
    videoRefs.current[idx] = el;
  };

  // Request webcam stream once on mount
  useEffect(() => {
    let mounted = true;
    async function startCamera() {
      try {
        const s = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
          audio: false,
        });
        if (!mounted) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        setWebcamStream(s);

        // Attach stream to any mounted video refs
        videoRefs.current.forEach((v) => {
          if (v) {
            v.srcObject = s;
            v.play().catch(() => {});
          }
        });
      } catch (err) {
        console.error("getUserMedia error:", err);
        setError("Unable to access webcam. Check permissions or device.");
      }
    }
    startCamera();

    return () => {
      mounted = false;
      if (webcamStream) {
        webcamStream.getTracks().forEach((t) => t.stop());
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // run once

  // ensure newly-mounted video elements get the stream
  useEffect(() => {
    if (!webcamStream) return;
    videoRefs.current.forEach((v) => {
      if (v && v.srcObject !== webcamStream) {
        v.srcObject = webcamStream;
        v.play().catch(() => {});
      }
    });
  }, [webcamStream]);


  // ✅ Periodically capture frames from the main video and send to backend
  useEffect(() => {
    const video = videoRefs.current[0]; // primary video element
    if (!video) return;

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const fps = 1; // capture every 1s to reduce load

    const interval = setInterval(async () => {
      if (video.readyState === video.HAVE_ENOUGH_DATA) {
        // draw video frame onto canvas
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // convert to blob
        canvas.toBlob(async (blob) => {
          if (!blob) return;
          const formData = new FormData();
          formData.append("file", blob, "frame.jpg");

          try {
            const res = await fetch("http://localhost:8000/predict", {
              method: "POST",
              body: formData,
            });
            const data = await res.json();
            if (data?.status === "ok" && data.annotated_image_b64) {
              setAnnotatedSrc("data:image/jpeg;base64," + data.annotated_image_b64);
            }
          } catch (err) {
            console.error("Prediction error:", err);
          }
        }, "image/jpeg");
      }
    }, 1000 / fps);

    return () => clearInterval(interval);
  }, []);

  // Fullscreen change handler to detect enter/exit
  useEffect(() => {
    function onFullScreenChange() {
      const elem = document.fullscreenElement || document.webkitFullscreenElement;
      if (!elem) {
        // Exited fullscreen
        setFullscreenIndex(null);
        // Ensure all video elements resume playing after exit
        videoRefs.current.forEach((v) => {
          if (v) {
            v.play().catch(() => {});
          }
        });
        document.body.style.overflow = "";
      } else {
        // Find which video index is contained in the fullscreen element
        const foundIndex = videoRefs.current.findIndex((v) => v && elem.contains(v));
        if (foundIndex >= 0) {
          setFullscreenIndex(foundIndex);
          // Make sure the fullscreen video plays
          const v = videoRefs.current[foundIndex];
          if (v) v.play().catch(() => {});
          // Also keep other videos playing as well (some browsers may pause)
          videoRefs.current.forEach((otherV) => {
            if (otherV && otherV !== v) {
              otherV.play().catch(() => {});
            }
          });
          document.body.style.overflow = "hidden";
        } else {
          // Fullscreen element is not a video wrapper we control
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

  // Enter fullscreen for the clicked feed (use its wrapper so the close button is visible)
  const enterFullscreen = async (idx) => {
    const v = videoRefs.current[idx];
    if (!v) return;
    // use the video wrapper (parentElement) for fullscreen so close button is included
    const wrapper = v.parentElement;
    if (!wrapper) return;

    try {
      // requestFullscreen on wrapper (user gesture from click)
      if (wrapper.requestFullscreen) {
        await wrapper.requestFullscreen();
      } else if (wrapper.webkitRequestFullscreen) {
        // Safari/older webkit
        wrapper.webkitRequestFullscreen();
      } else if (v.requestFullscreen) {
        // fallback to video element itself
        await v.requestFullscreen();
      }
      // play to ensure playback continues
      v.play().catch(() => {});
      // ensure others play too
      videoRefs.current.forEach((ov) => {
        if (ov && ov !== v) ov.play().catch(() => {});
      });
    } catch (err) {
      console.error("Fullscreen request failed:", err);
    }
  };

  // Exit fullscreen
  const exitFullscreen = async () => {
    try {
      if (document.fullscreenElement || document.webkitFullscreenElement) {
        if (document.exitFullscreen) {
          await document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
          document.webkitExitFullscreen();
        }
      }
    } catch (err) {
      console.error("Error exiting fullscreen:", err);
    }
  };

  return (
    <div className="homepage-root">
      {/* Top section: cameras + powerbi cards -> occupies 100vh */}
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
                  className="video-large clickable"
                  muted
                  playsInline
                  autoPlay
                  aria-label="Primary camera feed - click to fullscreen"
                />

                {/* ✅ Overlay YOLO annotated image */}
                {annotatedSrc && (
                  <img
                    src={annotatedSrc}
                    alt="annotated"
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      height: "100%",
                      pointerEvents: "none",
                    }}
                  />
                )}

                <button
                  className="close-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    exitFullscreen();
                  }}
                  aria-hidden={fullscreenIndex !== 0 ? "true" : "false"}
                  title="Close fullscreen"
                >
                  ✕
                </button>
              </div>

              {!webcamStream && !error && (
                <div className="video-fallback">Starting camera...</div>
              )}
              {error && <div className="video-error">{error}</div>}
            </div>

            <div className="camera-right">
              <div className="grid-2x2">
                {[1, 2, 3, 4].map((i) => {
                  const index = i; // map 1..4 to indexes 1..4
                  return (
                    <div key={i} className="small-feed card">
                      <div className="video-wrapper" onClick={() => enterFullscreen(index)}>
                        <video
                          ref={(el) => setVideoRef(el, index)}
                          className="video-small clickable"
                          muted
                          playsInline
                          autoPlay
                          aria-label={`Camera feed ${index} - click to fullscreen`}
                        />

                        {/* ✅ Overlay YOLO annotated image */}
                        {annotatedSrc && (
                          <img
                            src={annotatedSrc}
                            alt="annotated"
                            style={{
                              position: "absolute",
                              top: 0,
                              left: 0,
                              width: "100%",
                              height: "100%",
                              pointerEvents: "none",
                            }}
                          />
                        )}

                        <button
                          className="close-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            exitFullscreen();
                          }}
                          aria-hidden={fullscreenIndex !== index ? "true" : "false"}
                          title="Close fullscreen"
                        >
                          ✕
                        </button>
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

      {/* Bottom section: analytics area -> occupies next 100vh (makes total ~200vh) */}
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
              {/* Space reserved for Power BI / chart integration */}
              <div className="powerbi-placeholder">Species distribution Power BI / Chart placeholder</div>
            </div>
          </div>

          <div className="analytics-right">
            <div className="section-header">
              <h3 className="section-title">Activity Timeline</h3>
            </div>

            <div className="analytics-placeholder">
              {/* Space reserved for Power BI / timeline integration */}
              <div className="powerbi-placeholder">Activity timeline Power BI / Chart placeholder</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}