import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import "../templates/Homepage.css";
import video1 from "../assets/videos/video1.mp4";
import video2 from "../assets/videos/video2.mp4";
import video3 from "../assets/videos/video3.mp4";
import video4 from "../assets/videos/video4.mp4";
import video5 from "../assets/videos/video5.mp4";
import card1 from "../assets/cards/card1.jpg";
import card2 from "../assets/cards/card2.jpg";
import card3 from "../assets/cards/card3.jpg";
import card4 from "../assets/cards/card4.jpg";

// Array of local video files
const videoFiles = [video1, video2, video3, video4, video5];

export default function Homepage() {
  // refs for five video elements: index 0 = left large, 1-4 = small feeds
  const videoRefs = useRef([]);
  videoRefs.current = videoRefs.current.slice(0, 5);

  const [fullscreenIndex, setFullscreenIndex] = useState(null);

  // New state to hold annotated YOLO output for each video feed
  const [annotatedSrcs, setAnnotatedSrcs] = useState(Array(5).fill(null));

  // Helper to collect refs
  const setVideoRef = (el, idx) => {
    videoRefs.current[idx] = el;
  };

  // useEffect to load local videos
  useEffect(() => {
    videoRefs.current.forEach((v, idx) => {
      if (v) {
        v.src = videoFiles[idx];
        v.loop = true;
        v.muted = true;
        v.playsInline = true;
        v.autoPlay = true;
        v.play().catch(() => {});
      }
    });
  }, []); // Run once on mount

  // Periodically capture frames from all 5 videos and send to backend
  useEffect(() => {
    const fps = 1; // Capture every 1 second to reduce load
    let interval = null;

    // Function to capture and send a single frame
    const captureAndSendFrame = async (video, index) => {
      if (!video || video.readyState !== video.HAVE_ENOUGH_DATA) {
        return;
      }

      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(async (blob) => {
        if (!blob) return;

        const formData = new FormData();
        formData.append("file", blob, `frame_${index}.jpg`);
        formData.append("camera_index", index); // Send the camera index

        try {
          const res = await fetch("http://localhost:8000/predict", {
            method: "POST",
            body: formData,
          });
          const data = await res.json();
          // Update the annotated image source for the specific camera index
          if (data?.status === "ok" && data.annotated_image_b64) {
            setAnnotatedSrcs(prevSrcs => {
              const newSrcs = [...prevSrcs];
              newSrcs[index] = "data:image/jpeg;base64," + data.annotated_image_b64;
              return newSrcs;
            });
          }
        } catch (err) {
          console.error(`Prediction error for camera ${index}:`, err);
        }
      }, "image/jpeg");
    };

    // Start the interval after a brief delay to ensure videos are loaded
    const startInterval = () => {
      interval = setInterval(() => {
        videoRefs.current.forEach((video, index) => {
          captureAndSendFrame(video, index);
        });
      }, 1000 / fps);
    };

    startInterval();

    return () => clearInterval(interval);
  }, []); // Re-run effect if videoRefs.current changes

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

                {/* Overlay YOLO annotated image for the large video feed (index 0) */}
                {annotatedSrcs[0] && (
                  <img
                    src={annotatedSrcs[0]}
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

                        {/* Overlay YOLO annotated image for the specific small video feed */}
                        {annotatedSrcs[index] && (
                          <img
                            src={annotatedSrcs[index]}
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
              {[card1, card2, card3, card4].map((card, index) => (
                <div key={index} className="powerbi-card">
                  <img
                    src={card}
                    alt={`Power BI Card ${index + 1}`}
                    className="powerbi-card-image"
                  />
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
              <iframe
                title="Power_bi_file"
                width="600"
                height="373.5"
                src="https://app.powerbi.com/view?r=eyJrIjoiNjNmZmExZjItMmU5Yi00NzFjLTg4MTYtNjRjMWY2MWYzY2UwIiwidCI6ImEyNDVhYTNhLWU4YjAtNGIyMy05NmM1LTUyMjQyMjM2OGRjNCJ9&pageName=7b77ee0a590a7409dbc0"
                frameBorder="0"
                allowFullScreen
              ></iframe>
            </div>
          </div>

          <div className="analytics-right">
            <div className="section-header">
              <h3 className="section-title">Activity Timeline</h3>
            </div>
            <div className="analytics-placeholder">
              <iframe
                title="Power_bi_file"
                width="600"
                height="373.5"
                src="https://app.powerbi.com/view?r=eyJrIjoiNjNmZmExZjItMmU5Yi00NzFjLTg4MTYtNjRjMWY2MWYzY2UwIiwidCI6ImEyNDVhYTNhLWU4YjAtNGIyMy05NmM1LTUyMjQyMjM2OGRjNCJ9&pageName=9d0f754aa63b331842d5"
                frameBorder="0"
                allowFullScreen
              ></iframe>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}