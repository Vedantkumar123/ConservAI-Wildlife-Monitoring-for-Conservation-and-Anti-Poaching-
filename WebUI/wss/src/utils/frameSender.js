// wss/src/utils/frameSender.js
export async function captureAndSendFrame(videoEl, apiUrl, callback) {
  if (!videoEl || videoEl.readyState < 2) return;

  // Draw video frame into canvas
  const canvas = document.createElement("canvas");
  canvas.width = videoEl.videoWidth;
  canvas.height = videoEl.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

  // Convert canvas to Blob (JPEG)
  const blob = await new Promise((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", 0.8)
  );

  // Prepare multipart/form-data
  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");

  try {
    const res = await fetch(apiUrl, {
      method: "POST",
      body: formData, // 👈 No need for headers, browser sets them
    });

    const json = await res.json();
    if (callback) callback(json);
  } catch (err) {
    console.error("Frame send error", err);
    if (callback) callback(null);
  }
}
