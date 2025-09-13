// This script runs in the background and will handle API calls for one video stream.

self.onmessage = async (event) => {
  const { imageData, index, apiUrl } = event.data;

  // Create a canvas from the image data to convert it to a blob
  const canvas = new OffscreenCanvas(imageData.width, imageData.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imageData, 0, 0);

  // Convert canvas to a JPEG blob
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.85 });

  const formData = new FormData();
  formData.append('file', blob, 'frame.jpg');

  try {
    const response = await fetch(apiUrl, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    if (data?.status === 'ok' && data.annotated_image_b64) {
      const annotatedSrc = 'data:image/jpeg;base64,' + data.annotated_image_b64;
      // Send the result and the original index back to the main thread
      self.postMessage({ annotatedSrc, index });
    }
  } catch (err) {
    // In a real app, you might want to send error messages back too
    console.error('Worker prediction error:', err);
  }
};