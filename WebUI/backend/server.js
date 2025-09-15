// backend/server.js
require('dotenv').config();
const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');

const app = express();
const PORT = process.env.PORT || 5000;
const PY_SERVICE = process.env.PY_SERVICE_URL || 'http://localhost:8000';

// Increase payload limit (JSON arrays can be large)
app.use(bodyParser.json({ limit: '100mb' }));

// Optional health
app.get('/api/health', (req, res) => res.json({ status: 'ok' }));

// Forward frame JSON to Python model server
app.post('/api/predict', async (req, res) => {
  try {
    // req.body is expected: { width, height, data }
    const payload = req.body;

    // Forward to Python FastAPI
    const response = await axios.post(`${PY_SERVICE}/predict`, payload, {
      headers: { 'Content-Type': 'application/json' },
      timeout: 20000, // adjust
    });

    res.json(response.data);
  } catch (err) {
    console.error('Error forwarding to python service:', err.message || err);
    if (err.response && err.response.data) {
      res.status(500).json({ status: 'error', details: err.response.data });
    } else {
      res.status(500).json({ status: 'error', message: err.message });
    }
  }
});

app.listen(PORT, () => {
  console.log(`Express gateway listening on ${PORT}`);
});
