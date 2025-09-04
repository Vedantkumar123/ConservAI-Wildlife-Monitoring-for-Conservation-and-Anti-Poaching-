import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Homepage from "./components/Homepage";
import DetectionTrends from "./components/DetectionTrends";
import Faq from "./components/Faq";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Homepage />} />
        <Route path="/detection-trends" element={<DetectionTrends />} />
        <Route path="/faq" element={<Faq />} />
      </Routes>
    </Router>
  );
}