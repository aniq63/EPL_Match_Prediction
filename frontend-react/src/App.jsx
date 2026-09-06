import { useEffect } from "react";
import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar.jsx";
import Footer from "./components/Footer.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Analytics from "./pages/Analytics.jsx";
import Predictions from "./pages/Predictions.jsx";
import History from "./pages/History.jsx";
import { api } from "./api.js";

export default function App() {
  useEffect(() => {
    const prefetchTimer = window.setTimeout(() => {
      api.getAnalytics().catch(() => {});
      api.getPredictions().catch(() => {});
    }, 1000);

    return () => window.clearTimeout(prefetchTimer);
  }, []);

  return (
    <div className="app-shell">
      <Navbar />
      <main style={{ flex: 1 }}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}
