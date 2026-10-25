// App.jsx — root component: routing + model-loaded state
import { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import Predict   from "./pages/Predict";
import Research  from "./pages/Research";
import { fetchHealth } from "./services/api";

export default function App() {
  const [modelLoaded, setModelLoaded] = useState(false);

  // Poll /api/health once on mount to set sidebar status
  useEffect(() => {
    fetchHealth()
      .then(d => setModelLoaded(d.model_loaded === true))
      .catch(() => setModelLoaded(false));
  }, []);

  return (
    <BrowserRouter>
      <div className="app-shell">
        <Navbar modelLoaded={modelLoaded} />
        <main className="main-content">
          <Routes>
            <Route path="/"         element={<Dashboard />} />
            <Route path="/predict"  element={<Predict />}   />
            <Route path="/research" element={<Research />}  />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
