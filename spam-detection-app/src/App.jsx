import { useEffect, useState } from 'react'
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom'
import Header from './components/Header'
import Footer from './components/Footer'
import Home from './pages/Home'
import About from './pages/About'
import SpamDetector from './pages/SpamDetector'

function AppContent({ labMode, onToggleLabMode }) {
  const location = useLocation();
  const currentPath = location.pathname;

  return (
    <>
      <Header
        currentPath={currentPath}
        labMode={labMode}
        onToggleLabMode={onToggleLabMode}
      />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/spam-detector" element={<SpamDetector labMode={labMode} />} />
        <Route path="*" element={<Home />} />
      </Routes>
      <Footer />
    </>
  );
}

function App() {
  const [labMode, setLabMode] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.localStorage.getItem('labMode') === 'true';
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;

    window.localStorage.setItem('labMode', labMode ? 'true' : 'false');

    const root = document.documentElement;
    if (labMode) {
      root.setAttribute('data-mode', 'lab');
    } else {
      root.removeAttribute('data-mode');
    }
  }, [labMode]);

  const handleToggleLabMode = () => {
    setLabMode((prev) => !prev);
  };

  return (
    <Router>
      <AppContent
        labMode={labMode}
        onToggleLabMode={handleToggleLabMode}
      />
    </Router>
  );
}

export default App
