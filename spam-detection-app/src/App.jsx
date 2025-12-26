import { useEffect, useState } from 'react'
import { BrowserRouter as Router, Routes, Route, useLocation, Navigate } from 'react-router-dom'
import Header from './components/Header'
import Footer from './components/Footer'
import Home from './pages/Home'
import About from './pages/About'
import SpamDetector from './pages/SpamDetector'
import Vulnerabilities from './pages/Vulnerabilities'
import Login from './pages/Login'

function AppContent({ labMode, onToggleLabMode, isAuthenticated, username, onLogin, onLogout }) {
  const location = useLocation();
  const currentPath = location.pathname;

  return (
    <>
      <Header
        currentPath={currentPath}
        labMode={labMode}
        onToggleLabMode={onToggleLabMode}
        isAuthenticated={isAuthenticated}
        username={username}
        onLogout={onLogout}
      />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/spam-detector" element={<SpamDetector labMode={labMode} />} />
        <Route 
          path="/login" 
          element={
            isAuthenticated ? (
              <Navigate to="/" replace />
            ) : (
              <Login onLogin={onLogin} />
            )
          } 
        />
        {labMode && <Route path="/vulnerabilities" element={<Vulnerabilities labMode={labMode} />} />}
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

  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    if (typeof window === 'undefined') return false;
    const token = localStorage.getItem('auth_token');
    return !!token;
  });

  const [username, setUsername] = useState(() => {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem('username') || null;
  });

  // Check authentication status on mount and when token changes
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('auth_token');
      if (!token) {
        setIsAuthenticated(false);
        setUsername(null);
        // Disable lab mode if not authenticated
        if (labMode) {
          setLabMode(false);
        }
        return;
      }

      try {
        const response = await fetch('http://localhost:8000/auth/session', {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const data = await response.json();
          setIsAuthenticated(true);
          setUsername(data.username);
        } else {
          // Token invalid, clear it
          localStorage.removeItem('auth_token');
          localStorage.removeItem('username');
          setIsAuthenticated(false);
          setUsername(null);
          if (labMode) {
            setLabMode(false);
          }
        }
      } catch (err) {
        console.error('Auth check failed:', err);
        localStorage.removeItem('auth_token');
        localStorage.removeItem('username');
        setIsAuthenticated(false);
        setUsername(null);
        if (labMode) {
          setLabMode(false);
        }
      }
    };

    checkAuth();
  }, []);

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

  const handleToggleLabMode = async () => {
    // If trying to enable lab mode, check authentication
    if (!labMode) {
      if (!isAuthenticated) {
        // Redirect to login if not authenticated
        window.location.href = '/login';
        return;
      }
      
      // Verify authentication before enabling lab mode
      const token = localStorage.getItem('auth_token');
      if (token) {
        try {
          const response = await fetch('http://localhost:8000/auth/check-lab-access', {
            headers: {
              'Authorization': `Bearer ${token}`,
            },
          });
          
          if (!response.ok) {
            alert('Authentication required to access Lab Mode. Please log in.');
            window.location.href = '/login';
            return;
          }
        } catch (err) {
          alert('Failed to verify authentication. Please log in again.');
          window.location.href = '/login';
          return;
        }
      } else {
        alert('Authentication required to access Lab Mode. Please log in.');
        window.location.href = '/login';
        return;
      }
    }
    
    setLabMode((prev) => !prev);
  };

  const handleLogin = (user, token) => {
    setIsAuthenticated(true);
    setUsername(user);
    localStorage.setItem('auth_token', token);
    localStorage.setItem('username', user);
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUsername(null);
    localStorage.removeItem('auth_token');
    localStorage.removeItem('username');
    // Disable lab mode on logout
    if (labMode) {
      setLabMode(false);
    }
  };

  return (
    <Router>
      <AppContent
        labMode={labMode}
        onToggleLabMode={handleToggleLabMode}
        isAuthenticated={isAuthenticated}
        username={username}
        onLogin={handleLogin}
        onLogout={handleLogout}
      />
    </Router>
  );
}

export default App
