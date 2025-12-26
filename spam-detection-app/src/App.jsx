import { useEffect, useState } from 'react'
import { BrowserRouter as Router, Routes, Route, useLocation, Navigate } from 'react-router-dom'
import Header from './components/Header'
import Footer from './components/Footer'
import Home from './pages/Home'
import About from './pages/About'
import SpamDetector from './pages/SpamDetector'
import Vulnerabilities from './pages/Vulnerabilities'
import Login from './pages/Login'

function AppContent({ labMode, onToggleLabMode, isAuthenticated, username, onLogin, onLogout, csrfToken, onRefreshCsrf }) {
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
        csrfToken={csrfToken}
      />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/spam-detector" element={<SpamDetector labMode={labMode} csrfToken={csrfToken} onRefreshCsrf={onRefreshCsrf} />} />
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

  // CSRF token state (stored in memory, not localStorage for security)
  const [csrfToken, setCsrfToken] = useState(null);

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
          // Store CSRF token from session response
          if (data.csrf_token) {
            setCsrfToken(data.csrf_token);
            console.log('✅ CSRF Token fetched and stored:', data.csrf_token.substring(0, 20) + '...');
          } else {
            console.warn('⚠️ No CSRF token in session response:', data);
            // Don't clear auth, just warn - token might be generated on next request
          }
        } else if (response.status === 401) {
          // Only clear auth on 401 (unauthorized), not on other errors
          console.warn('Session check returned 401 - token invalid');
          localStorage.removeItem('auth_token');
          localStorage.removeItem('username');
          setIsAuthenticated(false);
          setUsername(null);
          setCsrfToken(null);
          if (labMode) {
            setLabMode(false);
          }
        } else {
          // Other errors (500, network, etc.) - don't log out, just warn
          console.warn('Session check failed with status:', response.status, '- keeping session');
          // Keep existing auth state, just try to refresh CSRF token
          if (response.ok === false && response.status !== 401) {
            // Try to get CSRF token anyway if we have a username
            const storedUsername = localStorage.getItem('username');
            if (storedUsername) {
              setIsAuthenticated(true);
              setUsername(storedUsername);
            }
          }
        }
      } catch (err) {
        // Network errors - don't log out, just warn
        console.warn('Auth check network error (keeping session):', err);
        // Keep existing auth state if we have a token
        const storedUsername = localStorage.getItem('username');
        if (storedUsername && localStorage.getItem('auth_token')) {
          setIsAuthenticated(true);
          setUsername(storedUsername);
          // Try to fetch CSRF token in background
          fetch('http://localhost:8000/auth/session', {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            },
          })
          .then(r => r.ok ? r.json() : null)
          .then(data => {
            if (data && data.csrf_token) {
              setCsrfToken(data.csrf_token);
            }
          })
          .catch(() => {}); // Silent fail
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

  const handleLogin = async (user, token, csrfTokenFromLogin = null) => {
    setIsAuthenticated(true);
    setUsername(user);
    localStorage.setItem('auth_token', token);
    localStorage.setItem('username', user);
    
    // Use CSRF token from login response if provided, otherwise fetch it
    if (csrfTokenFromLogin) {
      setCsrfToken(csrfTokenFromLogin);
      console.log('✅ CSRF Token received from login response:', csrfTokenFromLogin.substring(0, 20) + '...');
    } else {
      // Fallback: Fetch CSRF token after login if not in response
      console.warn('⚠️ No CSRF token in login response, fetching from session...');
      try {
        const response = await fetch('http://localhost:8000/auth/session', {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
        if (response.ok) {
          const data = await response.json();
          if (data.csrf_token) {
            setCsrfToken(data.csrf_token);
            console.log('✅ CSRF Token fetched after login:', data.csrf_token.substring(0, 20) + '...');
          } else {
            console.warn('⚠️ No CSRF token in session response after login:', data);
            // Try to get CSRF token from dedicated endpoint
            try {
              const csrfResponse = await fetch('http://localhost:8000/auth/csrf', {
                headers: {
                  'Authorization': `Bearer ${token}`,
                },
              });
              if (csrfResponse.ok) {
                const csrfData = await csrfResponse.json();
                if (csrfData.csrf_token) {
                  setCsrfToken(csrfData.csrf_token);
                  console.log('✅ CSRF Token fetched from /auth/csrf endpoint');
                }
              }
            } catch (csrfErr) {
              console.warn('Failed to fetch CSRF from dedicated endpoint:', csrfErr);
            }
          }
        } else {
          console.error('Session check failed after login:', response.status);
        }
      } catch (err) {
        console.error('Failed to fetch CSRF token after login:', err);
        // Don't block login, but try again in background
        setTimeout(async () => {
          try {
            const retryResponse = await fetch('http://localhost:8000/auth/session', {
              headers: {
                'Authorization': `Bearer ${token}`,
              },
            });
            if (retryResponse.ok) {
              const data = await retryResponse.json();
              if (data.csrf_token) {
                setCsrfToken(data.csrf_token);
                console.log('✅ CSRF Token fetched on retry');
              }
            }
          } catch (retryErr) {
            console.warn('CSRF token retry failed:', retryErr);
          }
        }, 1000);
      }
    }
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUsername(null);
    setCsrfToken(null);
    localStorage.removeItem('auth_token');
    localStorage.removeItem('username');
    // Disable lab mode on logout
    if (labMode) {
      setLabMode(false);
    }
  };

  const handleRefreshCsrf = async () => {
    const token = localStorage.getItem('auth_token');
    if (!token) return false;
    
    try {
      const response = await fetch('http://localhost:8000/auth/session', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        if (data.csrf_token) {
          setCsrfToken(data.csrf_token);
          console.log('✅ CSRF Token refreshed');
          return true;
        }
      }
    } catch (err) {
      console.error('Failed to refresh CSRF token:', err);
    }
    return false;
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
        csrfToken={csrfToken}
        onRefreshCsrf={handleRefreshCsrf}
      />
    </Router>
  );
}

export default App
