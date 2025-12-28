import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/login.css';

function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [testLabMode, setTestLabMode] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      // Use testLabMode state for testing SQL injection vulnerability
      // This allows testing Lab Mode behavior without being logged in
      const labMode = testLabMode;
      
      const response = await fetch('http://localhost:8000/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Lab-Mode': labMode ? 'true' : 'false',
        },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        const errorMessage = errorData.detail || 'Login failed';
        
        // Handle rate limiting (429 Too Many Requests)
        if (response.status === 429) {
          throw new Error(errorMessage);
        }
        
        throw new Error(errorMessage);
      }

      const data = await response.json();
      
      // Store token in localStorage
      localStorage.setItem('auth_token', data.access_token);
      localStorage.setItem('username', data.username);
      
      // Call parent callback with CSRF token if available
      if (onLogin) {
        onLogin(data.username, data.access_token, data.csrf_token);
      }
      
      // Redirect to home
      navigate('/');
    } catch (err) {
      setError(err.message || 'Login failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <h1>HackTheStack</h1>
        <h2>Login to Access Lab Mode</h2>
        <p className="login-subtitle">
          Lab Mode contains intentionally vulnerable features for educational purposes.
          Please log in with your credentials to access it.
        </p>
        
        <form onSubmit={handleSubmit} className="login-form">
          {error && <div className="error-message">{error}</div>}
          
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              autoComplete="username"
              disabled={loading}
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <div className="password-input-wrapper">
              <input
                type={showPassword ? "text" : "password"}
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                disabled={loading}
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword(!showPassword)}
                disabled={loading}
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? "👁️" : "👁️‍🗨️"}
              </button>
            </div>
          </div>
          
          <button 
            type="submit" 
            className="login-button"
            disabled={loading}
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>
        
        <div className="login-mode-toggle">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={testLabMode}
              onChange={(e) => setTestLabMode(e.target.checked)}
              disabled={loading}
            />
            <span className="toggle-text">
              Test in Lab Mode (SQL Injection & Weak Auth vulnerable)
            </span>
          </label>
          {testLabMode && (
            <div className="lab-mode-warning">
              ⚠️ Lab Mode: Login has multiple vulnerabilities enabled
              <br />
              <strong style={{ marginTop: '0.5rem', display: 'block' }}>Vulnerabilities:</strong>
              <ul style={{ margin: '0.5rem 0', paddingLeft: '1.5rem' }}>
                <li><strong>SQL Injection:</strong> Username/password fields vulnerable to SQL injection</li>
                <li><strong>Weak Authentication:</strong> No rate limiting - unlimited failed login attempts allowed</li>
              </ul>
              <strong style={{ marginTop: '0.5rem', display: 'block' }}>SQL Injection payloads:</strong>
              <ul style={{ margin: '0.5rem 0', paddingLeft: '1.5rem' }}>
                <li>Username: <code>nevan' OR '1'='1' --</code> | Password: <code>nevan</code> (username injection)</li>
                <li>Username: <code>nevan</code> | Password: <code>' OR '1'='1' --</code> (password injection)</li>
                <li>Username: <code>' OR '1'='1</code> | Password: <code>nevan</code> (returns first user)</li>
              </ul>
              <small style={{ marginTop: '0.5rem', display: 'block', fontStyle: 'italic' }}>
                In Secure Mode: After 3 failed attempts, login is disabled for 30 seconds. In Lab Mode, unlimited attempts are allowed.
              </small>
            </div>
          )}
        </div>
        
        <div className="login-info">
          <p><strong>Test Accounts:</strong></p>
          <ul>
            <li>Username: <code>nevan</code> | Password: <code>nevan</code></li>
            <li>Username: <code>naven</code> | Password: <code>naven</code></li>
          </ul>
          <p style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#888' }}>
            <strong>SQL Injection Test:</strong> In Lab Mode, try:
            <br />
            • Username: <code>nevan' OR '1'='1' --</code> with password <code>nevan</code> (username injection)
            <br />
            • Username: <code>nevan</code> with password <code>' OR '1'='1' --</code> (password injection)
            <br />
            <small style={{ display: 'block', marginTop: '0.25rem' }}>
              SQL injection can bypass authentication. Check backend logs to see the vulnerable query being executed.
            </small>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Login;

