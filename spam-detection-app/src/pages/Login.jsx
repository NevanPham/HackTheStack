import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/login.css';

function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Login failed');
      }

      const data = await response.json();
      
      // Store token in localStorage
      localStorage.setItem('auth_token', data.access_token);
      localStorage.setItem('username', data.username);
      
      // Call parent callback
      if (onLogin) {
        onLogin(data.username, data.access_token);
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
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete="current-password"
              disabled={loading}
            />
          </div>
          
          <button 
            type="submit" 
            className="login-button"
            disabled={loading}
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>
        
        <div className="login-info">
          <p><strong>Test Accounts:</strong></p>
          <ul>
            <li>Username: <code>nevan</code> | Password: <code>nevan</code></li>
            <li>Username: <code>naven</code> | Password: <code>naven</code></li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Login;

