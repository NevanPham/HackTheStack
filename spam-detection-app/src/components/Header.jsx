import { Link } from 'react-router-dom';

function Header({ currentPath, labMode, onToggleLabMode, isAuthenticated, username, onLogout, csrfToken }) {
  return (
    <header className="navbar">
      <div className="mode-controls">
        <button
          type="button"
          className="mode-toggle"
          onClick={onToggleLabMode}
          aria-pressed={labMode}
          aria-label={labMode ? 'Switch to secure mode' : 'Switch to lab mode'}
          disabled={!isAuthenticated && !labMode}
          title={!isAuthenticated && !labMode ? 'Login required to access Lab Mode' : ''}
        >
          <span>{labMode ? 'LAB MODE' : 'SECURE'}</span>
        </button>
        {isAuthenticated && (
          <div 
            className={`csrf-status-badge ${labMode ? 'csrf-vulnerable' : 'csrf-protected'}`}
            title={labMode 
              ? 'CSRF Protection: DISABLED (Lab Mode) - Requests can succeed without CSRF tokens' 
              : 'CSRF Protection: ACTIVE (Secure Mode) - Requests require valid CSRF tokens'}
          >
            <span className="csrf-icon">{labMode ? '⚠️' : '🛡️'}</span>
            <span className="csrf-text">{labMode ? 'CSRF Vulnerable' : 'CSRF Protected'}</span>
          </div>
        )}
      </div>
      <nav>
        <ul className="nav-links">
          <li>
            <Link
              to="/"
              className={currentPath === '/' ? 'active' : ''}
            >
              HOME
            </Link>
          </li>
          <li>
            <Link
              to="/about"
              className={currentPath === '/about' ? 'active' : ''}
            >
              ABOUT
            </Link>
          </li>
          <li>
            <Link
              to="/spam-detector"
              className={currentPath === '/spam-detector' ? 'active' : ''}
            >
              SPAM DETECTOR
            </Link>
          </li>
          {labMode && (
            <li>
              <Link
                to="/vulnerabilities"
                className={currentPath === '/vulnerabilities' ? 'active' : ''}
              >
                VULNERABILITIES
              </Link>
            </li>
          )}
        </ul>
      </nav>
      <div className="header-right">
        {isAuthenticated ? (
          <div className="user-info">
            <span className="username">Logged in as: {username}</span>
            <button
              type="button"
              className="logout-button"
              onClick={onLogout}
              aria-label="Logout"
            >
              Logout
            </button>
          </div>
        ) : (
          <Link to="/login" className="login-link">
            Login
          </Link>
        )}
      </div>
    </header>
  );
}

export default Header;