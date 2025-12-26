import { Link } from 'react-router-dom';

function Header({ currentPath, labMode, onToggleLabMode, isAuthenticated, username, onLogout }) {
  return (
    <header className="navbar">
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