import { Link } from 'react-router-dom';

function Header({ currentPath, labMode, onToggleLabMode }) {
  return (
    <header className="navbar">
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
      <button
        type="button"
        className="mode-toggle"
        onClick={onToggleLabMode}
        aria-pressed={labMode}
        aria-label={labMode ? 'Switch to secure mode' : 'Switch to lab mode'}
      >
        <span>{labMode ? 'LAB MODE' : 'SECURE'}</span>
      </button>
    </header>
  );
}

export default Header;