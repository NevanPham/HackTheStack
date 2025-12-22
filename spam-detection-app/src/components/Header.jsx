import { Link } from 'react-router-dom';

function Header({ currentPath }) {
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
        </ul>
      </nav>
    </header>
  );
}

export default Header;