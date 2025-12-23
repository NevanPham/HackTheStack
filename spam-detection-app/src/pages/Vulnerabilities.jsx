import { Navigate, Link } from 'react-router-dom';
import '../styles/vulnerabilities.css';

function Vulnerabilities({ labMode }) {
  if (!labMode) {
    return <Navigate to="/" replace />;
  }

  return (
    <div className="vulnerabilities-page">
      <div className="vuln-hero">
        <div>
          <h1>Vulnerabilities (Lab Mode Only)</h1>
          <p className="lead">
            High-level notes on intentionally vulnerable behaviors in Lab Mode. Secure Mode keeps protections on; Lab Mode relaxes them so you can observe the difference.
          </p>
        </div>
      </div>

      <section className="vuln-section">
        <div className="vuln-card">
          <p className="vuln-label">Vulnerability #1: Reflected XSS in Message Preview</p>
          <div className="vuln-grid">
            <div className="vuln-tile">
              <h3>😱 What went wrong</h3>
              <p>
                In Lab Mode, the Spam Detector preview renders user input as HTML. Tags like {'<b>bold</b>'} are interpreted instead of shown literally, so untrusted content can inject markup.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>🤔 How it’s exploited</h3>
              <p>
                When the page echoes your submission, the browser executes any provided HTML. Harmless formatting (e.g., {'<i>italic</i>'}) will style the preview; more dangerous payloads would also run in Lab Mode.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>😁 How it should be fixed</h3>
              <p>
                Render user input as text so the browser does not interpret tags. In Secure Mode, the preview escapes HTML, showing {'<b>bold</b>'} literally and preventing injected markup from rendering.
              </p>
            </div>
          </div>

          <div className="try-it-card">
            <h3>Try it</h3>
            <p>
              Go to the <Link to="/spam-detector">Spam Detector</Link>, switch between Secure and Lab modes, and submit a message with harmless formatting tags (e.g., {'<b>Hello</b>'}). Notice that Secure Mode shows the tags literally, while Lab Mode renders the formatting.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Vulnerabilities;

