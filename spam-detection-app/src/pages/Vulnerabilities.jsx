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
              <h3>❌ What went wrong</h3>
              <p>
                In Lab Mode, the Spam Detector preview renders user input as HTML. Tags like {'<b>bold</b>'} are interpreted instead of shown literally, so untrusted content can inject markup.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>🔍 How it's exploited</h3>
              <p>
                When user input is rendered as HTML, the browser treats it as part of the page's structure and executes it. This breaks the trust boundary between user data and application code, allowing arbitrary client-side behavior—not just visual styling, but any JavaScript the browser can execute.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>⚠️ Realistic attack scenario</h3>
              <p>
                An attacker submits a crafted message containing malicious markup. When the page reflects it in the preview during the same request, the browser executes it immediately. In real applications, this can be abused to manipulate the UI, steal sensitive data exposed in the page, or perform actions on behalf of the user.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>✅ How it should be fixed</h3>
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

      <section className="vuln-section">
        <div className="vuln-card">
          <p className="vuln-label">Vulnerability #2: Stored XSS in Saved Analyses</p>
          <div className="vuln-grid">
            <div className="vuln-tile">
              <h3>❌ What went wrong</h3>
              <p>
                In Lab Mode, the Saved Analyses detail view renders stored message text as HTML. User input is saved to the database and later displayed unsafely, so tags like {'<b>hi</b>'} are interpreted when viewing saved analyses instead of shown literally.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>🔍 How it's exploited</h3>
              <p>
                When stored user input is rendered as HTML, the browser treats it as part of the page and executes it. This breaks the trust boundary, allowing arbitrary client-side behavior. Unlike Reflected XSS, the payload persists in storage and executes whenever the saved item is viewed, not just during the initial request.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>⚠️ Realistic attack scenario</h3>
              <p>
                An attacker saves malicious input into normal application data. The payload persists in storage. When another user or an admin views the saved item later, it executes automatically. This persistence and cross-user impact makes Stored XSS higher severity than Reflected XSS, as a single malicious entry can affect multiple victims over time.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>✅ How it should be fixed</h3>
              <p>
                Render stored user input as plain text so the browser does not interpret tags. In Secure Mode, saved messages are displayed literally, showing {'<b>hi</b>'} as text and preventing stored markup from rendering. The data is stored correctly; only the rendering should be safe.
              </p>
            </div>
          </div>

          <div className="try-it-card">
            <h3>Try it</h3>
            <p>
              Go to the <Link to="/spam-detector">Spam Detector</Link>, submit a message with formatting tags (e.g., {'<b>test</b>'}), save the analysis, then view it in both Secure and Lab modes. Notice that Secure Mode shows the tags literally in the saved analysis detail view, while Lab Mode renders the formatting.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Vulnerabilities;

