## HackTheStack - Phase 3: Authentication & Advanced Abuse

Phase 3 builds on the Phase 2 web application and introduces **advanced security topics** around authentication, authorization, and backend abuse. The goal is to keep the app feeling like a real product, while adding **carefully gated vulnerabilities** for educational use in **Lab Mode** only.

**Current Status:** Phase 3 features are **fully implemented**. All planned vulnerabilities (SSRF, Insecure File Upload, Security Misconfiguration, SQL Injection, Weak Authentication) are implemented and documented in the `/vulnerabilities` page (Lab Mode only).

---

## Table of Contents

- [Overview](#overview)
- [Security Modes](#security-modes)
- [Phase 3 Focus Areas](#phase-3-focus-areas)
- [Vulnerability Template](#vulnerability-template)
- [Development Principles](#development-principles)
- [Current Implementation Status](#current-implementation-status)

---

## Overview

Phase 3 extends the existing spam detection app with **authentication**, **session/state**, and **high‑impact backend behaviors**. 

For every feature, we will implement:

- A **Secure Mode** (default) that reflects production‑grade best practices.
- A **Lab Mode** that is visually distinct and intentionally vulnerable, but only when explicitly enabled and enforced **server‑side**.

All vulnerabilities are:

- For **local, educational** use only.
- Designed to be **manually exploitable** via browser, DevTools, curl, or tools like Burp.
- Documented with **what went wrong**, **how it’s exploited**, and **how to fix it**.

---

## Security Modes

- **Secure Mode (default)**  
  - Production-grade security practices.  
  - Defensive input validation and strict backend checks.  
  - Access control enforced (e.g., user ownership verification).  
  - User input is HTML-escaped to prevent XSS.  
  - No intentional vulnerabilities exposed.

- **Lab Mode (explicit opt‑in)**  
  - Clearly different UI styling / banners indicating training mode.  
  - Selected vulnerabilities are enabled on the server side.  
  - Useful for demos, workshops, and self‑study.  
  - Vulnerabilities are documented in the `/vulnerabilities` page.

### Lab Mode Implementation

Lab Mode is controlled by a **two-layer security model**:

1. **Server-side master switch** (`LAB_MODE_ENABLED` environment variable):
   - Set `LAB_MODE_ENABLED=true` to allow Lab Mode (default: `true`)
   - Set `LAB_MODE_ENABLED=false` to disable Lab Mode entirely (safety override)
   - Location: `backend/main.py` line 46

2. **Client-side toggle** (`X-Lab-Mode` header):
   - Frontend sends `X-Lab-Mode: true` header when Lab Mode is enabled in UI
   - Backend checks both the environment variable AND the header
   - Both must be enabled for Lab Mode to activate

**Current Implementation:**
- Lab Mode toggle is available in the frontend header
- Backend endpoints check `LAB_MODE_ENABLED` env var and `X-Lab-Mode` header
- Vulnerabilities are gated server-side in endpoints like `/analysis/list` and `/analysis/{id}`
- See `backend/main.py` for implementation details

---

## Phase 3 Focus Areas

Phase 3 will focus on **Authentication & Advanced Abuse** scenarios, including but not limited to:

- **Weak / Misconfigured Authentication & Brute Force**
  - Simple login flows with poor rate limiting or lockout.  
  - Weak password rules and missing monitoring.

- **SSRF (Server‑Side Request Forgery)**
  - Backend endpoints that fetch external resources based on user‑controlled URLs.  
  - Insecure variants that allow access to internal addresses/services in Lab Mode.

- **Insecure File Upload**
  - File upload feature for spam analysis or attachments.  
  - Vulnerable variants that relax content‑type, size, or storage validation.

Additional advanced topics (e.g., session fixation, privilege escalation, business logic abuse) can be added later as we expand this phase.

---

## Vulnerability Template

Every vulnerability we introduce in Phase 3 will follow this educational format:

- **❌ What went wrong**  
  - Brief description of the insecure behavior or bad design decision.

- **🔍 How it’s exploited**  
  - Step‑by‑step manual exploitation flow (browser / DevTools / curl / Burp, etc.).  
  - Example payloads or requests.

- **⚠️ Realistic attack scenario**  
  - Contextual example of how this vulnerability could be abused in practice.

- **✅ How it should be fixed**  
  - Recommended secure design and implementation notes.  
  - References to where the secure variant lives in the codebase.

We will create one section per vulnerability (e.g., "Weak Login Brute Force", "SSRF in URL Fetcher", "Unsafe File Upload") using this template.

**Note:** Phase 2 vulnerabilities (Reflected XSS, Stored XSS, IDOR) are already implemented and documented in the `/vulnerabilities` page. They follow this same template.

---

## Development Principles

For all Phase 3 work:

- **Secure‑first**: Implement the secure version of a feature before the vulnerable variant.  
- **Single codebase**: Secure and vulnerable paths live side‑by‑side, gated by **server‑side** mode checks.  
- **Realistic UX**: The app should still feel like a normal AI spam product, not a CTF puzzle.  
- **Document trust boundaries**: Clearly state which inputs are untrusted and where validation / authorization happens.  
- **No real data**: Use only synthetic or demo data; never rely on real user content or secrets.

---

## Current Implementation Status

### Phase 2 Vulnerabilities (Implemented)

The following vulnerabilities are **currently implemented** and documented in the `/vulnerabilities` page:

1. **Vulnerability #1: Reflected XSS in Message Preview**
   - Lab Mode: Renders user input as HTML in message preview
   - Secure Mode: HTML-escapes all user input
   - Location: `spam-detection-app/src/pages/SpamDetector.jsx`

2. **Vulnerability #2: Stored XSS in Saved Analyses**
   - Lab Mode: Renders stored message text as HTML in saved analysis detail view
   - Secure Mode: Displays stored messages as plain text
   - Location: `spam-detection-app/src/pages/SpamDetector.jsx`

3. **Vulnerability #3: IDOR in Saved Analyses Access**
   - Lab Mode: API doesn't verify ownership; returns all analyses and allows access to any analysis by ID
   - Secure Mode: Filters analyses by `user_id` and returns 403 Forbidden for unauthorized access
   - Location: `backend/main.py`

4. **Vulnerability #4: CSRF (Cross-Site Request Forgery)**
   - Lab Mode: Accepts state-changing requests without CSRF token validation
   - Secure Mode: Requires valid CSRF token for all state-changing requests
   - Location: `backend/main.py`

### Phase 3 Vulnerabilities (Implemented)

The following Phase 3 vulnerabilities are **fully implemented** and documented in the `/vulnerabilities` page:

1. **Vulnerability #5: SQL Injection (Login)**
   - Lab Mode: Uses string interpolation to construct SQL queries, allowing SQL injection in username/password fields
   - Secure Mode: Uses parameterized queries (prepared statements) to prevent SQL injection
   - Location: `backend/main.py` - `authenticate_user()` function
   - Status: ✅ Implemented

2. **Vulnerability #6: Weak Authentication (Brute Force)**
   - Lab Mode: No rate limiting or account lockout; allows unlimited failed login attempts
   - Secure Mode: Implements rate limiting (3 attempts) with 30-second lockout period
   - Location: `backend/main.py` - `check_rate_limit()` and `record_failed_attempt()` functions
   - Status: ✅ Implemented

3. **Vulnerability #7: Server-Side Request Forgery (SSRF)**
   - Lab Mode: Relaxed URL validation allows requests to internal/private IP addresses
   - Secure Mode: Strict validation blocks loopback, private, and link-local IP ranges; resolves DNS and checks resolved IPs
   - Location: `backend/main.py` - `/predict` endpoint with `_validate_url_secure()` function
   - Status: ✅ Implemented

4. **Vulnerability #8: Insecure File Upload (TXT Upload)**
   - Lab Mode: Skips MIME type validation, increases size limit to 10MB, stores files using original filename
   - Secure Mode: Validates both extension and MIME type, enforces 100KB limit, processes files in memory without storage
   - Location: `backend/main.py` - `/upload/txt-analyze` endpoint
   - Status: ✅ Implemented

5. **Vulnerability #9: Security Misconfiguration**
   - Lab Mode: Verbose error messages with stack traces, permissive CORS (*), exposed debug endpoints
   - Secure Mode: Generic error messages, restrictive CORS (trusted origins only), debug endpoints blocked (403)
   - Location: `backend/main.py` - Error handlers, `DynamicCORSMiddleware`, `/debug/info` and `/debug/health` endpoints
   - Status: ✅ Implemented

All vulnerabilities follow the **Vulnerability Template** and are documented in the `/vulnerabilities` page with detailed explanations of what went wrong, how they're exploited, realistic attack scenarios, and how they should be fixed.


