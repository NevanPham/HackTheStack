## HackTheStack - Phase 3: Authentication & Advanced Abuse

Phase 3 builds on the Phase 2 web application and introduces **advanced security topics** around authentication, authorization, and backend abuse. The goal is to keep the app feeling like a real product, while adding **carefully gated vulnerabilities** for educational use in **Lab Mode** only.

**Current Status:** Phase 3 features are **not yet implemented**. This document outlines the planned work. Phase 2 vulnerabilities (XSS, IDOR) are currently implemented and documented in the `/vulnerabilities` page (Lab Mode only).

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
   - Location: `spam-detection-app/src/pages/SpamDetector.jsx` (lines 529-538)

2. **Vulnerability #2: Stored XSS in Saved Analyses**
   - Lab Mode: Renders stored message text as HTML in saved analysis detail view
   - Secure Mode: Displays stored messages as plain text
   - Location: `spam-detection-app/src/pages/SpamDetector.jsx` (lines 731-740)

3. **Vulnerability #3: IDOR in Saved Analyses Access**
   - Lab Mode: API doesn't verify ownership; returns all analyses and allows access to any analysis by ID
   - Secure Mode: Filters analyses by `user_id` and returns 403 Forbidden for unauthorized access
   - Location: `backend/main.py` (lines 622-641, 702-708)

### Phase 3 Features (Planned - Not Yet Implemented)

The following features are **planned** but not yet implemented:

1. **Authentication & Sessions**
   - Basic login/logout flow and session handling (Secure Mode).  
   - Lab Mode variants demonstrating weak authentication and brute force.
   - Status: ⏳ Not started

2. **SSRF‑Style Fetch Endpoint**
   - Secure endpoint for fetching approved external resources.  
   - Lab Mode variant allowing unsafe target URLs for SSRF learning.
   - Status: ⏳ Not started

3. **File Upload for Spam Analysis**
   - Secure file upload for email/message dumps.  
   - Lab Mode variant with relaxed validation and storage rules.
   - Status: ⏳ Not started

Each item above will eventually get its own detailed section following the **Vulnerability Template** defined earlier, and will be added to the `/vulnerabilities` page.


