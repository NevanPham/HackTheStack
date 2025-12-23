## HackTheStack - Phase 3: Authentication & Advanced Abuse

Phase 3 builds on the Phase 2 web application and introduces **advanced security topics** around authentication, authorization, and backend abuse. The goal is to keep the app feeling like a real product, while adding **carefully gated vulnerabilities** for educational use in **Lab Mode** only.

This document is intentionally **minimal** for now. We will grow it as we design and implement each feature and its secure/vulnerable variants.

---

## Table of Contents

- [Overview](#overview)
- [Security Modes](#security-modes)
- [Phase 3 Focus Areas](#phase-3-focus-areas)
- [Vulnerability Template](#vulnerability-template)
- [Development Principles](#development-principles)
- [Planned Work](#planned-work)

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
  - Stronger authentication and session handling.  
  - Defensive input validation and strict backend checks.  
  - No intentional vulnerabilities exposed.

- **Lab Mode (explicit opt‑in)**  
  - Clearly different UI styling / banners indicating training mode.  
  - Selected vulnerabilities are enabled on the server side.  
  - Useful for demos, workshops, and self‑study.

Implementation details for how the mode is toggled and enforced (e.g., config flag, environment variable) will be defined as we start coding Phase 3.

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
  - Step‑by‑step manual exploitation flow (browser / curl / Burp, etc.).  
  - Example payloads or requests.

- **✅ How it should be fixed**  
  - Recommended secure design and implementation notes.  
  - References to where the secure variant lives in the codebase.

We will create one section per vulnerability (e.g., “Weak Login Brute Force”, “SSRF in URL Fetcher”, “Unsafe File Upload”) using this template.

---

## Development Principles

For all Phase 3 work:

- **Secure‑first**: Implement the secure version of a feature before the vulnerable variant.  
- **Single codebase**: Secure and vulnerable paths live side‑by‑side, gated by **server‑side** mode checks.  
- **Realistic UX**: The app should still feel like a normal AI spam product, not a CTF puzzle.  
- **Document trust boundaries**: Clearly state which inputs are untrusted and where validation / authorization happens.  
- **No real data**: Use only synthetic or demo data; never rely on real user content or secrets.

---

## Planned Work

This section will be expanded as we implement concrete features. Initial high‑level roadmap:

1. **Authentication & Sessions**
   - Basic login/logout flow and session handling (Secure Mode).  
   - Lab Mode variants demonstrating weak authentication and brute force.

2. **SSRF‑Style Fetch Endpoint**
   - Secure endpoint for fetching approved external resources.  
   - Lab Mode variant allowing unsafe target URLs for SSRF learning.

3. **File Upload for Spam Analysis**
   - Secure file upload for email/message dumps.  
   - Lab Mode variant with relaxed validation and storage rules.

Each item above will eventually get its own detailed section following the **Vulnerability Template** defined earlier.


