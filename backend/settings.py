"""
Settings module for HackTheStack backend.

This module provides environment-based configuration with strict Lab Mode gating.
Settings are read at runtime (not at import time) to allow testing with monkeypatch.
"""

import os
from typing import Literal


def get_settings() -> dict:
    """
    Get application settings from environment variables.
    
    Returns:
        dict with keys:
            - app_env: "development" | "production" (default: "development")
            - lab_mode_enabled: bool (default: False)
            - lab_available: bool (computed from app_env and lab_mode_enabled)
    """
    app_env = os.getenv("APP_ENV", "development").lower()
    lab_mode_enabled_str = os.getenv("LAB_MODE_ENABLED", "false").lower()
    lab_mode_enabled = lab_mode_enabled_str in ("true", "1", "yes")
    
    # Lab mode is only available in development environment AND when explicitly enabled
    lab_available = (app_env == "development") and lab_mode_enabled
    
    return {
        "app_env": app_env,
        "lab_mode_enabled": lab_mode_enabled,
        "lab_available": lab_available,
    }


def resolve_mode(request) -> Literal["secure", "lab"]:
    """
    Resolve the security mode for a request based on environment settings and request headers.
    
    Args:
        request: FastAPI/Starlette Request object
        
    Returns:
        "secure" or "lab"
        
    Rules:
        - If lab_available is False: always return "secure" (ignore any request toggle)
        - If lab_available is True: return "lab" only when X-Lab-Mode header is "true" (case-insensitive)
        - Otherwise: return "secure"
    """
    settings = get_settings()
    
    # If lab mode is not available, always return secure
    if not settings["lab_available"]:
        return "secure"
    
    # Check for X-Lab-Mode header (case-insensitive)
    x_lab_mode = request.headers.get("X-Lab-Mode", "").strip().lower()
    
    # Return "lab" only if header is explicitly "true"
    if x_lab_mode in ("true", "1", "yes"):
        return "lab"
    
    # Default to secure
    return "secure"

