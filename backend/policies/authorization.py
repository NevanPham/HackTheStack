"""
Centralized authorization policies for FastAPI routes.

This module provides reusable FastAPI dependencies for authentication and
authorization, with mode-aware behavior (Secure vs Lab Mode).
"""

from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException, status, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import sqlite3
import json
import logging
from pathlib import Path

from backend.settings import resolve_mode, get_settings

# Database path - matches main.py
project_root = Path(__file__).parent.parent.parent
DB_PATH = project_root / "backend" / "db" / "analyses.db"


def _get_db_connection() -> sqlite3.Connection:
    """Return a new SQLite connection for the current request."""
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def verify_token(token: str, secret_key: str, algorithm: str = "HS256") -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token.
    
    Note: This is a standalone version to avoid circular imports.
    The secret_key and algorithm should match main.py configuration.
    """
    from jose import JWTError, jwt
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload
    except JWTError:
        return None

logger = logging.getLogger(__name__)

# Security scheme for JWT tokens
security = HTTPBearer()


class CurrentUser:
    """Represents the currently authenticated user."""
    
    def __init__(self, username: str, user_id: int):
        self.username = username
        self.id = user_id  # Database user ID
        # For compatibility with existing code that uses username as user_id
        self.user_id = username  # Username used as user_id in X-User-Id header


def require_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    secret_key: str = None,
    algorithm: str = "HS256"
) -> CurrentUser:
    """
    FastAPI dependency that verifies authentication and returns the current user.
    
    This dependency:
    - Extracts and verifies the JWT token from the Authorization header
    - Returns a CurrentUser object if authentication succeeds
    - Raises HTTPException(401) if authentication fails
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: CurrentUser = Depends(require_user)):
            return {"message": f"Hello, {user.username}"}
    """
    import os
    # Get secret key from environment or use default (matches main.py)
    if secret_key is None:
        secret_key = os.getenv("JWT_SECRET_KEY", "hackthestack-secret-key-change-in-production")
    
    token = credentials.credentials
    payload = verify_token(token, secret_key, algorithm)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username: str = payload.get("sub")
    user_id: int = payload.get("user_id")
    
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # If user_id is not in token, fetch it from database
    if user_id is None:
        conn = _get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM users WHERE username = ?",
                (username,)
            )
            row = cursor.fetchone()
            if row:
                user_id = row[0]
            else:
                # Fallback: use username as identifier
                user_id = 0
        finally:
            conn.close()
    
    return CurrentUser(username=username, user_id=user_id)


def require_analysis_access(
    analysis_id: int,
    request: Request,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
) -> Dict[str, Any]:
    """
    FastAPI dependency that enforces access control for analysis resources.
    
    This dependency:
    - Fetches the analysis from the database
    - In Secure Mode: enforces ownership (analysis.user_id == user.id)
    - In Lab Mode: bypasses ownership check (intentional IDOR vulnerability)
    - Returns a dict with analysis data if access is granted
    - Raises HTTPException(404) if analysis not found
    - Raises HTTPException(403) if access denied in Secure Mode
    
    Usage:
        @app.get("/analysis/{analysis_id}")
        async def get_analysis(
            analysis_data: Dict = Depends(require_analysis_access)
        ):
            return SavedAnalysisDetail(**analysis_data)
    
    Note: The analysis_id parameter is automatically extracted from the route path.
    """
    # Resolve mode using centralized resolver
    mode = resolve_mode(request)
    
    # Fetch analysis from database
    conn = _get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, message_text, selected_models, prediction_summary, created_at, user_id
            FROM saved_analyses
            WHERE id = ?
            """,
            (analysis_id,),
        )
        row = cursor.fetchone()
    finally:
        conn.close()
    
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with id {analysis_id} not found",
        )
    
    # Mode-aware access control
    if mode == "secure":
        # Secure Mode: Enforce ownership
        # Compare analysis owner with requesting user from X-User-Id header
        # Note: x_user_id is the username (from X-User-Id header)
        # row["user_id"] is also the username stored in the database
        if not x_user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="X-User-Id header is required",
            )
        
        analysis_owner = row["user_id"]
        requesting_user_id = x_user_id
        
        if analysis_owner != requesting_user_id:
            logger.warning(
                f"Access denied: User {requesting_user_id} attempted to access "
                f"analysis {analysis_id} owned by {analysis_owner}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied to analysis {analysis_id}",
            )
        # Ownership check passed - don't include user_id in response
        user_id = None
    else:
        # Lab Mode: Intentionally bypass ownership check (IDOR vulnerability)
        logger.warning(
            f"⚠️ Lab Mode: Allowing access to analysis {analysis_id} "
            f"without ownership verification (IDOR vulnerability)"
        )
        # Include user_id in response to make vulnerability visible
        user_id = row["user_id"]
    
    # Return analysis data as dict - route handler will convert to SavedAnalysisDetail
    return {
        "id": row["id"],
        "message_text": row["message_text"],
        "selected_models": json.loads(row["selected_models"]),
        "prediction_summary": json.loads(row["prediction_summary"])
        if row["prediction_summary"] is not None
        else None,
        "created_at": str(row["created_at"]),
        "user_id": user_id,
    }


def get_user_id_from_header(
    x_user_id: Optional[str] = Header(None, alias="X-User-Id")
) -> str:
    """
    FastAPI dependency that extracts user_id from X-User-Id header.
    
    This is used for routes that need the user_id but don't require full authentication
    (e.g., /analysis/list, /analysis/save).
    
    Raises HTTPException(400) if header is missing.
    """
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-User-Id header is required",
        )
    return x_user_id

