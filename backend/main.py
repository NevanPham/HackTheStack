import sys
from pathlib import Path
import time
import json
import sqlite3
import os
from datetime import datetime, timedelta
from secrets import token_urlsafe
import socket
import ipaddress
from urllib.parse import urlparse

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import bcrypt
from jose import JWTError, jwt
import httpx

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, status, Header, Request, Depends, File, UploadFile, Form
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import traceback

from src.lstm.model import LSTMTextClassifier
from src.xgb.model import XGBTextClassifier
from src.kmeans.model import KMeansTextInferencer

# Initialize logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model instances
models = {"lstm": None, "xgboost": None, "kmeans": None}

# Model metadata
model_metadata = {}

VALID_MODEL_IDS = ["xgboost", "lstm", "kmeans"]

DB_PATH = project_root / "backend" / "db" / "analyses.db"

# Import settings module for Lab Mode gating
from backend.settings import get_settings, resolve_mode
from backend.policies.authorization import require_analysis_access, get_user_id_from_header

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "hackthestack-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# Security scheme for JWT tokens
security = HTTPBearer()

# CSRF token storage: in-memory dict mapping username -> csrf_token
# In production, consider using Redis or database-backed storage
csrf_tokens: Dict[str, str] = {}

# Rate limiting storage: in-memory dict mapping identifier -> {"attempts": int, "lockout_until": float}
# In production, consider using Redis or database-backed storage
# Structure: {identifier: {"attempts": count, "lockout_until": timestamp}}
failed_login_attempts: Dict[str, Dict[str, Any]] = {}

# Rate limiting configuration
MAX_LOGIN_ATTEMPTS = 3
LOCKOUT_DURATION_SECONDS = 30


def _init_db() -> None:
    """Initialize SQLite database for saved analyses and users."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        # Create saved_analyses table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_text TEXT NOT NULL,
                selected_models TEXT NOT NULL,
                prediction_summary TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        
        # Migration: Add user_id column if it doesn't exist
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(saved_analyses)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'user_id' not in columns:
            conn.execute("ALTER TABLE saved_analyses ADD COLUMN user_id TEXT")
            conn.commit()
            logger.info("Added user_id column to saved_analyses table")
        
        # Create users table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        
        # Initialize default users if they don't exist
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        if user_count == 0:
            # Create password hashes for nevan and naven
            nevan_hash = bcrypt.hashpw("nevan".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            naven_hash = bcrypt.hashpw("naven".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                ("nevan", nevan_hash)
            )
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                ("naven", naven_hash)
            )
            conn.commit()
            logger.info("Initialized default users: nevan and naven")
    finally:
        conn.close()


def _get_db_connection() -> sqlite3.Connection:
    """Return a new SQLite connection for the current request."""
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup"""
    global models, model_metadata

    print("Loading models...")

    # Initialize database for saved analyses
    try:
        _init_db()
        print("✓ Database initialized")
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")

    try:
        # Load LSTM model
        lstm_dir = project_root / "models/lstm/deep_bilstm_chosen"
        models["lstm"] = LSTMTextClassifier(str(lstm_dir))

        # Load LSTM metadata
        with open(lstm_dir / "lstm_metadata.json", "r") as f:
            lstm_meta = json.load(f)
        model_metadata["lstm"] = {
            "accuracy": lstm_meta.get("test_accuracy", 0.90),
            "precision": lstm_meta.get("test_precision", 0.89),
            "recall": lstm_meta.get("test_recall", 0.88),
            "f1": lstm_meta.get("test_f1", 0.88),
        }
        print("✓ LSTM model loaded")

    except Exception as e:
        print(f"✗ Failed to load LSTM model: {e}")
        models["lstm"] = None

    try:
        # Load XGBoost model
        xgb_dir = project_root / "models/xgboost"
        models["xgboost"] = XGBTextClassifier(str(xgb_dir))

        # Load XGBoost metadata
        with open(xgb_dir / "xgb_results_summary_randomized.json", "r") as f:
            xgb_meta = json.load(f)
        model_metadata["xgboost"] = {
            "accuracy": xgb_meta["test_metrics"]["accuracy"],
            "precision": xgb_meta["test_metrics"]["precision"],
            "recall": xgb_meta["test_metrics"]["recall"],
            "f1": xgb_meta["test_metrics"]["f1"],
        }
        print("✓ XGBoost model loaded")

    except Exception as e:
        print(f"✗ Failed to load XGBoost model: {e}")
        models["xgboost"] = None

    try:
        # Load K-Means model
        kmeans_dir = project_root / "models/kmeans/k3/tfidf_1000"
        models["kmeans"] = KMeansTextInferencer(str(kmeans_dir))

        # Load K-Means metadata
        with open(kmeans_dir / "clustering_results.json", "r") as f:
            kmeans_meta = json.load(f)
        model_metadata["kmeans"] = {
            "silhouette_score": kmeans_meta.get("silhouette_score", 0.15),
            "n_clusters": kmeans_meta.get("n_clusters", 3),
            "inertia": kmeans_meta.get("inertia", 0),
        }
        print("✓ K-Means model loaded")

    except Exception as e:
        print(f"✗ Failed to load K-Means model: {e}")
        models["kmeans"] = None

    print(
        f"Loaded {sum(1 for m in models.values() if m is not None)}/3 models successfully"
    )
    
    # Log Lab Mode configuration (already fixed above)

    yield

    # Cleanup
    print("Shutting down...")


# Determine if we should show detailed API docs based on environment
# In production/Secure Mode, we might want to hide these, but for this app
# we'll keep them available (they can be restricted via authentication if needed)
app = FastAPI(
    title="Spam Detection API",
    description="Multi-model spam detection API supporting XGBoost, LSTM, and K-Means",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    # Note: In a real production deployment, consider disabling docs_url and redoc_url
    # or protecting them with authentication
)


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    """
    Custom CORS middleware that applies different CORS policies based on Lab Mode.
    
    Vulnerability #9 (Security Misconfiguration): In Lab Mode, CORS is permissive,
    allowing any origin, methods, and headers. This makes the application vulnerable
    to cross-origin attacks. In Secure Mode, CORS is restrictive, only allowing
    trusted frontend origins.
    """
    async def dispatch(self, request: StarletteRequest, call_next):
        # Resolve mode using centralized resolver
        mode = resolve_mode(request)
        is_lab_mode = (mode == "lab")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            if is_lab_mode:
                # Lab Mode: Permissive CORS (Vulnerability #9)
                response = Response()
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Methods"] = "*"
                response.headers["Access-Control-Allow-Headers"] = "*"
                response.headers["Access-Control-Allow-Credentials"] = "true"
                return response
            else:
                # Secure Mode: Restrictive CORS
                origin = request.headers.get("origin")
                allowed_origins = [
                    "http://localhost:5173",
                    "http://localhost:3000",
                    "http://localhost:5174",
                    "http://127.0.0.1:5173",
                ]
                if origin in allowed_origins:
                    response = Response()
                    response.headers["Access-Control-Allow-Origin"] = origin
                    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
                    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-User-Id, X-Lab-Mode, X-CSRF-Token"
                    response.headers["Access-Control-Allow-Credentials"] = "true"
                    return response
                else:
                    return Response(status_code=403)
        
        # Process the request
        response = await call_next(request)
        
        # Add CORS headers to response
        if is_lab_mode:
            # Lab Mode: Permissive CORS (Vulnerability #9)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
        else:
            # Secure Mode: Restrictive CORS
            origin = request.headers.get("origin")
            allowed_origins = [
                "http://localhost:5173",
                "http://localhost:3000",
                "http://localhost:5174",
                "http://127.0.0.1:5173",
            ]
            if origin in allowed_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-User-Id, X-Lab-Mode, X-CSRF-Token"
                response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response


# Add custom CORS middleware
app.add_middleware(DynamicCORSMiddleware)


class PredictionRequest(BaseModel):
    text: str
    models: List[str] = ["xgboost"]  # Default to xgboost


class BatchPredictionRequest(BaseModel):
    texts: List[str]
    models: List[str] = ["xgboost"]


class SaveAnalysisRequest(BaseModel):
    message_text: str
    selected_models: List[str]
    prediction_summary: Optional[Dict[str, Any]] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    username: str
    csrf_token: str


class SessionResponse(BaseModel):
    username: str
    authenticated: bool
    csrf_token: Optional[str] = None


class CSRFResponse(BaseModel):
    csrf_token: str


class SavedAnalysisSummary(BaseModel):
    id: int
    created_at: str
    snippet: str
    user_id: Optional[str] = None  # Only included in Lab Mode to show ownership


class SavedAnalysisDetail(BaseModel):
    id: int
    message_text: str
    selected_models: List[str]
    prediction_summary: Optional[Dict[str, Any]] = None
    created_at: str
    user_id: Optional[str] = None  # Only included in Lab Mode to show ownership


class ModelPrediction(BaseModel):
    model_id: str
    model_name: str
    prediction: int  # 0 = ham, 1 = spam
    spam_probability: float
    confidence: float
    processing_time_ms: float
    # Additional model-specific data
    cluster_id: Optional[int] = None
    cluster_distances: Optional[Dict[int, float]] = None
    user_point_2d: Optional[List[float]] = None


class URLMetadata(BaseModel):
    url: str
    status_code: Optional[int] = None
    content_length: Optional[int] = None
    content_type: Optional[str] = None
    fetch_successful: bool
    error_message: Optional[str] = None


class PredictionResponse(BaseModel):
    predictions: List[ModelPrediction]
    text_stats: Dict[str, Any]
    total_processing_time_ms: float
    url_metadata: Optional[URLMetadata] = None


class BatchPredictionItem(BaseModel):
    text_index: int
    predictions: List[ModelPrediction]
    text_stats: Dict[str, Any]
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    items: List[BatchPredictionItem]
    summary: Dict[str, Any]
    total_processing_time_ms: float


def _validate_model_ids(model_ids: List[str]) -> List[str]:
    if not model_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one model must be selected",
        )

    invalid = [m for m in model_ids if m not in VALID_MODEL_IDS]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid model(s): {', '.join(invalid)}. Valid models: {', '.join(VALID_MODEL_IDS)}"
            ),
        )

    # Preserve order but remove duplicates
    seen = set()
    unique_models = []
    for model_id in model_ids:
        if model_id not in seen:
            seen.add(model_id)
            unique_models.append(model_id)
    return unique_models


def _compute_text_stats(text: str) -> Dict[str, Any]:
    words = text.split()
    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": max(1, text.count(".") + text.count("!") + text.count("?")),
        "avg_word_length": round(np.mean([len(w) for w in words]) if words else 0, 2),
    }


def _is_url(text: str) -> bool:
    """
    Detect if the input text is a URL.
    
    Args:
        text: Input text to check
        
    Returns:
        True if text appears to be a URL, False otherwise
    """
    text = text.strip()
    if not text:
        return False
    
    # Check if it starts with http:// or https://
    if text.startswith(("http://", "https://")):
        try:
            parsed = urlparse(text)
            # Basic validation: must have scheme and netloc
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
    
    return False


def _is_private_ip(ip: str) -> bool:
    """
    Check if an IP address is in a private, loopback, or link-local range.
    
    Args:
        ip: IP address string
        
    Returns:
        True if IP is private/loopback/link-local, False otherwise
    """
    try:
        ip_obj = ipaddress.ip_address(ip)
        return (
            ip_obj.is_loopback
            or ip_obj.is_private
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
        )
    except ValueError:
        return False


def _validate_url_secure(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate URL for Secure Mode: strict validation to prevent SSRF.
    
    Vulnerability #7 (SSRF): In Lab Mode, URL validation is relaxed, allowing
    requests to internal/private IP addresses. This makes the application vulnerable
    to Server-Side Request Forgery attacks.
    
    Args:
        url: URL string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        parsed = urlparse(url)
        
        # Only allow http and https schemes
        if parsed.scheme not in ("http", "https"):
            return False, f"Only http and https schemes are allowed, got: {parsed.scheme}"
        
        # Must have a netloc (hostname)
        if not parsed.netloc:
            return False, "URL must have a valid hostname"
        
        # Extract hostname (remove port if present)
        hostname = parsed.netloc.split(":")[0]
        
        # Block common internal hostnames
        blocked_hostnames = [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "::1",
            "localhost.localdomain",
        ]
        if hostname.lower() in blocked_hostnames:
            return False, f"Access to internal hostname '{hostname}' is not allowed"
        
        # Resolve DNS and check IP address
        try:
            # Get all IP addresses for the hostname
            ip_addresses = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            for addr_info in ip_addresses:
                ip = addr_info[4][0]  # Extract IP from (host, port) tuple
                
                # Check if IP is private/loopback/link-local
                if _is_private_ip(ip):
                    return False, f"Access to private/internal IP address '{ip}' (resolved from '{hostname}') is not allowed"
        except socket.gaierror:
            return False, f"Failed to resolve hostname '{hostname}'"
        except Exception as e:
            return False, f"Error validating hostname: {str(e)}"
        
        return True, None
        
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"


async def _fetch_url(url: str, timeout: float = 5.0) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetch content from a URL with timeout and error handling.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (success, metadata_dict, error_message)
    """
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            
            # Get content length (may be None for streaming responses)
            content_length = None
            if "content-length" in response.headers:
                try:
                    content_length = int(response.headers["content-length"])
                except ValueError:
                    pass
            
            metadata = {
                "status_code": response.status_code,
                "content_length": content_length,
                "content_type": response.headers.get("content-type"),
                "fetch_successful": True,
                "error_message": None,
            }
            
            return True, metadata, None
            
    except httpx.TimeoutException:
        return False, None, "Request timed out"
    except httpx.RequestError as e:
        return False, None, f"Request failed: {str(e)}"
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def generate_csrf_token() -> str:
    """Generate a secure CSRF token."""
    return token_urlsafe(32)


def get_csrf_token_for_user(username: str) -> str:
    """Get or generate a CSRF token for a user."""
    if username not in csrf_tokens:
        csrf_tokens[username] = generate_csrf_token()
    return csrf_tokens[username]


def validate_csrf_token(username: str, token: Optional[str]) -> bool:
    """Validate a CSRF token for a user."""
    logger.info(f"CSRF Debug: validate_csrf_token called for user '{username}', token provided: {bool(token)}")
    if not token:
        logger.warning(f"CSRF Debug: No token provided for user '{username}'")
        return False
    stored_token = csrf_tokens.get(username)
    if not stored_token:
        logger.warning(f"CSRF Debug: No stored CSRF token found for user '{username}'. Available users: {list(csrf_tokens.keys())}")
        return False
    tokens_match = stored_token == token
    logger.info(f"CSRF Debug: Token validation result: {tokens_match} (stored: {stored_token[:10]}..., provided: {token[:10]}...)")
    return tokens_match


def get_client_identifier(request: Request, username: str) -> str:
    """Get a unique identifier for rate limiting (IP address + username)."""
    # Get client IP address
    client_ip = request.client.host if request.client else "unknown"
    # Use IP + username as identifier to prevent one user from blocking others
    return f"{client_ip}:{username}"


def check_rate_limit(identifier: str, mode: str) -> Optional[str]:
    """
    Check if the identifier is rate limited.
    
    Vulnerability #6 (Weak Authentication): In Lab Mode, rate limiting is bypassed,
    allowing unlimited failed login attempts. This makes the application vulnerable
    to brute force attacks.
    
    Args:
        identifier: Unique identifier for rate limiting (IP:username)
        mode: "secure" or "lab"
    
    Returns:
        Error message if rate limited, None otherwise
    """
    if mode == "lab":
        # Lab Mode: Rate limiting is bypassed (Vulnerability #6)
        # This allows unlimited failed login attempts, making brute force attacks possible
        return None
    
    # Secure Mode: Enforce rate limiting
    current_time = time.time()
    attempt_data = failed_login_attempts.get(identifier)
    
    if attempt_data:
        lockout_until = attempt_data.get("lockout_until", 0)
        
        # Check if still locked out (only if lockout is actually set)
        if lockout_until > 0 and current_time < lockout_until:
            remaining_seconds = int(lockout_until - current_time) + 1
            return f"Too many failed login attempts. Please try again in {remaining_seconds} seconds."
        
        # Lockout expired, reset attempts (only if lockout was actually set)
        if lockout_until > 0 and current_time >= lockout_until:
            failed_login_attempts[identifier] = {"attempts": 0, "lockout_until": 0}
    
    return None


def record_failed_attempt(identifier: str, mode: str) -> None:
    """
    Record a failed login attempt.
    
    Args:
        identifier: Unique identifier for rate limiting (IP:username)
        mode: "secure" or "lab"
    """
    if mode == "lab":
        # Lab Mode: Don't record failed attempts (Vulnerability #6)
        return
    
    # Secure Mode: Record and enforce rate limiting
    current_time = time.time()
    
    if identifier not in failed_login_attempts:
        failed_login_attempts[identifier] = {"attempts": 0, "lockout_until": 0}
    
    attempt_data = failed_login_attempts[identifier]
    attempt_data["attempts"] += 1
    
    # If max attempts reached, set lockout
    if attempt_data["attempts"] >= MAX_LOGIN_ATTEMPTS:
        attempt_data["lockout_until"] = current_time + LOCKOUT_DURATION_SECONDS
        logger.warning(f"Rate limit triggered for {identifier}: {attempt_data['attempts']} attempts, locked until {attempt_data['lockout_until']}")
    else:
        logger.info(f"Failed login attempt {attempt_data['attempts']}/{MAX_LOGIN_ATTEMPTS} for {identifier}")


def clear_failed_attempts(identifier: str) -> None:
    """Clear failed login attempts for a successful login."""
    if identifier in failed_login_attempts:
        del failed_login_attempts[identifier]
        logger.info(f"Cleared failed login attempts for {identifier}")


def require_csrf_token(
    username: str,
    csrf_token: Optional[str],
    request: Request,
    method: str
) -> None:
    """
    Validate CSRF token for state-changing requests in Secure Mode.
    
    Vulnerability #4 (CSRF): In Lab Mode, CSRF protection is intentionally bypassed,
    allowing state-changing requests without CSRF token validation. This makes the
    application vulnerable to Cross-Site Request Forgery attacks.
    
    Args:
        username: The authenticated user's username
        csrf_token: The CSRF token from X-CSRF-Token header
        request: FastAPI Request object for mode resolution
        method: HTTP method (POST, PUT, PATCH, DELETE)
    
    Raises:
        HTTPException: If CSRF validation fails in Secure Mode
    """
    # Resolve mode using centralized resolver
    mode = resolve_mode(request)
    logger.info(f"CSRF Debug: Mode resolved to '{mode}' for CSRF validation")
    
    if mode == "lab":
        # Lab Mode: CSRF protection is bypassed (Vulnerability #4)
        # State-changing requests succeed without CSRF token validation
        logger.info("CSRF Debug: Lab Mode detected - bypassing CSRF validation")
        return
    
    # Secure Mode: Require CSRF token for state-changing methods
    # This prevents CSRF attacks by validating the token matches the user's session
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        if not validate_csrf_token(username, csrf_token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token validation failed. Please refresh the page and try again.",
            )


def authenticate_user(username: str, password: str, mode: str = "secure") -> Optional[Dict[str, Any]]:
    """
    Authenticate a user by username and password.
    
    Vulnerability #5 (SQL Injection): In Lab Mode, the authentication query uses unsafe
    string interpolation for both username and password, allowing SQL injection attacks.
    An attacker can manipulate either field to bypass authentication.
    
    Args:
        username: The username to authenticate
        password: The password to verify
        mode: "secure" or "lab"
    
    Returns:
        Dict with user id and username if authentication succeeds, None otherwise
    """
    conn = _get_db_connection()
    try:
        cursor = conn.cursor()
        
        if mode == "lab":
            # Lab Mode: Vulnerable SQL query with string interpolation for BOTH username and password
            # This allows SQL injection attacks through either parameter
            # Example exploits:
            #   - username = "admin' OR '1'='1" --" (bypasses username check)
            #   - username = "nevan" and password = "' OR '1'='1' --" (bypasses password check)
            
            # First, try to find user by username (vulnerable to username injection)
            query = f"SELECT id, username, password_hash FROM users WHERE username = '{username}'"
            logger.warning(f"⚠️ Lab Mode SQL Injection: Executing query: {query}")
            cursor.execute(query)
        else:
            # Secure Mode: Use parameterized query to prevent SQL injection
            cursor.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ?",
                (username,)
            )
        
        row = cursor.fetchone()
        if row is None:
            logger.info(f"Authentication failed: No user found for username input")
            return None
        
        user_id, db_username, stored_password_hash = row
        logger.info(f"User found: {db_username} (ID: {user_id})")
        
        if mode == "lab":
            # In Lab Mode, check if password contains SQL injection patterns
            # If so, bypass password verification (vulnerability demonstration)
            sql_injection_patterns = ["' OR", "'OR", "OR '1'='1", "OR 1=1", "OR '1'='1'", "--"]
            password_upper = password.upper()
            has_sql_injection = any(pattern in password_upper for pattern in sql_injection_patterns)
            
            if has_sql_injection:
                # SQL injection detected in password - authentication bypassed (vulnerability)
                logger.warning(f"⚠️ Lab Mode: SQL Injection detected in password field - authentication bypassed for user: {db_username}")
                return {"id": user_id, "username": db_username}
            elif bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
                # Normal password verification if no injection detected
                logger.info(f"Authentication successful for user: {db_username}")
                return {"id": user_id, "username": db_username}
            else:
                logger.info(f"Authentication failed: Password mismatch for user: {db_username}")
                return None
        else:
            # Secure Mode: Always verify password with bcrypt
            if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
                logger.info(f"Authentication successful for user: {db_username}")
                return {"id": user_id, "username": db_username}
            
            logger.info(f"Authentication failed: Password mismatch for user: {db_username}")
            return None
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Get user information by username."""
    conn = _get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username FROM users WHERE username = ?",
            (username,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {"id": row[0], "username": row[1]}
    finally:
        conn.close()


async def _run_models_for_text(
    text: str, model_ids: List[str]
) -> Tuple[List[ModelPrediction], Dict[str, Any], float]:
    start_time = time.time()
    predictions: List[ModelPrediction] = []

    for model_id in model_ids:
        model_start = time.time()

        try:
            if model_id == "xgboost":
                pred = await predict_xgboost(text)
            elif model_id == "lstm":
                pred = await predict_lstm(text)
            elif model_id == "kmeans":
                pred = await predict_kmeans(text)
            else:
                continue

            pred["processing_time_ms"] = round((time.time() - model_start) * 1000, 2)
            predictions.append(ModelPrediction(**pred))

        except Exception as exc:
            print(f"Error predicting with {model_id}: {exc}")
            continue

    text_stats = _compute_text_stats(text)
    total_time_ms = round((time.time() - start_time) * 1000, 2)

    return predictions, text_stats, total_time_ms


@app.post("/auth/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    http_request: Request,
):
    """
    Authenticate user and return JWT token.
    
    Vulnerability #5 (SQL Injection): In Lab Mode, the authentication function uses
    unsafe SQL query construction, making it vulnerable to SQL injection attacks.
    
    Vulnerability #6 (Weak Authentication): In Lab Mode, rate limiting is bypassed,
    allowing unlimited failed login attempts and making brute force attacks possible.
    """
    # Resolve mode using centralized resolver
    mode = resolve_mode(http_request)
    
    # Get client identifier for rate limiting
    identifier = get_client_identifier(http_request, request.username)
    
    # Check rate limit in Secure Mode (Vulnerability #6: bypassed in Lab Mode)
    rate_limit_error = check_rate_limit(identifier, mode)
    if rate_limit_error:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=rate_limit_error,
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = authenticate_user(request.username, request.password, mode=mode)
    if not user:
        # Record failed attempt in Secure Mode (Vulnerability #6: not recorded in Lab Mode)
        record_failed_attempt(identifier, mode)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Clear failed attempts on successful login
    clear_failed_attempts(identifier)
    
    access_token_expires = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    access_token = create_access_token(
        data={"sub": user["username"], "user_id": user["id"]},
        expires_delta=access_token_expires
    )
    
    # Generate CSRF token for the user
    csrf_token = get_csrf_token_for_user(user["username"])
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        username=user["username"],
        csrf_token=csrf_token
    )


@app.get("/auth/session", response_model=SessionResponse)
async def get_session(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get current session information from JWT token."""
    token = credentials.credentials
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get or generate CSRF token for the user
    csrf_token = get_csrf_token_for_user(username)
    
    return SessionResponse(
        username=username,
        authenticated=True,
        csrf_token=csrf_token
    )


@app.get("/auth/csrf", response_model=CSRFResponse)
async def get_csrf_token(
    credentials: HTTPAuthorizationCredentials = security
):
    """Get CSRF token for the current session."""
    token = credentials.credentials
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get or generate CSRF token for the user
    csrf_token = get_csrf_token_for_user(username)
    
    return CSRFResponse(csrf_token=csrf_token)


@app.get("/auth/check-lab-access")
async def check_lab_access(
    authorization: Optional[str] = Header(None)
):
    """Check if user is authenticated and can access lab mode."""
    if not authorization or not authorization.startswith("Bearer "):
        return {"authenticated": False, "can_access_lab": False}
    
    token = authorization.replace("Bearer ", "")
    payload = verify_token(token)
    if payload is None:
        return {"authenticated": False, "can_access_lab": False}
    
    username: str = payload.get("sub")
    if username is None:
        return {"authenticated": False, "can_access_lab": False}
    
    settings = get_settings()
    return {"authenticated": True, "can_access_lab": settings["lab_available"], "username": username}


@app.get("/api/mode")
async def get_mode(request: Request):
    """
    Test-only introspection endpoint for mode resolution.
    
    Returns the current mode and lab availability status.
    This endpoint is useful for testing and debugging mode resolution.
    """
    settings = get_settings()
    mode = resolve_mode(request)
    
    return {
        "mode": mode,
        "lab_available": settings["lab_available"],
    }


@app.get("/")
async def root():
    """Health check endpoint"""
    available_models = [name for name, model in models.items() if model is not None]
    return {
        "status": "ok",
        "message": "Spam Detection API is running",
        "available_models": available_models,
        "endpoints": {"predict": "/predict", "model_info": "/models/info"},
    }


@app.get("/models/info")
async def get_model_info(
    request: Request
):
    """
    Get information about all loaded models.
    
    Vulnerability #9 (Security Misconfiguration): In Lab Mode, this endpoint
    returns detailed model information including file paths and internal details.
    In Secure Mode, only basic status information is returned.
    """
    mode = resolve_mode(request)
    is_lab_mode = (mode == "lab")
    
    info = {}

    for model_id, model in models.items():
        if model is not None:
            base_info = {
                "status": "loaded",
                "metadata": model_metadata.get(model_id, {}),
                "description": {
                    "xgboost": "Gradient Boosting classifier - Fast and accurate",
                    "lstm": "Deep Bidirectional LSTM - Captures context and word order",
                    "kmeans": "Unsupervised clustering - Distance-based classification",
                }.get(model_id, ""),
            }
            
            if is_lab_mode:
                # Lab Mode: Include detailed debug information (Vulnerability #9)
                base_info["debug_info"] = {
                    "model_type": type(model).__name__,
                    "model_path": str(project_root / "models" / model_id) if model_id in ["lstm", "xgboost", "kmeans"] else None,
                }
            
            info[model_id] = base_info
        else:
            error_info = {"status": "not_loaded"}
            if is_lab_mode:
                # Lab Mode: Include error details (Vulnerability #9)
                error_info["error"] = "Model failed to load"
                error_info["debug_info"] = {
                    "expected_path": str(project_root / "models" / model_id),
                }
            else:
                # Secure Mode: Generic error
                error_info["error"] = "Model unavailable"
            
            info[model_id] = error_info

    return info


@app.get("/debug/info")
async def debug_info(
    request: Request
):
    """
    Debug endpoint - only accessible in Lab Mode.
    
    Vulnerability #9 (Security Misconfiguration): This endpoint exposes internal
    application details including environment variables, file paths, and system
    information. In Secure Mode, this endpoint returns 403 Forbidden.
    """
    mode = resolve_mode(request)
    
    if mode != "lab":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debug endpoints are not available in Secure Mode"
        )
    
    # Lab Mode: Expose detailed debug information (Vulnerability #9)
    return {
        "application": {
            "name": "Spam Detection API",
            "version": "1.0.0",
            "framework": "FastAPI",
            "python_version": sys.version,
        },
        "environment": {
            "settings": get_settings(),
            "project_root": str(project_root),
            "db_path": str(DB_PATH),
            "jwt_secret_key_set": bool(SECRET_KEY and SECRET_KEY != "hackthestack-secret-key-change-in-production"),
        },
        "models": {
            "loaded": [k for k, v in models.items() if v is not None],
            "available": list(models.keys()),
        },
        "paths": {
            "backend": str(project_root / "backend"),
            "models": str(project_root / "models"),
            "uploads": str(project_root / "backend" / "uploads"),
        },
        "warning": "This endpoint exposes sensitive information and should not be accessible in production"
    }


@app.get("/debug/health")
async def debug_health(
    request: Request
):
    """
    Detailed health check endpoint - only accessible in Lab Mode.
    
    Vulnerability #9 (Security Misconfiguration): This endpoint provides detailed
    system health information including database status, model loading status,
    and internal metrics. In Secure Mode, this endpoint returns 403 Forbidden.
    """
    mode = resolve_mode(request)
    
    if mode != "lab":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debug endpoints are not available in Secure Mode"
        )
    
    # Lab Mode: Expose detailed health information (Vulnerability #9)
    db_exists = DB_PATH.exists()
    db_size = DB_PATH.stat().st_size if db_exists else 0
    
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "database": {
            "exists": db_exists,
            "path": str(DB_PATH),
            "size_bytes": db_size,
        },
        "models": {
            model_id: {
                "loaded": model is not None,
                "type": type(model).__name__ if model else None,
            }
            for model_id, model in models.items()
        },
        "system": {
            "settings": get_settings(),
            "csrf_tokens_count": len(csrf_tokens),
            "failed_login_attempts_count": len(failed_login_attempts),
        }
    }


# Add global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc):
    """
    Handle validation errors with different verbosity based on Lab Mode.
    
    Vulnerability #9 (Security Misconfiguration): In Lab Mode, error responses
    include detailed validation errors, stack traces, and request body details.
    This leaks internal application structure and can aid attackers.
    """
    # Resolve mode using centralized resolver
    mode = resolve_mode(request)
    is_lab_mode = (mode == "lab")
    
    logger.error(f"Validation error: {exc}")
    
    if is_lab_mode:
        # Lab Mode: Verbose error messages (Vulnerability #9)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "detail": str(exc.errors()),
                "body": str(exc.body),
                "path": str(request.url.path),
                "method": request.method,
                "debug_info": {
                    "exception_type": type(exc).__name__,
                    "full_errors": exc.errors(),
                }
            },
        )
    else:
        # Secure Mode: Generic error messages
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "detail": "Invalid request format",
            },
        )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc):
    """
    Handle general exceptions with different verbosity based on Lab Mode.
    
    Vulnerability #9 (Security Misconfiguration): In Lab Mode, error responses
    include stack traces, file paths, and internal exception details. This leaks
    sensitive information about the application structure and can aid attackers.
    """
    # Resolve mode using centralized resolver
    mode = resolve_mode(request)
    is_lab_mode = (mode == "lab")
    
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    if is_lab_mode:
        # Lab Mode: Verbose error with stack trace (Vulnerability #9)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
                "path": str(request.url.path),
                "method": request.method,
                "debug_info": {
                    "file": __file__,
                    "line": exc.__traceback__.tb_lineno if exc.__traceback__ else None,
                }
            },
        )
    else:
        # Secure Mode: Generic error message
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred",
            },
        )


# Enhance predict endpoint with validation
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    http_request: Request
):
    """
    Predict spam probability using selected models.
    
    If the input text is detected as a URL, the backend will fetch the URL content
    for analysis. URL validation is enforced in Secure Mode to prevent SSRF attacks.
    
    Vulnerability #7 (SSRF): In Lab Mode, URL validation is relaxed, allowing
    requests to internal/private IP addresses. This makes the application vulnerable
    to Server-Side Request Forgery attacks.

    - **text**: The message text to analyze (or a URL to fetch)
    - **models**: List of model IDs to use (xgboost, lstm, kmeans)
    """
    start_time = time.time()

    # Input validation
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Text input cannot be empty"
        )

    if len(request.text) > 10000:  # Set reasonable limit
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text exceeds maximum length of 10000 characters",
        )

    # Resolve mode using centralized resolver
    mode = resolve_mode(http_request)
    is_lab_mode = (mode == "lab")

    # Check if input is a URL and handle URL fetching
    url_metadata = None
    text_to_analyze = request.text
    
    if _is_url(request.text):
        url = request.text.strip()
        
        # Validate URL based on mode
        if is_lab_mode:
            # Lab Mode: Skip or relax validation (SSRF vulnerability)
            # Only basic scheme check, allow all destinations
            try:
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Only http and https schemes are allowed, got: {parsed.scheme}"
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid URL format: {str(e)}"
                )
            
            logger.warning(f"⚠️ Lab Mode SSRF: Fetching URL without strict validation: {url}")
        else:
            # Secure Mode: Strict validation to prevent SSRF
            is_valid, error_msg = _validate_url_secure(url)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"URL validation failed: {error_msg}"
                )
        
        # Fetch the URL
        fetch_success, metadata, error_msg = await _fetch_url(url)
        
        if fetch_success and metadata:
            url_metadata = URLMetadata(
                url=url,
                status_code=metadata.get("status_code"),
                content_length=metadata.get("content_length"),
                content_type=metadata.get("content_type"),
                fetch_successful=True,
                error_message=None
            )
            # Use URL metadata in analysis (e.g., status code, content length)
            # The original URL text is still analyzed by the models
            logger.info(f"Successfully fetched URL: {url} (status: {metadata.get('status_code')})")
        else:
            url_metadata = URLMetadata(
                url=url,
                status_code=None,
                content_length=None,
                content_type=None,
                fetch_successful=False,
                error_message=error_msg
            )
            logger.warning(f"Failed to fetch URL: {url} - {error_msg}")

    model_ids = _validate_model_ids(request.models)

    predictions, text_stats, total_time = await _run_models_for_text(
        text_to_analyze, model_ids
    )

    return PredictionResponse(
        predictions=predictions,
        text_stats=text_stats,
        total_processing_time_ms=round(total_time, 2),
        url_metadata=url_metadata,
    )


@app.post("/upload/txt-analyze", response_model=PredictionResponse)
async def analyze_txt_file(
    http_request: Request,
    file: UploadFile = File(...),
    models: str = Form("[\"xgboost\"]"),  # JSON string from form data
):
    """
    Analyze a .txt file for spam detection.
    
    Vulnerability #8 (Insecure File Upload): In Lab Mode, file validation is relaxed,
    allowing files with incorrect MIME types, larger file sizes, and storing files
    using the original filename. This makes the application vulnerable to file upload
    attacks, path traversal, and storage of malicious content.
    
    Secure Mode (strict validation):
    - Only .txt extension allowed
    - MIME type must be text/plain
    - Maximum file size: 100KB
    - Content read as UTF-8 text
    - File processed in memory and discarded (not stored)
    
    Lab Mode (vulnerable):
    - Only checks filename extension (.txt)
    - Skips MIME type validation
    - Significantly increased size limit (10MB)
    - Stores files using original filename in uploads/ folder
    
    Returns the same prediction response as /predict endpoint.
    """
    # Resolve mode using centralized resolver
    mode = resolve_mode(http_request)
    is_lab_mode = (mode == "lab")
    
    # Validate file extension (required in both modes)
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename"
        )
    
    filename_lower = file.filename.lower()
    if not filename_lower.endswith('.txt'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only .txt files are allowed. Received: {file.filename}"
        )
    
    # Validate MIME type based on mode
    if is_lab_mode:
        # Lab Mode: Skip MIME type validation (Vulnerability #8)
        # Trust only the filename extension, which can be easily spoofed
        logger.warning(f"⚠️ Lab Mode: Skipping MIME type validation for file: {file.filename}")
    else:
        # Secure Mode: Strict MIME type validation
        if file.content_type and file.content_type not in ('text/plain', 'text/plain; charset=utf-8'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Expected text/plain, got: {file.content_type}"
            )
    
    # File size limits based on mode
    if is_lab_mode:
        # Lab Mode: Significantly increased size limit (Vulnerability #8)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        logger.warning(f"⚠️ Lab Mode: Using relaxed file size limit (10MB) for file: {file.filename}")
    else:
        # Secure Mode: Strict size limit
        MAX_FILE_SIZE = 100 * 1024  # 100KB
    
    # Read file content
    try:
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE // (1024 * 1024) if MAX_FILE_SIZE >= 1024 * 1024 else MAX_FILE_SIZE // 1024}{'MB' if MAX_FILE_SIZE >= 1024 * 1024 else 'KB'}. File size: {len(content)} bytes"
            )
        
        # Decode as UTF-8 text
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File is not valid UTF-8 text: {str(e)}"
            )
        
        # Validate text is not empty
        if not text_content or not text_content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty or contains only whitespace"
            )
        
        # Validate text length (same as regular predict endpoint)
        if len(text_content) > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File content exceeds maximum length of 10000 characters"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading uploaded file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )
    
    # Store file in Lab Mode using original filename (Vulnerability #8)
    if is_lab_mode:
        try:
            # Create uploads directory if it doesn't exist
            uploads_dir = project_root / "backend" / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            
            # Use original filename (vulnerable to path traversal and filename trust issues)
            # In a real attack, an attacker could use filenames like "../../../etc/passwd" or "malicious.exe.txt"
            file_path = uploads_dir / file.filename
            
            # Write file content
            with open(file_path, 'wb') as f:
                f.write(content)
            
            logger.warning(f"⚠️ Lab Mode: Stored uploaded file using original filename: {file.filename} at {file_path}")
        except Exception as e:
            logger.error(f"Error storing file in Lab Mode: {str(e)}")
            # Don't fail the request if storage fails, just log it
    
    # Parse models from JSON string
    try:
        model_list = json.loads(models)
        if not isinstance(model_list, list):
            model_list = ["xgboost"]
    except (json.JSONDecodeError, TypeError):
        model_list = ["xgboost"]
    
    # Check if content is a URL and handle URL fetching (same logic as /predict)
    url_metadata = None
    text_to_analyze = text_content
    
    if _is_url(text_content):
        url = text_content.strip()
        
        # Validate URL based on mode
        if is_lab_mode:
            # Lab Mode: Skip or relax validation (SSRF vulnerability)
            try:
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Only http and https schemes are allowed, got: {parsed.scheme}"
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid URL format: {str(e)}"
                )
            
            logger.warning(f"⚠️ Lab Mode SSRF: Fetching URL without strict validation: {url}")
        else:
            # Secure Mode: Strict validation to prevent SSRF
            is_valid, error_msg = _validate_url_secure(url)
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"URL validation failed: {error_msg}"
                )
        
        # Fetch the URL
        fetch_success, metadata, error_msg = await _fetch_url(url)
        
        if fetch_success and metadata:
            url_metadata = URLMetadata(
                url=url,
                status_code=metadata.get("status_code"),
                content_length=metadata.get("content_length"),
                content_type=metadata.get("content_type"),
                fetch_successful=True,
                error_message=None
            )
            logger.info(f"Successfully fetched URL: {url} (status: {metadata.get('status_code')})")
        else:
            url_metadata = URLMetadata(
                url=url,
                status_code=None,
                content_length=None,
                content_type=None,
                fetch_successful=False,
                error_message=error_msg
            )
            logger.warning(f"Failed to fetch URL: {url} - {error_msg}")
    
    # Validate model IDs
    model_ids = _validate_model_ids(model_list)
    
    # Run predictions using existing pipeline
    predictions, text_stats, total_time = await _run_models_for_text(
        text_to_analyze, model_ids
    )
    
    # In Secure Mode: File is automatically discarded after this function returns
    # (processed in memory, never stored)
    # In Lab Mode: File is stored in uploads/ directory using original filename
    
    return PredictionResponse(
        predictions=predictions,
        text_stats=text_stats,
        total_processing_time_ms=round(total_time, 2),
        url_metadata=url_metadata,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict spam probabilities for a batch of texts."""

    batch_start = time.time()

    if not request.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one text must be provided",
        )

    model_ids = _validate_model_ids(request.models)

    items: List[BatchPredictionItem] = []
    summary_data: Dict[str, Dict[str, Any]] = {
        model_id: {
            "spam": 0,
            "ham": 0,
            "confidences": [],
            "spam_probabilities": [],
        }
        for model_id in model_ids
    }

    for index, text in enumerate(request.texts):
        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Text at index {index} cannot be empty",
            )

        if len(text) > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Text at index {index} exceeds maximum length of 10000 characters",
            )

        predictions, text_stats, processing_time = await _run_models_for_text(
            text, model_ids
        )

        items.append(
            BatchPredictionItem(
                text_index=index,
                predictions=predictions,
                text_stats=text_stats,
                processing_time_ms=processing_time,
            )
        )

        for prediction in predictions:
            model_summary = summary_data.get(prediction.model_id)
            if model_summary is None:
                continue

            if prediction.prediction == 1:
                model_summary["spam"] += 1
            else:
                model_summary["ham"] += 1

            model_summary["confidences"].append(prediction.confidence)
            model_summary["spam_probabilities"].append(prediction.spam_probability)

    summary = {
        "texts_processed": len(request.texts),
        "per_model": {},
    }

    for model_id, data in summary_data.items():
        total_predictions = data["spam"] + data["ham"]
        avg_confidence = (
            sum(data["confidences"]) / len(data["confidences"])
            if data["confidences"]
            else 0
        )
        avg_spam_probability = (
            sum(data["spam_probabilities"]) / len(data["spam_probabilities"])
            if data["spam_probabilities"]
            else 0
        )

        summary["per_model"][model_id] = {
            "predictions": total_predictions,
            "spam": data["spam"],
            "ham": data["ham"],
            "avg_confidence": round(avg_confidence, 4),
            "avg_spam_probability": round(avg_spam_probability, 4),
        }

    total_processing_time_ms = round((time.time() - batch_start) * 1000, 2)

    return BatchPredictionResponse(
        items=items,
        summary=summary,
        total_processing_time_ms=total_processing_time_ms,
    )


@app.post("/analysis/save", response_model=SavedAnalysisDetail)
async def save_analysis(
    request: SaveAnalysisRequest,
    http_request: Request,
    x_user_id: str = Depends(get_user_id_from_header),
    x_csrf_token: Optional[str] = Header(None, alias="X-CSRF-Token"),
    authorization: Optional[str] = Header(None)
):
    """
    Persist a completed analysis to SQLite.
    
    CSRF protection is enforced in Secure Mode via require_csrf_token.
    User ID validation is handled by get_user_id_from_header dependency.
    """
    
    # Get username from JWT token for CSRF validation
    username = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        payload = verify_token(token)
        if payload:
            username = payload.get("sub")
            logger.info(f"CSRF Debug: Extracted username from JWT: {username}")
        else:
            logger.warning("CSRF Debug: JWT token verification failed")
    else:
        logger.warning("CSRF Debug: No Authorization header or not Bearer token")
    
    # Validate CSRF token in Secure Mode
    if username:
        logger.info(f"CSRF Debug: Validating CSRF for user '{username}', token present: {bool(x_csrf_token)}")
        require_csrf_token(username, x_csrf_token, http_request, "POST")
    else:
        logger.warning("CSRF Debug: No username extracted, skipping CSRF validation")
    
    message_text = request.message_text
    if not message_text or not message_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message text cannot be empty",
        )

    # Validate and normalize selected models using existing rules
    model_ids = _validate_model_ids(request.selected_models)

    # Store selected models and prediction summary as JSON strings
    selected_models_json = json.dumps(model_ids)
    prediction_summary_json = (
        json.dumps(request.prediction_summary) if request.prediction_summary is not None else None
    )

    conn = _get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO saved_analyses (message_text, selected_models, prediction_summary, user_id)
            VALUES (?, ?, ?, ?)
            """,
            (message_text, selected_models_json, prediction_summary_json, x_user_id),
        )
        conn.commit()
        analysis_id = cursor.lastrowid

        # Fetch the full row including created_at
        cursor.execute(
            "SELECT id, message_text, selected_models, prediction_summary, created_at, user_id "
            "FROM saved_analyses WHERE id = ?",
            (analysis_id,),
        )
        row = cursor.fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save analysis",
        )

    # Resolve mode using centralized resolver
    mode = resolve_mode(http_request)
    
    # Include user_id in Lab Mode to make IDOR vulnerability visible
    user_id = row["user_id"] if mode == "lab" else None

    return SavedAnalysisDetail(
        id=row["id"],
        message_text=row["message_text"],
        selected_models=json.loads(row["selected_models"]),
        prediction_summary=json.loads(row["prediction_summary"])
        if row["prediction_summary"] is not None
        else None,
        created_at=str(row["created_at"]),
        user_id=user_id,
    )


@app.get("/analysis/list", response_model=List[SavedAnalysisSummary])
async def list_analyses(
    request: Request,
    x_user_id: str = Depends(get_user_id_from_header)
):
    """
    Return a list of saved analyses with minimal fields for history view.
    
    Access control is enforced by mode-aware filtering:
    - Secure Mode: Only returns analyses owned by the requesting user
    - Lab Mode: Returns all analyses (intentional IDOR vulnerability)
    """
    
    # Resolve mode using centralized resolver
    mode = resolve_mode(request)
    
    conn = _get_db_connection()
    try:
        cursor = conn.cursor()
        # In Lab Mode: show all analyses (IDOR vulnerability)
        # In Secure Mode: only show analyses owned by the user
        if mode == "lab":
            cursor.execute(
                """
                SELECT id, message_text, created_at, user_id
                FROM saved_analyses
                ORDER BY datetime(created_at) DESC, id DESC
                """
            )
        else:
            cursor.execute(
                """
                SELECT id, message_text, created_at
                FROM saved_analyses
                WHERE user_id = ?
                ORDER BY datetime(created_at) DESC, id DESC
                """,
                (x_user_id,),
            )
        rows = cursor.fetchall()
    finally:
        conn.close()

    summaries: List[SavedAnalysisSummary] = []
    for row in rows:
        message_text = row["message_text"] or ""
        snippet = message_text[:50]
        # Include user_id in Lab Mode to make IDOR vulnerability visible
        settings = get_settings()
        user_id = row["user_id"] if "user_id" in row.keys() and settings["lab_available"] else None
        summaries.append(
            SavedAnalysisSummary(
                id=row["id"],
                created_at=str(row["created_at"]),
                snippet=snippet,
                user_id=user_id,
            )
        )

    return summaries


@app.get("/analysis/{analysis_id}", response_model=SavedAnalysisDetail)
async def get_analysis(
    analysis_id: int,
    request: Request,
    analysis_data: Dict[str, Any] = Depends(require_analysis_access)
):
    """
    Return full details for a single saved analysis.
    
    Access control is enforced by the require_analysis_access dependency:
    - Secure Mode: Only allows access to analyses owned by the requesting user
    - Lab Mode: Allows access to any analysis (intentional IDOR vulnerability)
    """
    # Convert analysis_data dict to SavedAnalysisDetail
    # The dependency already handled all authorization and data fetching
    return SavedAnalysisDetail(**analysis_data)


async def predict_xgboost(text: str) -> Dict[str, Any]:
    """Predict using XGBoost model"""
    model = models["xgboost"]

    proba = model.predict_proba([text])[0]
    pred = model.predict([text])[0]

    spam_prob = float(proba)
    confidence = abs(spam_prob - 0.5) * 2  # 0-1 scale

    return {
        "model_id": "xgboost",
        "model_name": "XGBoost",
        "prediction": int(pred),
        "spam_probability": round(spam_prob, 4),
        "confidence": round(confidence, 4),
    }


async def predict_lstm(text: str) -> Dict[str, Any]:
    """Predict using LSTM model"""
    model = models["lstm"]

    proba = model.predict_proba([text])[0]
    pred = model.predict([text])[0]

    spam_prob = float(proba)
    confidence = abs(spam_prob - 0.5) * 2

    return {
        "model_id": "lstm",
        "model_name": "LSTM",
        "prediction": int(pred),
        "spam_probability": round(spam_prob, 4),
        "confidence": round(confidence, 4),
    }


async def predict_kmeans(text: str) -> Dict[str, Any]:
    """Predict using K-Means clustering model"""
    model = models["kmeans"]

    # Get cluster assignment and distances
    cluster_id, distances, point_2d = model.predict_with_details([text])

    # Debug logging
    print(
        f"K-Means prediction - cluster_id: {cluster_id}, point_2d: {point_2d}, PCA loaded: {model.pca_2d is not None}"
    )

    # Convert distances to spam probability
    # Assuming cluster 0 is ham, cluster 1+ are spam-like
    spam_clusters = [1, 2]  # Adjust based on your cluster analysis

    total_dist = sum(distances.values())
    if total_dist > 0:
        spam_prob = sum(distances.get(c, 0) for c in spam_clusters) / total_dist
        # Invert because closer distance = higher probability
        spam_prob = 1 - spam_prob
    else:
        spam_prob = 0.5

    # Determine prediction based on closest cluster
    prediction = 1 if cluster_id in spam_clusters else 0

    # Calculate confidence based on distance separation
    dist_values = list(distances.values())
    if len(dist_values) > 1:
        confidence = abs(dist_values[0] - dist_values[1]) / max(dist_values)
    else:
        confidence = 0.5

    return {
        "model_id": "kmeans",
        "model_name": "K-Means",
        "prediction": prediction,
        "spam_probability": round(float(spam_prob), 4),
        "confidence": round(float(confidence), 4),
        "cluster_id": int(cluster_id),
        "cluster_distances": {int(k): round(float(v), 4) for k, v in distances.items()},
        "user_point_2d": [round(float(x), 4) for x in point_2d] if point_2d else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
