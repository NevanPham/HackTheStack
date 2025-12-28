import sys
from pathlib import Path
import time
import json
import sqlite3
import os
from datetime import datetime, timedelta
from secrets import token_urlsafe

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import bcrypt
from jose import JWTError, jwt

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, status, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

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

# Lab Mode configuration (server-side master switch)
# If set to false, Lab Mode is disabled regardless of client header (safety override)
# If set to true (default), Lab Mode is controlled by X-Lab-Mode header from frontend
LAB_MODE_ENABLED = os.getenv("LAB_MODE_ENABLED", "true").lower() in ("true", "1", "yes")

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
    
    # Log Lab Mode configuration
    if LAB_MODE_ENABLED:
        print("⚠️  LAB MODE ALLOWED - IDOR vulnerability can be toggled via frontend")
        logger.info("Lab Mode is allowed - Frontend can toggle IDOR vulnerability via X-Lab-Mode header")
    else:
        print("✓ Secure Mode Only - Lab Mode disabled (set LAB_MODE_ENABLED=true to enable)")
        logger.info("Secure Mode Only - Lab Mode disabled by environment variable")

    yield

    # Cleanup
    print("Shutting down...")


app = FastAPI(
    title="Spam Detection API",
    description="Multi-model spam detection API supporting XGBoost, LSTM, and K-Means",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class PredictionResponse(BaseModel):
    predictions: List[ModelPrediction]
    text_stats: Dict[str, Any]
    total_processing_time_ms: float


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


def check_rate_limit(identifier: str, is_lab_mode: bool) -> Optional[str]:
    """
    Check if the identifier is rate limited.
    
    Vulnerability #6 (Weak Authentication): In Lab Mode, rate limiting is bypassed,
    allowing unlimited failed login attempts. This makes the application vulnerable
    to brute force attacks.
    
    Args:
        identifier: Unique identifier for rate limiting (IP:username)
        is_lab_mode: If True, bypass rate limiting (Lab Mode)
    
    Returns:
        Error message if rate limited, None otherwise
    """
    if is_lab_mode:
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


def record_failed_attempt(identifier: str, is_lab_mode: bool) -> None:
    """
    Record a failed login attempt.
    
    Args:
        identifier: Unique identifier for rate limiting (IP:username)
        is_lab_mode: If True, don't record attempts (Lab Mode)
    """
    if is_lab_mode:
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
    x_lab_mode: Optional[str],
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
        x_lab_mode: The X-Lab-Mode header value
        method: HTTP method (POST, PUT, PATCH, DELETE)
    
    Raises:
        HTTPException: If CSRF validation fails in Secure Mode
    """
    # Determine Lab Mode using server-side gating (env var + header)
    # This ensures the vulnerability is only active when explicitly enabled
    is_lab_mode = LAB_MODE_ENABLED and x_lab_mode and x_lab_mode.lower() in ("true", "1", "yes")
    
    if is_lab_mode:
        # Lab Mode: CSRF protection is bypassed (Vulnerability #4)
        # State-changing requests succeed without CSRF token validation
        return
    
    # Secure Mode: Require CSRF token for state-changing methods
    # This prevents CSRF attacks by validating the token matches the user's session
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        if not validate_csrf_token(username, csrf_token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token validation failed. Please refresh the page and try again.",
            )


def authenticate_user(username: str, password: str, is_lab_mode: bool = False) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user by username and password.
    
    Vulnerability #5 (SQL Injection): In Lab Mode, the authentication query uses unsafe
    string interpolation for both username and password, allowing SQL injection attacks.
    An attacker can manipulate either field to bypass authentication.
    
    Args:
        username: The username to authenticate
        password: The password to verify
        is_lab_mode: If True, use vulnerable SQL query construction (Lab Mode)
    
    Returns:
        Dict with user id and username if authentication succeeds, None otherwise
    """
    conn = _get_db_connection()
    try:
        cursor = conn.cursor()
        
        if is_lab_mode:
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
        
        if is_lab_mode:
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
    x_lab_mode: Optional[str] = Header(None, alias="X-Lab-Mode")
):
    """
    Authenticate user and return JWT token.
    
    Vulnerability #5 (SQL Injection): In Lab Mode, the authentication function uses
    unsafe SQL query construction, making it vulnerable to SQL injection attacks.
    
    Vulnerability #6 (Weak Authentication): In Lab Mode, rate limiting is bypassed,
    allowing unlimited failed login attempts and making brute force attacks possible.
    """
    # Determine Lab Mode using server-side gating (env var + header)
    is_lab_mode = LAB_MODE_ENABLED and x_lab_mode and x_lab_mode.lower() in ("true", "1", "yes")
    
    # Get client identifier for rate limiting
    identifier = get_client_identifier(http_request, request.username)
    
    # Check rate limit in Secure Mode (Vulnerability #6: bypassed in Lab Mode)
    rate_limit_error = check_rate_limit(identifier, is_lab_mode)
    if rate_limit_error:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=rate_limit_error,
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = authenticate_user(request.username, request.password, is_lab_mode=is_lab_mode)
    if not user:
        # Record failed attempt in Secure Mode (Vulnerability #6: not recorded in Lab Mode)
        record_failed_attempt(identifier, is_lab_mode)
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
    credentials: HTTPAuthorizationCredentials = security
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
    
    return {"authenticated": True, "can_access_lab": True, "username": username}


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
async def get_model_info():
    """Get information about all loaded models"""
    info = {}

    for model_id, model in models.items():
        if model is not None:
            info[model_id] = {
                "status": "loaded",
                "metadata": model_metadata.get(model_id, {}),
                "description": {
                    "xgboost": "Gradient Boosting classifier - Fast and accurate",
                    "lstm": "Deep Bidirectional LSTM - Captures context and word order",
                    "kmeans": "Unsupervised clustering - Distance-based classification",
                }.get(model_id, ""),
            }
        else:
            info[model_id] = {"status": "not_loaded", "error": "Model failed to load"}

    return info


# Add global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": str(exc.errors()),
            "body": str(exc.body),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
        },
    )


# Enhance predict endpoint with validation
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict spam probability using selected models

    - **text**: The message text to analyze
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

    model_ids = _validate_model_ids(request.models)

    predictions, text_stats, total_time = await _run_models_for_text(
        request.text, model_ids
    )

    return PredictionResponse(
        predictions=predictions,
        text_stats=text_stats,
        total_processing_time_ms=round(total_time, 2),
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
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    x_lab_mode: Optional[str] = Header(None, alias="X-Lab-Mode"),
    x_csrf_token: Optional[str] = Header(None, alias="X-CSRF-Token"),
    authorization: Optional[str] = Header(None)
):
    """Persist a completed analysis to SQLite."""
    # Require user_id header
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-User-Id header is required",
        )
    
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
        logger.info(f"CSRF Debug: Validating CSRF for user '{username}', token present: {bool(x_csrf_token)}, lab_mode: {x_lab_mode}")
        require_csrf_token(username, x_csrf_token, x_lab_mode, "POST")
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

    # Check if Lab Mode is enabled (env var must allow it AND header must request it)
    is_lab_mode = LAB_MODE_ENABLED and x_lab_mode and x_lab_mode.lower() in ("true", "1", "yes")
    
    # Include user_id in Lab Mode to make IDOR vulnerability visible
    user_id = row["user_id"] if is_lab_mode else None

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
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    x_lab_mode: Optional[str] = Header(None, alias="X-Lab-Mode")
):
    """Return a list of saved analyses with minimal fields for history view."""
    # Require user_id header
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-User-Id header is required",
        )
    
    # Check if Lab Mode is enabled (env var must allow it AND header must request it)
    is_lab_mode = LAB_MODE_ENABLED and x_lab_mode and x_lab_mode.lower() in ("true", "1", "yes")
    
    conn = _get_db_connection()
    try:
        cursor = conn.cursor()
        # In Lab Mode: show all analyses (IDOR vulnerability)
        # In Secure Mode: only show analyses owned by the user
        if is_lab_mode:
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
        user_id = row["user_id"] if "user_id" in row.keys() and LAB_MODE_ENABLED else None
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
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    x_lab_mode: Optional[str] = Header(None, alias="X-Lab-Mode")
):
    """Return full details for a single saved analysis."""
    # Require user_id header
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-User-Id header is required",
        )
    
    # Check if Lab Mode is enabled (env var must allow it AND header must request it)
    is_lab_mode = LAB_MODE_ENABLED and x_lab_mode and x_lab_mode.lower() in ("true", "1", "yes")
    
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
    
    # IDOR Vulnerability: In Lab Mode, bypass ownership check
    # Secure Mode: Enforce ownership - return 403 if analysis belongs to another user
    if not is_lab_mode and row["user_id"] != x_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to analysis {analysis_id}",
        )

    # Include user_id in Lab Mode to make IDOR vulnerability visible
    user_id = row["user_id"] if is_lab_mode else None
    
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
