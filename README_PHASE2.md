# Spam Detection System - Phase 2: Web Application

A full-stack web application for real-time spam detection using the trained machine learning models. The application consists of a FastAPI backend serving model predictions and a React frontend providing an interactive user interface.

## Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Environment Setup](#environment-setup)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Application Features](#application-features)
- [Project Structure](#project-structure)

## Overview

Phase 2 builds upon the trained models from Phase 1 to create a production-ready web application. The system provides:

- Real-time spam detection through a web interface
- Multi-model prediction support (XGBoost, LSTM, K-Means)
- Interactive visualizations of prediction results
- Model performance comparison
- Batch prediction capabilities
- RESTful API for integration

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **PyTorch**: Deep learning framework for LSTM model inference
- **XGBoost**: Gradient boosting library for classification
- **scikit-learn**: Machine learning utilities and K-Means clustering
- **Pydantic**: Data validation using Python type annotations

### Frontend
- **React 19.1.1**: Component-based UI library
- **Vite 7.1.7**: Fast build tool and development server
- **React Router DOM 7.9.5**: Client-side routing
- **Chart.js 4.5.1**: Data visualization library
- **D3.js 7.9.0**: Advanced data visualization
- **ESLint**: Code quality and linting

## Environment Setup

### Prerequisites

- Python 3.10 or higher
- Node.js 16.x or higher
- npm or yarn package manager
- CUDA-compatible GPU (optional, for faster inference)
- Trained models from Phase 1 (located in `models/` directory)

### System Requirements

- Minimum 8GB RAM
- 2GB available disk space
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Backend Setup

### 1. Python Environment Configuration

Create and activate a conda environment:

```bash
conda create --name spam-detection python=3.11
conda activate spam-detection
```

### 2. Install PyTorch with CUDA Support

For GPU acceleration (recommended):
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

For CPU-only installation:
```bash
pip3 install torch torchvision
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Key backend dependencies include:
- **fastapi**: Web framework for API endpoints
- **uvicorn[standard]**: ASGI server with auto-reload
- **pydantic**: Request/response validation
- **python-multipart**: Form data parsing
- **torch**: PyTorch for LSTM inference
- **xgboost==3.0.5**: XGBoost model inference
- **scikit-learn==1.7.2**: Machine learning utilities
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **joblib==1.5.2**: Model serialization

### 4. Verify Installation

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import fastapi; import uvicorn; print('Backend packages installed successfully')"
```

### 5. Verify Model Files

Ensure the following trained models exist:
- `models/lstm/deep_bilstm_chosen/` - LSTM model files
- `models/xgboost/` - XGBoost model files
- `models/kmeans/k3/tfidf_1000/` - K-Means model files

## Frontend Setup

### 1. Navigate to Frontend Directory

```bash
cd spam-detection-app
```

### 2. Install Node Dependencies

```bash
npm install
```

This installs:
- React and React DOM
- React Router for navigation
- Chart.js and react-chartjs-2 for visualizations
- D3.js for advanced charts
- Vite build tools
- ESLint for code quality

### 3. Verify Installation

```bash
npm list react vite
```

## Running the Application

### Starting the Backend Server

From the project root directory:

```bash
uvicorn backend.main:app --reload
```

The backend server will start on `http://localhost:8000`

**Server Options:**
- `--reload`: Auto-reload on code changes (development mode)
- `--host 0.0.0.0`: Allow external connections
- `--port 8000`: Specify port (default: 8000)

**Startup Output:**
```
Loading models...
✓ LSTM model loaded
✓ XGBoost model loaded
✓ K-Means model loaded
Loaded 3/3 models successfully
INFO:     Uvicorn running on http://localhost:8000
```

### Starting the Frontend Development Server

From the `spam-detection-app` directory:

```bash
npm run dev
```

The frontend will start on `http://localhost:5173`

**Development Server Features:**
- Hot Module Replacement (HMR)
- Fast refresh on code changes
- Source maps for debugging
- Optimized asset serving

**Startup Output:**
```
  VITE v7.1.7  ready in 320 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Accessing the Application

Open your web browser and navigate to:
```
http://localhost:5173
```

The application will automatically connect to the backend API at `http://localhost:8000`

### Production Build

To create an optimized production build:

```bash
cd spam-detection-app
npm run build
```

Build output will be generated in `spam-detection-app/dist/`

To preview the production build:
```bash
npm run preview
```

## API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### 1. Health Check

**GET** `/`

Check API status and available models.

**Response:**
```json
{
  "status": "ok",
  "message": "Spam Detection API is running",
  "available_models": ["lstm", "xgboost", "kmeans"],
  "endpoints": {
    "predict": "/predict",
    "model_info": "/models/info"
  }
}
```

#### 2. Get Model Information

**GET** `/models/info`

Retrieve metadata and performance metrics for all loaded models.

**Response:**
```json
{
  "lstm": {
    "status": "loaded",
    "metadata": {
      "accuracy": 0.97,
      "precision": 0.95,
      "recall": 0.97,
      "f1": 0.96
    },
    "description": "Deep Bidirectional LSTM - Captures context and word order"
  },
  "xgboost": {
    "status": "loaded",
    "metadata": {
      "accuracy": 0.92,
      "precision": 0.87,
      "recall": 0.90,
      "f1": 0.88
    },
    "description": "Gradient Boosting classifier - Fast and accurate"
  },
  "kmeans": {
    "status": "loaded",
    "metadata": {
      "silhouette_score": 0.15,
      "n_clusters": 3,
      "inertia": 0
    },
    "description": "Unsupervised clustering - Distance-based classification"
  }
}
```

#### 3. Single Text Prediction

**POST** `/predict`

Predict spam probability for a single text message.

**Request Body:**
```json
{
  "text": "Congratulations! You've won a $1000 gift card!",
  "models": ["lstm", "xgboost", "kmeans"]
}
```

**Request Parameters:**
- `text` (string, required): Message text to analyze (max 10,000 characters)
- `models` (array, optional): List of model IDs to use. Default: `["xgboost"]`

**Response:**
```json
{
  "predictions": [
    {
      "model_id": "lstm",
      "model_name": "LSTM",
      "prediction": 1,
      "spam_probability": 0.9534,
      "confidence": 0.9068,
      "processing_time_ms": 45.23
    },
    {
      "model_id": "xgboost",
      "model_name": "XGBoost",
      "prediction": 1,
      "spam_probability": 0.8921,
      "confidence": 0.7842,
      "processing_time_ms": 12.45
    },
    {
      "model_id": "kmeans",
      "model_name": "K-Means",
      "prediction": 1,
      "spam_probability": 0.7234,
      "confidence": 0.6543,
      "processing_time_ms": 8.67,
      "cluster_id": 2,
      "cluster_distances": {
        "0": 2.3456,
        "1": 1.2345,
        "2": 0.5678
      },
      "user_point_2d": [0.3456, 0.7890]
    }
  ],
  "text_stats": {
    "char_count": 49,
    "word_count": 8,
    "sentence_count": 1,
    "avg_word_length": 5.13
  },
  "total_processing_time_ms": 66.35
}
```

**Response Fields:**
- `prediction`: 0 = ham (legitimate), 1 = spam
- `spam_probability`: Probability of being spam (0.0 - 1.0)
- `confidence`: Model confidence in prediction (0.0 - 1.0)
- `cluster_id`: Assigned cluster (K-Means only)
- `cluster_distances`: Distance to each cluster center (K-Means only)
- `user_point_2d`: 2D projection coordinates for visualization (K-Means only)

#### 4. Batch Prediction

**POST** `/predict/batch`

Predict spam probabilities for multiple texts in a single request.

**Request Body:**
```json
{
  "texts": [
    "Congratulations! You've won a prize!",
    "Hey, are we still meeting tomorrow?",
    "Click here to claim your reward NOW!"
  ],
  "models": ["lstm", "xgboost"]
}
```

**Request Parameters:**
- `texts` (array, required): List of message texts to analyze
- `models` (array, optional): List of model IDs to use. Default: `["xgboost"]`

**Response:**
```json
{
  "items": [
    {
      "text_index": 0,
      "predictions": [...],
      "text_stats": {...},
      "processing_time_ms": 58.23
    },
    {
      "text_index": 1,
      "predictions": [...],
      "text_stats": {...},
      "processing_time_ms": 52.41
    },
    {
      "text_index": 2,
      "predictions": [...],
      "text_stats": {...},
      "processing_time_ms": 55.67
    }
  ],
  "summary": {
    "texts_processed": 3,
    "per_model": {
      "lstm": {
        "predictions": 3,
        "spam": 2,
        "ham": 1,
        "avg_confidence": 0.8234,
        "avg_spam_probability": 0.7123
      },
      "xgboost": {
        "predictions": 3,
        "spam": 2,
        "ham": 1,
        "avg_confidence": 0.7845,
        "avg_spam_probability": 0.6912
      }
    }
  },
  "total_processing_time_ms": 166.31
}
```

### Error Responses

#### 400 Bad Request
```json
{
  "detail": "Text input cannot be empty"
}
```

#### 422 Unprocessable Entity
```json
{
  "error": "Validation Error",
  "detail": "[error details]",
  "body": "[request body]"
}
```

#### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "detail": "An unexpected error occurred"
}
```

### Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These interfaces allow you to test API endpoints directly from the browser.

## Application Features

### 1. Home Page

**URL:** `http://localhost:5173/`

**Features:**
- Project overview and introduction
- Dataset statistics visualization
- Feature importance analysis chart
- Feature distribution visualization
- Correlation heatmap
- Quick navigation to spam detector

**Visualizations:**
- Dataset Overview: Shows distribution of spam vs. ham messages
- Feature Importance: Bar chart displaying the most influential features
- Feature Distribution: Histogram of feature values
- Correlation Heatmap: Matrix showing feature correlations

### 2. Spam Detector Page

**URL:** `http://localhost:5173/spam-detector`

**Features:**
- Text input area (up to 1,000 words)
- Paste text button with clipboard integration
- Model selection checkboxes (LSTM, XGBoost, K-Means)
- Real-time prediction with loading indicators
- Scroll-to-top navigation button
- Auto-scroll to results

**Input Validation:**
- Minimum 1 word required
- Maximum 1,000 words allowed
- At least one model must be selected
- Real-time word count display

**Prediction Results:**
Three interactive visualizations:

1. **Confidence Chart (Doughnut Chart)**
   - Visual representation of spam probability
   - Color-coded risk levels:
     - High Risk: Red (>70% spam probability)
     - Moderate Risk: Orange (40-70%)
     - Low Risk: Green (<40%)
   - Center displays spam percentage

2. **Model Comparison Chart (Bar Chart)**
   - Side-by-side comparison of model predictions
   - Metrics displayed:
     - Accuracy
     - Detection Rate (recall)
     - False Alarm Rate (1 - precision)
   - Color-coded bars for easy interpretation

3. **K-Means Cluster Analysis (Scatter Plot)**
   - Visualization of clustering results
   - Shows user input position relative to cluster centers
   - Displays distance to each cluster
   - Interactive zoom and pan capabilities

**Result Information:**
- Prediction label (SPAM or HAM)
- Spam probability percentage
- Model confidence score
- Processing time for each model
- Text statistics (character count, word count, etc.)

### 3. About Page

**URL:** `http://localhost:5173/about`

**Features:**
- Project background and objectives
- Team information
- Technology stack overview
- Links to documentation

### UI Components

**Header:**
- Navigation menu with active route highlighting
- Links to Home, Spam Detector, and About pages
- Responsive design for mobile devices

**Footer:**
- Copyright information
- Project metadata
- Additional links

## Project Structure

### Backend Structure

```
backend/
└── main.py                        # FastAPI application entry point
    ├── Model Loading              # Lifespan event for loading models on startup
    ├── API Routes                 # Endpoint definitions
    │   ├── GET /                  # Health check
    │   ├── GET /models/info       # Model metadata
    │   ├── POST /predict          # Single prediction
    │   └── POST /predict/batch    # Batch prediction
    ├── Request Models             # Pydantic schemas for validation
    │   ├── PredictionRequest      # Single text request
    │   ├── BatchPredictionRequest # Multiple texts request
    │   ├── ModelPrediction        # Model prediction response
    │   ├── PredictionResponse     # Single prediction response
    │   └── BatchPredictionResponse # Batch prediction response
    ├── Prediction Functions       # Model inference logic
    │   ├── predict_xgboost()      # XGBoost inference
    │   ├── predict_lstm()         # LSTM inference
    │   └── predict_kmeans()       # K-Means inference
    ├── Utility Functions          # Helper functions
    │   ├── _validate_model_ids()  # Model ID validation
    │   ├── _compute_text_stats()  # Text statistics computation
    │   └── _run_models_for_text() # Orchestrate multi-model prediction
    └── Exception Handlers         # Global error handling
        ├── validation_exception_handler()
        └── general_exception_handler()
```

**Backend File:** `backend/main.py`

**Key Components:**

1. **Model Loading (Lines 40-114)**
   - Loads all three models during application startup
   - Reads model metadata from JSON files
   - Provides graceful fallback if models fail to load

2. **CORS Configuration (Lines 123-135)**
   - Allows frontend connections from localhost:5173, 3000, 5174
   - Enables credentials and all HTTP methods

3. **Request Validation (Lines 138-177)**
   - Pydantic models ensure type safety
   - Automatic validation of request parameters
   - Detailed error messages for invalid inputs

4. **Prediction Endpoints (Lines 309-440)**
   - Single prediction: Processes one text at a time
   - Batch prediction: Processes multiple texts efficiently
   - Returns comprehensive results with timing information

### Frontend Structure

```
spam-detection-app/
├── public/                        # Static assets
│   ├── data/                      # Test data files
│   └── test-backend.html          # Backend testing utilities
│
├── src/
│   ├── main.jsx                   # React application entry point
│   ├── App.jsx                    # Root component with routing
│   │
│   ├── pages/                     # Page components (routes)
│   │   ├── Home.jsx               # Landing page with visualizations
│   │   ├── SpamDetector.jsx       # Main detection interface
│   │   └── About.jsx              # About page
│   │
│   ├── components/                # Reusable UI components
│   │   ├── Header.jsx             # Navigation header
│   │   ├── Footer.jsx             # Page footer
│   │   ├── ConfidenceChart.jsx    # Doughnut chart for spam probability
│   │   ├── ModelComparisonChart.jsx # Bar chart for model comparison
│   │   ├── KMeansClusterChart.jsx # Scatter plot for clustering
│   │   ├── DatasetOverview.jsx    # Dataset statistics
│   │   ├── FeatureImportanceChart.jsx # Feature importance bar chart
│   │   ├── FeatureDistributionChart.jsx # Feature distribution histogram
│   │   └── CorrelationHeatmap.jsx # Feature correlation matrix
│   │
│   └── styles/                    # CSS stylesheets
│       ├── index.css              # Main stylesheet entry
│       ├── base.css               # Base styles and resets
│       ├── layout.css             # Layout and grid
│       ├── components.css         # Component styles
│       ├── home.css               # Home page styles
│       ├── about.css              # About page styles
│       ├── spam-detector.css      # Spam detector page styles
│       ├── confidence-chart.css   # Confidence chart styles
│       ├── model-comparison-chart.css # Model comparison styles
│       ├── kmeans-cluster-chart.css # K-Means chart styles
│       ├── visualizations.css     # General visualization styles
│       └── responsive.css         # Responsive design
│
├── config/
│   ├── vite.config.js             # Vite build configuration
│   └── eslint.config.js           # ESLint rules
│
├── index.html                     # HTML entry point
├── package.json                   # Dependencies and scripts
└── package-lock.json              # Dependency lock file
```

### Key Frontend Files

**App.jsx** (`spam-detection-app/src/App.jsx`)
- React Router configuration
- Route definitions for all pages
- Header and Footer layout

**SpamDetector.jsx** (`spam-detection-app/src/pages/SpamDetector.jsx`)
- Main spam detection interface
- API integration with backend
- User input handling
- Results visualization

**Chart Components** (`spam-detection-app/src/components/`)
- Reusable visualization components
- Chart.js and D3.js integration
- Interactive features (zoom, tooltips)

### Configuration Files

**Backend Configuration:**
- Model paths are defined in `backend/main.py` (lines 49, 69, 89)
- CORS origins configured in `backend/main.py` (lines 126-131)
- API settings in FastAPI app initialization (lines 116-121)

**Frontend Configuration:**
- Vite config: `spam-detection-app/config/vite.config.js`
- ESLint rules: `spam-detection-app/config/eslint.config.js`
- Backend API URL: Hardcoded in `SpamDetector.jsx` as `http://localhost:8000/predict`

### Development Workflow

1. **Make Backend Changes**
   - Edit files in `backend/` or `src/` directories
   - Uvicorn auto-reloads on file changes (with `--reload` flag)
   - Test API endpoints at `http://localhost:8000/docs`

2. **Make Frontend Changes**
   - Edit files in `spam-detection-app/src/`
   - Vite Hot Module Replacement applies changes instantly
   - View changes at `http://localhost:5173`

3. **Code Quality**
   - Run ESLint: `npm run lint` (from `spam-detection-app/`)
   - Check console for warnings and errors
   - Fix linting issues before committing

4. **Testing**
   - Test backend endpoints using Swagger UI at `/docs`
   - Use `public/test-backend.html` for manual testing
   - Test frontend in multiple browsers

### Deployment Considerations

**Backend Deployment:**
- Remove `--reload` flag for production
- Set `host="0.0.0.0"` for external access
- Configure environment variables for sensitive data
- Use production-grade ASGI server (Gunicorn + Uvicorn workers)
- Add rate limiting and authentication middleware

**Frontend Deployment:**
- Build production bundle: `npm run build`
- Serve static files from `dist/` directory
- Update API URL for production backend
- Configure CORS to allow production domain
- Enable HTTPS for secure communication

**Environment Variables:**
Consider creating `.env` files for:
- Backend API URL
- Model paths
- CORS origins
- Port numbers
- Debug mode flags
