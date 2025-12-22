from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
from scipy import sparse

import json
from src.kmeans.feature_engineer import extract_text_features


class KMeansTextInferencer:
    def __init__(self, model_dir: Union[str, Path]):
        """
        Initialize K-Means text inferencer for spam detection.

        Loads trained K-Means model, cluster-to-class mapping, and preprocessing
        artifacts (TF-IDF vectorizer, scaler, feature metadata) from the specified
        model directory. The inferencer can then be used for clustering and
        classification of new text data.

        Args:
            model_dir (Union[str, Path]): Path to directory containing:
                - kmeans_model.pkl: Trained K-Means clustering model
                - cluster_mapping.pkl: Mapping from cluster IDs to class labels
                - tfidf_vectorizer.pkl: TF-IDF vectorizer for text preprocessing
                - scaler.pkl: Feature scaler for normalization
                - feature_metadata.json: Feature engineering metadata

        Raises:
            FileNotFoundError: If required model artifacts are missing
        """
        self.model_dir = Path(model_dir)

        # Load trained kmeans model and cluster mapping (cluster -> class)
        self.model = joblib.load(self.model_dir / "kmeans_model.pkl")
        self.cluster_mapping = joblib.load(self.model_dir / "cluster_mapping.pkl")

        # Normalize mapping keys to int
        self.cluster_mapping = {int(k): int(v) for k, v in self.cluster_mapping.items()}

        # Load feature artifacts from model directory for standalone inference
        self.tfidf_vectorizer = joblib.load(self.model_dir / "tfidf_vectorizer.pkl")
        self.scaler = joblib.load(self.model_dir / "scaler.pkl")
        with open(self.model_dir / "feature_metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        # Load PCA model for 2D visualization (if available)
        pca_path = self.model_dir / "pca_2d.pkl"
        if pca_path.exists():
            self.pca_2d = joblib.load(pca_path)
        else:
            # Try loading from preprocessed directory (fallback)
            preprocessed_dir = self.model_dir.parent.parent.parent / "preprocessed" / "kmeans" / "tfidf_1000"
            fallback_pca_path = preprocessed_dir / "pca_2d.pkl"
            if fallback_pca_path.exists():
                self.pca_2d = joblib.load(fallback_pca_path)
            else:
                self.pca_2d = None
                print("Warning: PCA model not found. 2D coordinates will not be available.")

    def predict_clusters(
        self, texts: List[str], sources: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Predict cluster assignments for input texts.

        Processes texts through the same feature engineering pipeline used during
        training (TF-IDF + engineered features + scaling) and assigns each text
        to the nearest cluster centroid.

        Args:
            texts (List[str]): List of text messages to cluster
            sources (Optional[List[str]]): Optional list of source identifiers
                for each text. If None, defaults to "unknown" for all texts.

        Returns:
            np.ndarray: Array of cluster IDs (integers) for each input text
        """
        X = self._build_kmeans_feature_matrix(
            texts, sources, self.tfidf_vectorizer, self.scaler, self.metadata
        )
        return self.model.predict(X)

    def predict(
        self, texts: List[str], sources: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Predict binary spam/ham labels for input texts.

        First assigns texts to clusters, then maps cluster assignments to
        binary class labels (0=ham, 1=spam) using the learned cluster mapping.

        Args:
            texts (List[str]): List of text messages to classify
            sources (Optional[List[str]]): Optional list of source identifiers
                for each text. If None, defaults to "unknown" for all texts.

        Returns:
            np.ndarray: Array of binary predictions (0=ham, 1=spam)
        """
        clusters = self.predict_clusters(texts, sources)
        preds = np.array(
            [self.cluster_mapping.get(int(c), 0) for c in clusters], dtype=int
        )
        return preds

    def _engineer_text_features(self, texts: List[str]) -> np.ndarray:
        """
        Engineer text-based features from input texts.

        Extracts various statistical and linguistic features from texts such as
        character counts, word counts, punctuation ratios, etc. These features
        complement the TF-IDF representation for better clustering performance.

        Args:
            texts (List[str]): List of input texts to process

        Returns:
            np.ndarray: Feature matrix of shape (n_texts, n_engineered_features)
                containing engineered text features
        """
        feats, _ = extract_text_features(texts)
        return feats

    def _create_source_features(
        self, sources: List[str], all_sources: List[str]
    ) -> np.ndarray:
        """
        Create one-hot encoded source features.

        Converts source identifiers into one-hot encoded vectors where each
        position corresponds to a specific source type. This allows the model
        to learn source-specific patterns in the data.

        Args:
            sources (List[str]): List of source identifiers for each text
            all_sources (List[str]): Complete list of all possible sources
                (determines the feature vector length)

        Returns:
            np.ndarray: One-hot encoded feature matrix of shape (n_texts, n_sources)
                where each row has exactly one 1.0 and the rest are 0.0
        """
        source_idx = {s: i for i, s in enumerate(all_sources)}
        rows = []
        for src in sources:
            vec = [0] * len(all_sources)
            if src in source_idx:
                vec[source_idx[src]] = 1
            rows.append(vec)
        return np.array(rows, dtype=float)

    def _build_kmeans_feature_matrix(
        self,
        texts: List[str],
        sources: Optional[List[str]],
        tfidf_vectorizer,
        scaler,
        metadata: dict,
    ) -> np.ndarray:
        """
        Build complete feature matrix for K-Means clustering.

        Combines TF-IDF features, engineered text features, and source features
        into a single normalized feature matrix. The matrix is scaled using the
        same scaler from training to ensure consistency.

        Args:
            texts (List[str]): Input texts to process
            sources (Optional[List[str]]): Source identifiers for each text
            tfidf_vectorizer: Fitted TF-IDF vectorizer from training
            scaler: Fitted feature scaler from training
            metadata (dict): Feature metadata containing source list and config

        Returns:
            np.ndarray: Normalized feature matrix of shape (n_texts, n_features)
                ready for K-Means clustering
        """
        # TF-IDF dense (training used dense concatenation before scaling)
        tfidf_sparse = tfidf_vectorizer.transform(texts)
        tfidf_dense = tfidf_sparse.toarray()

        # Engineered text features
        text_feats = self._engineer_text_features(texts)

        # Source features
        all_sources = metadata.get("sources", [])
        sources = sources if sources is not None else ["unknown"] * len(texts)
        source_feats = self._create_source_features(sources, all_sources)

        combined = np.hstack([tfidf_dense, text_feats, source_feats])
        combined_scaled = scaler.transform(combined)
        return combined_scaled

    def predict_with_details(
        self, texts: List[str], sources: Optional[List[str]] = None
    ) -> tuple:
        """
        Predict with cluster assignment, distances, and 2D coordinates for visualization.
        
        Args:
            texts (List[str]): List of text messages to cluster
            sources (Optional[List[str]]): Optional list of source identifiers
        
        Returns:
            tuple: (cluster_id, distances_dict, point_2d_coords)
                - cluster_id: Assigned cluster ID (int)
                - distances_dict: Dict mapping cluster_id -> distance (float)
                - point_2d_coords: [x, y] coordinates in normalized 0-100 range, or None if PCA unavailable
        """
        # Build feature matrix
        X = self._build_kmeans_feature_matrix(
            texts, sources, self.tfidf_vectorizer, self.scaler, self.metadata
        )
        
        # Predict cluster
        cluster_id = int(self.model.predict(X)[0])
        
        # Calculate distances to all centroids
        distances = {}
        for i, centroid in enumerate(self.model.cluster_centers_):
            dist = np.linalg.norm(X[0] - centroid)
            distances[int(i)] = float(dist)
        
        # Get 2D coordinates if PCA model is available
        point_2d = None
        if self.pca_2d is not None:
            # Transform to 2D using PCA
            point_2d_raw = self.pca_2d.transform(X)[0]
            
            # Get normalization ranges from metadata
            norm_ranges = self.metadata.get("pca_2d_normalization", {})
            x_min = norm_ranges.get("x_min", -5.0)
            x_max = norm_ranges.get("x_max", 12.0)
            y_min = norm_ranges.get("y_min", -5.0)
            y_max = norm_ranges.get("y_max", 8.0)
            
            # Normalize to 0-100 range
            x_norm = ((point_2d_raw[0] - x_min) / (x_max - x_min)) * 100 if (x_max - x_min) > 0 else 50.0
            y_norm = ((point_2d_raw[1] - y_min) / (y_max - y_min)) * 100 if (y_max - y_min) > 0 else 50.0
            
            # Clamp to 0-100
            x_norm = max(0.0, min(100.0, x_norm))
            y_norm = max(0.0, min(100.0, y_norm))
            
            point_2d = [float(x_norm), float(y_norm)]
        
        return cluster_id, distances, point_2d
