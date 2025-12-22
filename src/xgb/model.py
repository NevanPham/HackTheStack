import json
from pathlib import Path
from typing import List, Tuple, Optional, Union

import joblib
import numpy as np
from scipy import sparse

from src.xgb.feature_engineer import (
    extract_text_statistics,
    extract_special_character_features,
    extract_source_features,
)


class XGBTextClassifier:
    def __init__(self, model_dir: Union[str, Path]):
        """
        Initialize XGBoost text classifier for spam detection.

        Loads trained XGBoost model, optimal threshold, and preprocessing artifacts
        (TF-IDF vectorizer, feature metadata) from the specified model directory.
        The classifier can then be used for inference on new text data.

        Args:
            model_dir (Union[str, Path]): Path to directory containing:
                - xgb_model_randomized.pkl: Trained XGBoost model
                - *results_summary*.json: Results with optimal threshold
                - tfidf_vectorizer.pkl: TF-IDF vectorizer for text preprocessing
                - feature_metadata.json: Feature engineering metadata

        Raises:
            FileNotFoundError: If required preprocessing artifacts are missing
        """
        model_path = Path(model_dir)
        self.model = joblib.load(model_path / "xgb_model_randomized.pkl")

        # Load optimal threshold if available; fallback to 0.5
        threshold = 0.5
        summary_json = list(model_path.glob("*results_summary*.json"))
        if summary_json:
            try:
                with open(summary_json[0], "r") as f:
                    summary = json.load(f)
                threshold = float(summary.get("optimal_threshold", 0.5))
            except Exception:
                threshold = 0.5
        self.threshold = threshold

        # Prefer artifacts saved alongside model for standalone loading
        try:
            with open(model_path / "feature_metadata.json", "r") as f:
                self.metadata = json.load(f)
            self.tfidf_vectorizer = joblib.load(model_path / "tfidf_vectorizer.pkl")
        except Exception:
            raise FileNotFoundError(
                "Missing preprocessing artifacts in model directory. Ensure training saved tfidf_vectorizer.pkl and feature_metadata.json."
            )

    def predict_proba(
        self, texts: List[str], sources: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Predict spam probabilities for input texts.

        Processes texts through the same feature engineering pipeline used during
        training (TF-IDF + engineered features) and returns probability scores
        for each text being spam.

        Args:
            texts (List[str]): List of text messages to classify
            sources (Optional[List[str]]): Optional list of source identifiers
                for each text. If None, defaults to "unknown" for all texts.

        Returns:
            np.ndarray: Array of spam probabilities in range [0, 1], where
                1.0 indicates high confidence spam and 0.0 indicates ham.
        """
        X = self._build_feature_matrix(
            texts, sources, self.tfidf_vectorizer, self.metadata
        )
        proba = self.model.predict_proba(X)[:, 1]
        return proba

    def predict(
        self, texts: List[str], sources: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Predict binary spam/ham labels for input texts.

        Uses the optimal threshold determined during training to convert
        probability scores into binary predictions (0=ham, 1=spam).

        Args:
            texts (List[str]): List of text messages to classify
            sources (Optional[List[str]]): Optional list of source identifiers
                for each text. If None, defaults to "unknown" for all texts.

        Returns:
            np.ndarray: Array of binary predictions (0=ham, 1=spam)
        """
        proba = self.predict_proba(texts, sources)
        return (proba >= self.threshold).astype(int)

    def _engineer_numeric_features(
        self, texts: List[str], sources: Optional[List[str]], metadata: dict
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Engineer numeric features from texts and sources for model input.

        Recreates the same feature engineering pipeline used during training:
        - Text statistics (length, word count, etc.)
        - Special character features (punctuation, digits, etc.)
        - Source one-hot encoding features

        Args:
            texts (List[str]): Input texts to process
            sources (Optional[List[str]]): Source identifiers for each text
            metadata (dict): Feature metadata containing source list and config

        Returns:
            Tuple[np.ndarray, List[str]]:
                - Feature matrix (n_samples, n_numeric_features)
                - List of feature names for interpretability
        """
        """
        Recreate numeric features using the same feature functions as training
        for parity: extract_text_statistics, extract_special_character_features,
        and extract_source_features. Order matches training combine order.
        """
        # Stats and special-character features
        stats, stat_names = extract_text_statistics(texts)
        special, special_names = extract_special_character_features(texts)

        # Source one-hot in training order
        all_sources = metadata.get("sources", [])
        src_list = sources if sources is not None else ["unknown"] * len(texts)
        source_feats, source_names = extract_source_features(src_list, all_sources)

        numeric = np.hstack([stats, special, source_feats])
        feature_names = stat_names + special_names + source_names
        return numeric, feature_names

    def _build_feature_matrix(
        self,
        texts: List[str],
        sources: Optional[List[str]],
        tfidf_vectorizer,
        metadata: dict,
    ):
        """
        Build complete feature matrix combining TF-IDF and engineered features.

        Combines sparse TF-IDF features with dense engineered numeric features
        to create the final feature matrix that matches the training format.
        This ensures consistency between training and inference.

        Args:
            texts (List[str]): Input texts to process
            sources (Optional[List[str]]): Source identifiers for each text
            tfidf_vectorizer: Fitted TF-IDF vectorizer from training
            metadata (dict): Feature metadata for engineered features

        Returns:
            scipy.sparse.csr_matrix: Combined sparse feature matrix ready
                for XGBoost prediction
        """
        # TF-IDF (same vectorizer as training)
        tfidf = tfidf_vectorizer.transform(texts)

        # Numeric features
        numeric, _ = self._engineer_numeric_features(texts, sources, metadata)
        numeric_sparse = sparse.csr_matrix(numeric)

        # Combine horizontally
        X = sparse.hstack([tfidf, numeric_sparse])
        return X
