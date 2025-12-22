import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import List, Union
import numpy as np
import pickle

from src.lstm.preprocessing import clean_text_for_lstm, tokenize_text


class LSTMModel(nn.Module):
    """Configurable (Bi)LSTM model with optional attention for spam detection."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dims=[128, 64],  # List of LSTM hidden dimensions
        dense_layers=[64, 32],  # List of dense layer dimensions
        dropout_rate=0.3,
        bidirectional=True,
        use_attention=False,
        attention_dim=64,
        max_length=128,
        num_classes=1,  # Binary classification
        embedding_trainable=True,
    ):
        """
        Initialize configurable LSTM model for spam detection.

        Creates a flexible neural network architecture with:
        - Embedding layer for word representations
        - Stacked LSTM layers (optionally bidirectional)
        - Optional multi-head self-attention mechanism
        - Dense layers for final classification
        - Dropout for regularization

        Args:
            vocab_size (int): Size of vocabulary for embedding layer
            embedding_dim (int): Dimension of word embeddings (default: 128)
            hidden_dims (List[int]): Hidden dimensions for each LSTM layer (default: [128, 64])
            dense_layers (List[int]): Dimensions for dense layers (default: [64, 32])
            dropout_rate (float): Dropout rate for regularization (default: 0.3)
            bidirectional (bool): Whether to use bidirectional LSTM (default: True)
            use_attention (bool): Whether to use multi-head attention (default: False)
            attention_dim (int): Dimension for attention mechanism (default: 64)
            max_length (int): Maximum sequence length for padding (default: 128)
            num_classes (int): Number of output classes (default: 1 for binary)
            embedding_trainable (bool): Whether embeddings are trainable (default: True)
        """
        super(LSTMModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dense_layers = dense_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.max_length = max_length
        self.num_classes = num_classes

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.requires_grad = embedding_trainable

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        input_dim = embedding_dim

        for i, hidden_dim in enumerate(hidden_dims):
            lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if i < len(hidden_dims) - 1 else 0,
            )
            self.lstm_layers.append(lstm)
            # Update input dimension for next layer
            input_dim = hidden_dim * (2 if bidirectional else 1)

        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True,
            )
            self.attention_linear = nn.Linear(input_dim, attention_dim)
            final_lstm_dim = attention_dim
        else:
            final_lstm_dim = input_dim

        # Dense layers
        self.dense_layers_list = nn.ModuleList()
        input_dim = final_lstm_dim

        for dense_dim in dense_layers:
            self.dense_layers_list.append(nn.Linear(input_dim, dense_dim))
            self.dense_layers_list.append(nn.ReLU())
            self.dense_layers_list.append(nn.Dropout(dropout_rate))
            input_dim = dense_dim

        # Output layer
        self.output_layer = nn.Linear(input_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Processes input token sequences through the complete model architecture:
        1. Embedding layer converts token indices to dense vectors
        2. LSTM layers process sequential information (optionally bidirectional)
        3. Optional attention mechanism focuses on important tokens
        4. Dense layers perform final classification
        5. Sigmoid activation outputs spam probability

        Args:
            x (torch.LongTensor): Token indices of shape (batch_size, seq_len)
                where each value is a vocabulary index

        Returns:
            torch.FloatTensor: Spam probabilities of shape (batch_size,) in range [0, 1]
                where 1.0 indicates high confidence spam and 0.0 indicates ham
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)

        # LSTM layers
        lstm_out = embedded
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(lstm_out)
            lstm_out = self.dropout(lstm_out)

        # Attention mechanism (if enabled)
        if self.use_attention:
            # Self-attention
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Global average pooling with attention weights
            attended = self.attention_linear(attn_out)
            pooled = torch.mean(attended, dim=1)  # (batch_size, attention_dim)
        else:
            # Use last output or global max pooling
            if self.bidirectional:
                # Global max pooling for bidirectional LSTM
                pooled = torch.max(lstm_out, dim=1)[0]  # (batch_size, hidden_dim*2)
            else:
                # Use last output
                pooled = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Dense layers
        dense_out = pooled
        for layer in self.dense_layers_list:
            dense_out = layer(dense_out)

        # Output
        output = self.output_layer(dense_out)
        output = self.sigmoid(output)

        return output.squeeze()


class LSTMTextClassifier:
    def __init__(self, model_dir: Union[str, Path]):
        """
        Initialize LSTM text classifier for spam detection.

        Loads trained LSTM model, vocabulary, and preprocessing artifacts from
        the specified model directory. The classifier can then be used for
        inference on new text data.

        Args:
            model_dir (Union[str, Path]): Path to directory containing:
                - best_model.pth or final_model.pth: Trained model weights
                - model_info.json: Model architecture configuration
                - vocab.pkl: Vocabulary mapping (token -> index)
                - lstm_metadata.json: Preprocessing metadata (max_length, etc.)

        Raises:
            FileNotFoundError: If required model artifacts are missing
        """
        self.model, self.vocab, self.max_length = self._load_lstm_artifacts(model_dir)

    @torch.no_grad()
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict spam probabilities for input texts.

        Processes texts through the same preprocessing pipeline used during
        training (cleaning, tokenization, padding) and returns probability scores
        for each text being spam. Uses torch.no_grad() for efficient inference.

        Args:
            texts (List[str]): List of text messages to classify

        Returns:
            np.ndarray: Array of spam probabilities in range [0, 1], where
                1.0 indicates high confidence spam and 0.0 indicates ham
        """
        inputs = [self._text_to_indices(t, self.vocab, self.max_length) for t in texts]
        X = torch.LongTensor(inputs)
        logits = self.model(X)
        # Model already applies sigmoid and returns probabilities in [0,1]
        proba = logits.view(-1).cpu().numpy()
        return proba

    @torch.no_grad()
    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary spam/ham labels for input texts.

        Uses a specified threshold to convert probability scores into binary
        predictions (0=ham, 1=spam). Default threshold is 0.5.

        Args:
            texts (List[str]): List of text messages to classify
            threshold (float): Probability threshold for binary classification (default: 0.5)

        Returns:
            np.ndarray: Array of binary predictions (0=ham, 1=spam)
        """
        proba = self.predict_proba(texts)
        return (proba >= threshold).astype(int)

    def _load_lstm_artifacts(self, model_dir: Union[str, Path]):
        """
        Load LSTM model artifacts from the specified directory.

        Loads and reconstructs the complete LSTM model from saved artifacts:
        - Model weights from .pth file
        - Architecture configuration from model_info.json
        - Vocabulary mapping from vocab.pkl
        - Preprocessing metadata from lstm_metadata.json

        Args:
            model_dir (Union[str, Path]): Directory containing model artifacts

        Returns:
            tuple: (model, vocab, max_length) where:
                - model: Reconstructed and loaded LSTMModel
                - vocab: Vocabulary dictionary (token -> index)
                - max_length: Maximum sequence length for padding

        Raises:
            FileNotFoundError: If required artifacts are missing
        """
        model_path = Path(model_dir)

        # Load best model weights
        state_path = model_path / "best_model.pth"
        if not state_path.exists():
            state_path = model_path / "final_model.pth"

        # Load model info (hyperparameters)
        with open(model_path / "model_info.json", "r") as f:
            model_info = json.load(f)

        # Load vocab
        with open(model_path / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)

        # Load metadata for max_length
        with open(model_path / "lstm_metadata.json", "r") as f:
            metadata = json.load(f)
        max_length = int(metadata.get("max_length", 128))

        # Build model from saved config
        cfg = model_info.get("model_config", {})
        model = LSTMModel(
            vocab_size=model_info.get("vocab_size", len(vocab)),
            embedding_dim=cfg.get("embedding_dim", 128),
            hidden_dims=cfg.get("hidden_dims", [128, 64]),
            dense_layers=cfg.get("dense_layers", [64, 32]),
            dropout_rate=cfg.get("dropout_rate", 0.3),
            bidirectional=cfg.get("bidirectional", True),
            use_attention=cfg.get("use_attention", False),
            attention_dim=cfg.get("attention_dim", 64),
            max_length=max_length,
        )

        model.load_state_dict(torch.load(state_path, map_location=torch.device("cpu")))
        model.eval()

        return model, vocab, max_length

    def _text_to_indices(self, text: str, vocab: dict, max_length: int) -> List[int]:
        """
        Convert text to token indices for model input.

        Processes a single text through the same preprocessing pipeline used
        during training: cleaning, tokenization, vocabulary mapping, and padding.

        Args:
            text (str): Input text to process
            vocab (dict): Vocabulary mapping (token -> index)
            max_length (int): Maximum sequence length for padding/truncation

        Returns:
            List[int]: List of token indices of length max_length, padded with
                <PAD> tokens or truncated as needed
        """
        cleaned = clean_text_for_lstm(text)
        tokens = tokenize_text(cleaned)
        indices = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
        if len(indices) < max_length:
            indices = indices + [vocab["<PAD>"]] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
        return indices
