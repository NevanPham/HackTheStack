import pandas as pd
import numpy as np
import re
import json
import pickle
from pathlib import Path
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def load_preprocessed_data(input_dir="datasets/preprocessed"):
    """
    Load shared preprocessed train/test CSVs and optional metadata.

    Args:
        input_dir (str): Base directory with train.csv, test.csv, metadata.json.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict|None]: (train_data, test_data, metadata)
    """
    print("=" * 60)
    print("LOADING PREPROCESSED DATA FOR BILSTM")
    print("=" * 60)

    input_path = Path(input_dir)

    try:
        # Load train and test data
        train_data = pd.read_csv(input_path / "train.csv", encoding="utf-8")
        test_data = pd.read_csv(input_path / "test.csv", encoding="utf-8")

        # Load metadata if available
        try:
            with open(input_path / "metadata.json", "r") as f:
                metadata = json.load(f)
        except:
            metadata = None

        print(f"Loaded training data: {len(train_data):,} records")
        print(f"Loaded test data: {len(test_data):,} records")

        return train_data, test_data, metadata

    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return None, None, None


def clean_text_for_lstm(text):
    """
    Light cleaning tailored for LSTM inputs.

    Args:
        text (str): Raw text.

    Returns:
        str: Cleaned text.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove extra whitespace but preserve sentence structure
    text = re.sub(r"\s+", " ", text)

    # Keep basic punctuation that might be important for spam detection
    # Remove excessive punctuation patterns (already handled in shared processing)
    text = re.sub(r"[^\w\s.,!?<>-]", " ", text)

    # Clean up spacing around special tokens
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def tokenize_text(text):
    """
    Simple tokenizer preserving special tokens like <URL>, <EMAIL>.

    Args:
        text (str): Input text.

    Returns:
        list[str]: Tokens.
    """
    if not text:
        return []

    # Split on whitespace and basic punctuation
    tokens = re.findall(r"\b\w+\b|<\w+>|[.,!?]", text)
    return [token for token in tokens if len(token) > 0]


def build_vocabulary(texts, vocab_size=10000, min_freq=2):
    """
    Build token→index vocabulary with special tokens.

    Args:
        texts (list[str]): Training texts.
        vocab_size (int): Max vocab size.
        min_freq (int): Min token frequency.

    Returns:
        tuple[dict, dict]: (vocab, idx_to_token)
    """
    print("Building vocabulary...")

    # Tokenize all texts
    all_tokens = []
    for text in texts:
        tokens = tokenize_text(text)
        all_tokens.extend(tokens)

    # Count token frequencies
    token_counts = Counter(all_tokens)

    # Create vocabulary with special tokens
    vocab = {
        "<PAD>": 0,  # Padding token
        "<UNK>": 1,  # Unknown token
        "<SOS>": 2,  # Start of sequence (optional)
        "<EOS>": 3,  # End of sequence (optional)
    }

    # Add most frequent tokens
    most_common = token_counts.most_common(vocab_size - len(vocab))
    for token, count in most_common:
        if count >= min_freq:
            vocab[token] = len(vocab)
        if len(vocab) >= vocab_size:
            break

    # Create reverse vocabulary
    idx_to_token = {idx: token for token, idx in vocab.items()}

    print(f"Vocabulary size: {len(vocab):,}")
    print(f"Total tokens processed: {len(all_tokens):,}")
    print(f"Unique tokens: {len(token_counts):,}")

    return vocab, idx_to_token


def text_to_sequence(text, vocab, max_length=None):
    """
    Convert text to token indices with optional truncation.

    Args:
        text (str): Input text.
        vocab (dict): Token→index mapping.
        max_length (int|None): Optional cap.

    Returns:
        list[int]: Token indices.
    """
    tokens = tokenize_text(text)

    # Convert tokens to indices
    sequence = []
    for token in tokens:
        if token in vocab:
            sequence.append(vocab[token])
        else:
            sequence.append(vocab["<UNK>"])

    # Truncate if needed
    if max_length and len(sequence) > max_length:
        sequence = sequence[:max_length]

    return sequence


def create_lstm_dataset(data, vocab, max_length=128):
    """
    Build sequences and labels from DataFrame.

    Args:
        data (pd.DataFrame): Columns ['cleaned_text','label'].
        vocab (dict): Token→index mapping.
        max_length (int): Sequence length cap.

    Returns:
        tuple[list[list[int]], list[int]]: (sequences, labels)
    """
    print(f"Creating LSTM dataset with max_length={max_length}...")

    sequences = []
    labels = []

    for idx, row in data.iterrows():
        # Convert text to sequence
        sequence = text_to_sequence(row["cleaned_text"], vocab, max_length)

        sequences.append(sequence)
        labels.append(row["label"])

    return sequences, labels


def pad_sequences(sequences, max_length=None, padding_value=0):
    """
    Pad/truncate sequences to a uniform length.

    Args:
        sequences (list[list[int]]): Variable-length sequences.
        max_length (int|None): Target length or auto max.
        padding_value (int): PAD token index.

    Returns:
        tuple[np.ndarray, int]: (padded_sequences, used_max_length)
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded_sequences = []
    for sequence in sequences:
        if len(sequence) < max_length:
            # Pad with padding_value
            padded_seq = sequence + [padding_value] * (max_length - len(sequence))
        else:
            # Truncate if too long
            padded_seq = sequence[:max_length]

        padded_sequences.append(padded_seq)

    return np.array(padded_sequences), max_length


def analyze_sequence_lengths(sequences):
    """Print and return a suggested max_length from the 95th percentile."""
    lengths = [len(seq) for seq in sequences]

    print("Sequence length statistics:")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    print(f"  25th percentile: {np.percentile(lengths, 25):.1f}")
    print(f"  75th percentile: {np.percentile(lengths, 75):.1f}")
    print(f"  95th percentile: {np.percentile(lengths, 95):.1f}")
    print(f"  Max: {np.max(lengths)}")

    # Suggest optimal max_length (covers ~95% of sequences)
    suggested_max_length = int(np.percentile(lengths, 95))
    print(f"  Suggested max_length: {suggested_max_length}")

    return suggested_max_length


def create_pytorch_tensors(padded_sequences, labels):
    """Convert arrays to PyTorch tensors (Long for X, Float for y)."""
    X_tensor = torch.LongTensor(padded_sequences)
    y_tensor = torch.FloatTensor(labels)

    print(f"Created PyTorch tensors:")
    print(f"  X shape: {X_tensor.shape}")
    print(f"  y shape: {y_tensor.shape}")

    return X_tensor, y_tensor


def save_lstm_preprocessed_data(
    train_tensors,
    test_tensors,
    train_sources,
    test_sources,
    vocab,
    metadata,
    output_dir="datasets/preprocessed/lstm",
):
    """
    Save tensors, sources, vocab, and metadata.

    Args:
        train_tensors (tuple): (X_train, y_train).
        test_tensors (tuple): (X_test, y_test).
        train_sources (list|np.ndarray): Train sources.
        test_sources (list|np.ndarray): Test sources.
        vocab (dict): Token→index mapping.
        metadata (dict): LSTM preprocessing metadata.
        output_dir (str): Output directory.

    Returns:
        None

    Outputs:
        X_train.pt, y_train.pt, X_test.pt, y_test.pt,
        train_sources.pkl, test_sources.pkl, vocab.pkl, vocab.json, lstm_metadata.json
    """
    print("=" * 60)
    print("SAVING LSTM PREPROCESSED DATA")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    try:
        # Save PyTorch tensors
        X_train, y_train = train_tensors
        X_test, y_test = test_tensors

        torch.save(X_train, output_path / "X_train.pt")
        torch.save(y_train, output_path / "y_train.pt")
        torch.save(X_test, output_path / "X_test.pt")
        torch.save(y_test, output_path / "y_test.pt")

        # Save source information (like XGBoost preprocessing)
        with open(output_path / "train_sources.pkl", "wb") as f:
            pickle.dump(train_sources, f)
        with open(output_path / "test_sources.pkl", "wb") as f:
            pickle.dump(test_sources, f)

        # Save vocabulary
        with open(output_path / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

        # Save vocabulary as JSON for readability
        with open(output_path / "vocab.json", "w") as f:
            json.dump(vocab, f, indent=2)

        # Save preprocessing metadata
        with open(output_path / "lstm_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"LSTM preprocessed data saved to: {output_path}")
        print("Files created:")
        print(f"  - X_train.pt, y_train.pt ({X_train.shape[0]:,} samples)")
        print(f"  - X_test.pt, y_test.pt ({X_test.shape[0]:,} samples)")
        print(f"  - train_sources.pkl, test_sources.pkl")
        print(f"  - vocab.pkl, vocab.json ({len(vocab):,} tokens)")
        print(f"  - lstm_metadata.json")

    except Exception as e:
        print(f"Error saving LSTM data: {e}")


def run_lstm_preprocessing(
    input_dir="datasets/preprocessed",
    output_dir="datasets/preprocessed/lstm",
    vocab_size=10000,
    max_length=None,
    min_freq=2,
):
    """
    Full LSTM preprocessing: load, clean, vocab, sequences, pad, tensors, save.

    Args:
        input_dir (str): Shared preprocessed directory.
        output_dir (str): Output directory for LSTM tensors.
        vocab_size (int): Max vocab size.
        max_length (int|None): Optional cap (auto if None).
        min_freq (int): Min token frequency.

    Returns:
        bool: True on success.
    """
    print("STARTING BILSTM PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load preprocessed data
    train_data, test_data, shared_metadata = load_preprocessed_data(input_dir)
    if train_data is None:
        return False

    # Step 2: Additional text cleaning for LSTM
    print("\nApplying LSTM-specific text cleaning...")
    train_data = train_data.copy()
    test_data = test_data.copy()

    train_data["cleaned_text"] = train_data["cleaned_text"].apply(clean_text_for_lstm)
    test_data["cleaned_text"] = test_data["cleaned_text"].apply(clean_text_for_lstm)

    # Step 3: Build vocabulary from training data only
    print("\n" + "=" * 40)
    print("BUILDING VOCABULARY")
    print("=" * 40)

    vocab, idx_to_token = build_vocabulary(
        train_data["cleaned_text"].tolist(), vocab_size=vocab_size, min_freq=min_freq
    )

    # Step 4: Convert texts to sequences
    print("\n" + "=" * 40)
    print("CONVERTING TEXT TO SEQUENCES")
    print("=" * 40)

    train_sequences, train_labels = create_lstm_dataset(train_data, vocab)
    test_sequences, test_labels = create_lstm_dataset(test_data, vocab)

    # Step 5: Analyze sequence lengths and determine max_length
    print("\nAnalyzing sequence lengths...")
    all_sequences = train_sequences + test_sequences
    suggested_max_length = analyze_sequence_lengths(all_sequences)

    if max_length is None:
        max_length = min(suggested_max_length, 128)  # Cap at 128 for efficiency
        print(f"Using max_length: {max_length}")

    # Step 6: Pad sequences
    print("\n" + "=" * 40)
    print("PADDING SEQUENCES")
    print("=" * 40)

    X_train_padded, actual_max_length = pad_sequences(
        train_sequences, max_length, vocab["<PAD>"]
    )
    X_test_padded, _ = pad_sequences(test_sequences, max_length, vocab["<PAD>"])

    print(f"Padded sequences to length: {actual_max_length}")

    # Step 7: Create PyTorch tensors
    print("\nCreating PyTorch tensors...")
    train_tensors = create_pytorch_tensors(X_train_padded, train_labels)
    test_tensors = create_pytorch_tensors(X_test_padded, test_labels)

    # Step 8: Prepare metadata
    lstm_metadata = {
        "vocab_size": len(vocab),
        "max_length": actual_max_length,
        "num_train_samples": len(train_sequences),
        "num_test_samples": len(test_sequences),
        "padding_token": vocab["<PAD>"],
        "unknown_token": vocab["<UNK>"],
        "min_freq": min_freq,
        "sequence_length_stats": {
            "mean": float(np.mean([len(seq) for seq in all_sequences])),
            "median": float(np.median([len(seq) for seq in all_sequences])),
            "max": int(np.max([len(seq) for seq in all_sequences])),
            "percentile_95": int(
                np.percentile([len(seq) for seq in all_sequences], 95)
            ),
        },
        "class_distribution": {
            "train_spam_rate": float(np.mean(train_labels)),
            "test_spam_rate": float(np.mean(test_labels)),
        },
    }

    # Include shared metadata if available
    if shared_metadata:
        lstm_metadata["shared_preprocessing"] = shared_metadata

    # Step 9: Extract source information
    train_sources = train_data["source"].values
    test_sources = test_data["source"].values

    # Step 10: Save everything
    save_lstm_preprocessed_data(
        train_tensors,
        test_tensors,
        train_sources,
        test_sources,
        vocab,
        lstm_metadata,
        output_dir,
    )

    print("\n" + "=" * 60)
    print("BILSTM PREPROCESSING COMPLETED!")
    print("=" * 60)
    print(f"Vocabulary size: {len(vocab):,}")
    print(f"Max sequence length: {actual_max_length}")
    print(f"Training samples: {len(train_sequences):,}")
    print(f"Test samples: {len(test_sequences):,}")
    print(f"Output directory: {output_dir}")
    print("\nNext steps:")
    print("  1. Use the saved tensors for BiLSTM model training")
    print("  2. Load vocab.pkl for text preprocessing during inference")
    print("  3. Check lstm_metadata.json for preprocessing details")

    return True


def load_lstm_data_for_training(data_dir="datasets/preprocessed/lstm"):
    """
    Load tensors, vocab, and metadata for training.

    Args:
        data_dir (str): LSTM preprocessed directory.

    Returns:
        tuple: ((X_train, y_train), (X_test, y_test), vocab, metadata)
    """
    data_path = Path(data_dir)

    # Load tensors
    X_train = torch.load(data_path / "X_train.pt")
    y_train = torch.load(data_path / "y_train.pt")
    X_test = torch.load(data_path / "X_test.pt")
    y_test = torch.load(data_path / "y_test.pt")

    # Load vocabulary
    with open(data_path / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Load metadata
    with open(data_path / "lstm_metadata.json", "r") as f:
        metadata = json.load(f)

    return (X_train, y_train), (X_test, y_test), vocab, metadata


# Main execution
if __name__ == "__main__":
    # Run LSTM preprocessing pipeline
    success = run_lstm_preprocessing(
        input_dir="datasets/preprocessed",  # Input from shared preprocessing
        output_dir="preprocessed/lstm",  # Output directory for LSTM data
        vocab_size=10000,  # Vocabulary size
        max_length=None,  # Auto-determine from data (will cap at 128)
        min_freq=2,  # Minimum token frequency
    )

    if success:
        print("\nBiLSTM preprocessing pipeline completed successfully!")
        print("You can now use the preprocessed data for BiLSTM model training.")

        # Example of how to load the data for training
        print("\nExample usage for loading data:")
        print("```python")
        print("from lstm_preprocessing import load_lstm_data_for_training")
        print("")
        print("# Load preprocessed data")
        print(
            "(X_train, y_train), (X_test, y_test), vocab, metadata = load_lstm_data_for_training()"
        )
        print("")
        print("# Use in PyTorch DataLoader")
        print("from torch.utils.data import TensorDataset, DataLoader")
        print("train_dataset = TensorDataset(X_train, y_train)")
        print("train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)")
        print("```")
    else:
        print("\nBiLSTM preprocessing failed. Check error messages above.")
