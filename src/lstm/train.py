import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
import json
import pickle
from pathlib import Path
import time
from datetime import datetime
import warnings

from src.lstm.evaluation import (
    evaluate_model_with_sources,
    create_lstm_evaluation_plots,
)
from src.lstm.model import LSTMModel

warnings.filterwarnings("ignore")


def load_lstm_data(data_dir="datasets/preprocessed/lstm"):
    """
    Load preprocessed tensors, sources, vocabulary, and metadata.

    Args:
        data_dir (str): Directory containing LSTM tensors and metadata.

    Returns:
        tuple: (X_train, y_train, X_test, y_test, train_sources, test_sources, vocab, metadata)

    Expected files:
        X_train.pt, y_train.pt, X_test.pt, y_test.pt,
        train_sources.pkl, test_sources.pkl, vocab.pkl, lstm_metadata.json
    """
    print("Loading LSTM preprocessed data...")

    data_path = Path(data_dir)

    # Load tensors
    X_train = torch.load(data_path / "X_train.pt")
    y_train = torch.load(data_path / "y_train.pt")
    X_test = torch.load(data_path / "X_test.pt")
    y_test = torch.load(data_path / "y_test.pt")

    # Load source information (like XGBoost preprocessing)
    with open(data_path / "train_sources.pkl", "rb") as f:
        train_sources = pickle.load(f)
    with open(data_path / "test_sources.pkl", "rb") as f:
        test_sources = pickle.load(f)

    # Load vocabulary
    with open(data_path / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Load metadata
    with open(data_path / "lstm_metadata.json", "r") as f:
        metadata = json.load(f)

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training sources: {np.unique(train_sources)}")
    print(f"Test sources: {np.unique(test_sources)}")

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        train_sources,
        test_sources,
        vocab,
        metadata,
    )


def setup_stratified_split(y_train, train_sources, test_size=0.2, random_state=42):
    """
    Create (source+label) stratification keys for a train/val split.

    Args:
        y_train (torch.Tensor): Training labels.
        train_sources (list): Training sources.
        test_size (float): Validation proportion.
        random_state (int): RNG seed.

    Returns:
        np.ndarray: Stratification keys per sample.
    """
    print("Setting up stratified train-validation split based on source+label...")

    from sklearn.model_selection import train_test_split

    # Create stratification key combining source and label (same as XGBoost)
    stratify_labels = np.array(
        [f"{source}_{label}" for source, label in zip(train_sources, y_train)]
    )

    print("Stratification groups found:")
    unique_groups, counts = np.unique(stratify_labels, return_counts=True)
    for group, count in zip(unique_groups, counts):
        print(f"  {group}: {count} samples")

    return stratify_labels


def train_and_evaluate_model(
    X_train,
    y_train,
    X_test,
    y_test,
    train_sources,
    test_sources,
    vocab,
    metadata,
    model_config,
    train_config,
    output_dir="models/lstm",
    device="cuda",
):
    """
    Train LSTM with a stratified validation split, then evaluate on test.

    Args:
        X_train (torch.Tensor): Train sequences.
        y_train (torch.Tensor): Train labels.
        X_test (torch.Tensor): Test sequences.
        y_test (torch.Tensor): Test labels.
        train_sources (array-like): Train sources.
        test_sources (array-like): Test sources.
        vocab (dict): Vocabulary mapping.
        metadata (dict): Preprocessing metadata.
        model_config (dict): Model hyperparameters.
        train_config (dict): Training hyperparameters.
        output_dir (str): Output directory.
        device (str): 'cuda' or 'cpu'.

    Returns:
        tuple: (model, test_metrics, train_losses)
    """
    print("\nTraining model with validation split...")

    # Setup
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Create validation split for model selection
    stratify_labels = setup_stratified_split(y_train, train_sources, random_state=42)

    from sklearn.model_selection import train_test_split

    train_idx, val_idx = train_test_split(
        range(len(X_train)),
        test_size=0.2,
        stratify=stratify_labels,
        random_state=42,
    )

    # Split data
    X_final_train, X_val = X_train[train_idx], X_train[val_idx]
    y_final_train, y_val = y_train[train_idx], y_train[val_idx]

    print(f"Train set: {len(X_final_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Create data loaders
    train_dataset = TensorDataset(X_final_train, y_final_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=train_config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_config["batch_size"], shuffle=False
    )

    # Create model
    model = LSTMModel(
        vocab_size=len(vocab), max_length=metadata["max_length"], **model_config
    ).to(device)

    print(f"Model architecture:")
    print(model)

    # Setup optimizer and criterion
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config.get("weight_decay", 1e-5),
    )
    criterion = nn.BCELoss()

    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_config.get("lr_factor", 0.5),
        patience=train_config.get("lr_patience", 3),
    )

    # Training variables
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    # Training loop
    start_time = time.time()

    for epoch in range(train_config["epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            if train_config.get("clip_grad_norm"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_config["clip_grad_norm"]
                )

            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)

        if (
            epoch % train_config.get("print_every", 10) == 0
            or epoch == train_config["epochs"] - 1
        ):
            print(
                f"Epoch {epoch+1}/{train_config['epochs']}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        # Early stopping and model saving based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model based on validation loss
            torch.save(model.state_dict(), output_path / "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= train_config.get("early_stopping_patience", 15):
            print(f"Early stopping at epoch {epoch+1}")
            break

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Load best model for evaluation
    model.load_state_dict(torch.load(output_path / "best_model.pth"))

    # Evaluate on test set with source-based metrics
    evaluation_results = evaluate_model_with_sources(
        model, X_test, y_test, test_sources, device
    )

    metrics = evaluation_results["test_metrics"]

    # Create evaluation plots
    create_lstm_evaluation_plots(evaluation_results, y_test, output_path)

    # Save final model and results
    torch.save(model.state_dict(), output_path / "final_model.pth")

    # Save model architecture and training info
    model_info = {
        "model_config": model_config,
        "train_config": train_config,
        "vocab_size": len(vocab),
        "max_length": metadata["max_length"],
        "test_metrics": evaluation_results["test_metrics"],
        "test_performance_by_source": evaluation_results["test_performance_by_source"],
        "test_classification_report": evaluation_results["test_classification_report"],
        "training_time": training_time,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    # Save vocabulary and minimal preprocessing metadata for standalone inference
    try:
        with open(output_path / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        # Also save lstm_metadata.json to the model dir for max_length etc.
        with open(output_path / "lstm_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save LSTM artifacts for inference: {e}")

    return model, metrics, train_losses


def run_lstm_experiments(
    model_configs,
    train_config,
    data_dir="preprocessed/lstm",
    output_dir="models/lstm",
    device="cuda",
):
    """
    Train and evaluate multiple LSTM configs, saving artifacts and summary.

    Args:
        model_configs (list[dict]): List of {'name','config'}.
        train_config (dict): Training hyperparameters.
        data_dir (str): Preprocessed data directory.
        output_dir (str): Output directory.
        device (str): 'cuda' or 'cpu'.

    Returns:
        dict: Results per model with test metrics.

    Steps:
        1) Load data, 2) Train per config, 3) Evaluate, 4) Save results, 5) Print summary.
    """

    print("Starting LSTM Spam Detection Training Pipeline")
    print("=" * 60)

    # Load data
    X_train, y_train, X_test, y_test, train_sources, test_sources, vocab, metadata = (
        load_lstm_data(data_dir)
    )

    # Results storage
    results = {}

    # Run experiments
    for model_setup in model_configs:
        model_name = model_setup["name"]
        model_config = model_setup["config"]

        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        print(f"Model config: {model_config}")

        # Train and evaluate model
        model_output_dir = Path(output_dir) / model_name
        final_model, test_metrics, train_losses = train_and_evaluate_model(
            X_train,
            y_train,
            X_test,
            y_test,
            train_sources,
            test_sources,
            vocab,
            metadata,
            model_config,
            train_config,
            output_dir=model_output_dir,
            device=device,
        )

        # Store results
        results[model_name] = {
            "test_metrics": test_metrics,
            "model_config": model_config,
        }

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Test F1: {result['test_metrics']['f1']:.4f}")
        print(f"  Test Accuracy: {result['test_metrics']['accuracy']:.4f}")
        print(f"  Test Precision: {result['test_metrics']['precision']:.4f}")
        print(f"  Test Recall: {result['test_metrics']['recall']:.4f}")
        print(f"  Test ROC-AUC: {result['test_metrics']['roc_auc']:.4f}")

    return results


if __name__ == "__main__":
    # Define model configurations to test
    model_configs = [
        {
            "name": "basic_bilstm",
            "config": {
                "embedding_dim": 128,
                "hidden_dims": [64],
                "dense_layers": [32],
                "dropout_rate": 0.3,
                "bidirectional": True,
                "use_attention": False,
            },
        },
        {
            "name": "deep_bilstm",
            "config": {
                "embedding_dim": 128,
                "hidden_dims": [128, 64],
                "dense_layers": [64, 32],
                "dropout_rate": 0.3,
                "bidirectional": True,
                "use_attention": False,
            },
        },
        {
            "name": "bilstm_attention",
            "config": {
                "embedding_dim": 128,
                "hidden_dims": [128],
                "dense_layers": [64, 32],
                "dropout_rate": 0.3,
                "bidirectional": True,
                "use_attention": True,
                "attention_dim": 64,
            },
        },
        {
            "name": "unidirectional_lstm",
            "config": {
                "embedding_dim": 128,
                "hidden_dims": [128, 64],
                "dense_layers": [64, 32],
                "dropout_rate": 0.3,
                "bidirectional": False,
                "use_attention": False,
            },
        },
    ]

    # Training configuration
    train_config = {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "early_stopping_patience": 10,
        "lr_patience": 3,
        "lr_factor": 0.5,
        "clip_grad_norm": 1.0,
        "print_every": 5,
    }

    # Run experiments
    results = run_lstm_experiments(
        model_configs,
        train_config,
        data_dir="preprocessed/lstm",
        output_dir="models/lstm",
        device="cuda",
    )

    print("\nBiLSTM training pipeline completed!")
