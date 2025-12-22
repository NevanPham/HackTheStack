"""
Spam Detection Preprocessing Utilities

Concise utilities to load, clean, split, rebalance, visualize, and persist
spam detection datasets.

Modules:
- load_and_analyze_data: quick dataset overview
- clean_text_functional: text cleaning with token preservation
- create_stratified_train_test_splits: source+label stratified split
- rebalance_train_data: source-aware training rebalance with soft caps
- setup_kfold_strategy: stratified K-fold utility
- save_preprocessed_data: persist splits + metadata
- create_balance_analysis_plots: sanity-check visualizations
- run_base_preprocessing_pipeline: end-to-end runner
"""

import pandas as pd
import numpy as np
import re
import string
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings

warnings.filterwarnings("ignore")


def load_and_analyze_data(input_path="datasets/combined.csv"):
    """
    Load combined dataset and print a compact distribution summary.

    Args:
        input_path (str): Path to the combined CSV with columns ['text','label','source'].

    Returns:
        pd.DataFrame | None: Loaded DataFrame if successful, else None.

    Notes:
        - Prints dataset shape, columns, per-source counts and spam rates, overall spam rate.
    """
    print("=" * 60)
    print("LOADING AND ANALYZING DATA")
    print("=" * 60)

    try:
        data = pd.read_csv(input_path, encoding="utf-8")
        print(f"Loaded {len(data):,} records")

        # Basic analysis
        print(f"\nDataset shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")

        # Source distribution
        source_dist = data["source"].value_counts()
        source_pct = (source_dist / len(data) * 100).round(2)

        print(f"\nSource Distribution:")
        for source in source_dist.index:
            count = source_dist[source]
            pct = source_pct[source]
            print(f"  {source}: {count:,} ({pct}%)")

        # Spam rate by source
        print(f"\nSpam Rate by Source:")
        for source in data["source"].unique():
            source_data = data[data["source"] == source]
            spam_rate = source_data["label"].mean() * 100
            spam_count = source_data["label"].sum()
            total_count = len(source_data)
            print(f"  {source}: {spam_rate:.1f}% ({spam_count:,}/{total_count:,})")

        # Overall statistics
        overall_spam_rate = data["label"].mean() * 100
        print(f"\nOverall spam rate: {overall_spam_rate:.1f}%")

        return data

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def clean_text_functional(data):
    """
    Clean text while preserving key spam indicators.

    Args:
        data (pd.DataFrame): DataFrame with column 'text'.

    Returns:
        pd.DataFrame: DataFrame with 'cleaned_text', 'text_length', 'word_count'.

    Notes:
        - Replaces URLs/emails/phones/money/numbers with tokens.
        - Preserves emphasis patterns using tokens like <EXCITED>, <QUESTION>, <DOTS>.
        - Removes very short/empty cleaned entries.
    """
    print("\n" + "=" * 60)
    print("TEXT CLEANING")
    print("=" * 60)

    def clean_single_text(text):
        """
        Clean individual text entry with spam indicator preservation.

        Args:
            text (str): Raw text to clean

        Returns:
            str: Cleaned text with spam indicators preserved

        Processing Steps:
            1. Handle encoding issues
            2. Process markdown links and formatting
            3. Replace URLs, emails, phones with tokens
            4. Preserve emotional indicators (!, ?, repeated chars)
            5. Clean whitespace while maintaining structure
        """
        if pd.isna(text):
            return ""

        text = str(text)

        # Fix encoding issues
        text = text.encode("utf-8", errors="ignore").decode("utf-8")

        # Normalize patterns but preserve spam indicators

        # Markdown links [text](url) - extract the text part and replace URL
        text = re.sub(
            r"\[([^\]]+)\]\(https?://[^\)]+\)",
            r"\1 <URL> ",
            text,
        )
        text = re.sub(
            r"\[([^\]]+)\]\(www\.[^\)]+\)",
            r"\1 <URL> ",
            text,
        )

        # Other markdown formatting
        text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)  # Bold **text**
        text = re.sub(r"\*([^\*]+)\*", r"\1", text)  # Italic *text*
        text = re.sub(r"__([^_]+)__", r"\1", text)  # Bold __text__
        text = re.sub(r"_([^_]+)_", r"\1", text)  # Italic _text_
        text = re.sub(r"`([^`]+)`", r"\1", text)  # Code `text`
        text = re.sub(r"~~([^~]+)~~", r"\1", text)  # Strikethrough ~~text~~

        # Escaped brackets \[text\] - remove the backslash
        text = re.sub(r"\\\[([^\]]+)\\\]", r"[\1]", text)  # \[text\] -> [text]

        # URLs (after markdown processing)
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            " <URL> ",
            text,
        )
        text = re.sub(
            r"www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            " <URL> ",
            text,
        )

        # Email addresses
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " <EMAIL> ", text
        )

        # Phone numbers
        text = re.sub(
            r"(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
            " <PHONE> ",
            text,
        )

        # Currency and numbers
        text = re.sub(r"[$£€¥₹₽]\s*\d+(?:[.,]\d+)*", " <MONEY> ", text)
        text = re.sub(r"\b\d{4,}\b", " <NUMBER> ", text)

        # Spam indicators (preserve these patterns!)
        text = re.sub(r"!{2,}", " <EXCITED> ", text)
        text = re.sub(r"\?{2,}", " <QUESTION> ", text)
        text = re.sub(r"\.{3,}", " <DOTS> ", text)

        # Clean whitespace and special characters
        text = re.sub(r"[^\w\s!?.,<>-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    print("Cleaning text data...")
    data["cleaned_text"] = data["text"].apply(clean_single_text)

    # Remove invalid entries
    initial_count = len(data)
    data = data[data["cleaned_text"].str.len() >= 3]  # Remove very short texts
    data = data[data["cleaned_text"].str.len() > 0]  # Remove empty texts
    final_count = len(data)

    if initial_count != final_count:
        print(f"Removed {initial_count - final_count:,} invalid records")

    # Add text statistics
    data["text_length"] = data["cleaned_text"].str.len()
    data["word_count"] = data["cleaned_text"].str.split().str.len()

    print(f"Text cleaning completed. Final dataset: {final_count:,} records")
    return data


def rebalance_train_data(train_data, target_spam_rates=None, soft_cap_ratio=0.42):
    """
    Rebalance training data per source with optional target spam rates and soft caps.

    Args:
        train_data (pd.DataFrame): Columns ['text','label','source'].
        target_spam_rates (dict|None): e.g., {'sms':0.375,'email':0.375,'mix_email_sms':None,'youtube':None}.
        soft_cap_ratio (float): Max proportion per source after rebalancing (0-1).

    Returns:
        pd.DataFrame: Rebalanced training DataFrame.

    Notes:
        - For sources with a numeric target, undersamples ham to reach target.
        - For sources with None, preserves original ratio.
        - Applies per-source soft cap with stratified downsampling.
    """
    if target_spam_rates is None:
        target_spam_rates = {
            "sms": 0.375,  # 37.5% spam (middle of 35-40%)
            "email": 0.375,  # 37.5% spam (middle of 35-40%)
            "mix_email_sms": None,  # Keep original ratio
            "youtube": None,  # Keep original ratio
        }

    print("\n" + "=" * 60)
    print("REBALANCING TRAINING DATA")
    print("=" * 60)

    # Step 1: Analyze original distribution
    print("Original training distribution:")
    source_stats = []
    for source in train_data["source"].unique():
        source_data = train_data[train_data["source"] == source]
        spam_count = source_data["label"].sum()
        total_count = len(source_data)
        spam_rate = spam_count / total_count if total_count > 0 else 0

        source_stats.append(
            {
                "source": source,
                "total": total_count,
                "spam": spam_count,
                "ham": total_count - spam_count,
                "spam_rate": spam_rate,
            }
        )
        print(f"  {source}: {total_count:,} total, {spam_rate*100:.1f}% spam")

    # Step 2: Rebalance each source
    print("\nRebalancing sources...")
    print(f"Target spam rates: {target_spam_rates}")
    rebalanced_data_list = []

    for source_info in source_stats:
        source = source_info["source"]
        source_data = train_data[train_data["source"] == source].copy()

        spam_data = source_data[source_data["label"] == 1]
        ham_data = source_data[source_data["label"] == 0]

        target_spam_rate = (
            target_spam_rates.get(source)
            if isinstance(target_spam_rates, dict)
            else None
        )
        print(
            f"  Processing {source}: target_spam_rate = {target_spam_rate} (type: {type(target_spam_rate)})"
        )

        if target_spam_rate is None:
            # Keep original ratio for mix_email_sms and youtube
            rebalanced_source = source_data
            print(
                f"  {source}: Keeping original ratio ({source_info['spam_rate']*100:.1f}% spam)"
            )
        else:
            # Rebalance sms and email to target spam rate
            spam_count = len(spam_data)

            if spam_count == 0:
                print(f"  {source}: No spam data, skipping")
                rebalanced_source = source_data
            elif (
                not isinstance(target_spam_rate, (int, float))
                or target_spam_rate <= 0
                or target_spam_rate >= 1
            ):
                print(
                    f"  {source}: Invalid target_spam_rate {target_spam_rate}, keeping original"
                )
                rebalanced_source = source_data
            else:
                # Calculate required ham count to achieve target spam rate
                # target_spam_rate = spam_count / (spam_count + ham_count)
                # ham_count = spam_count * (1 - target_spam_rate) / target_spam_rate
                required_ham_count = int(
                    spam_count * (1 - target_spam_rate) / target_spam_rate
                )

                if required_ham_count <= len(ham_data):
                    # Undersample ham to achieve target
                    ham_balanced = ham_data.sample(
                        n=required_ham_count, random_state=42
                    )
                    rebalanced_source = pd.concat(
                        [spam_data, ham_balanced], ignore_index=True
                    )
                    new_spam_rate = len(spam_data) / len(rebalanced_source)
                    print(
                        f"  {source}: {len(source_data):,} → {len(rebalanced_source):,} ({new_spam_rate*100:.1f}% spam)"
                    )
                else:
                    # Not enough ham data, keep all
                    rebalanced_source = source_data
                    print(
                        f"  {source}: Insufficient ham data, keeping original ({source_info['spam_rate']*100:.1f}% spam)"
                    )

        rebalanced_data_list.append(rebalanced_source)

    # Combine rebalanced sources
    rebalanced_data = pd.concat(rebalanced_data_list, ignore_index=True)

    # Step 3: Apply soft cap
    print(f"\nApplying soft cap ({soft_cap_ratio*100:.0f}% max per source)...")
    total_samples = len(rebalanced_data)
    max_samples_per_source = int(total_samples * soft_cap_ratio)

    final_data_list = []
    for source in rebalanced_data["source"].unique():
        source_data = rebalanced_data[rebalanced_data["source"] == source]

        if len(source_data) > max_samples_per_source:
            # Stratified downsample to maintain spam/ham ratio
            source_capped = source_data.groupby("label", group_keys=False).apply(
                lambda x: x.sample(
                    n=int(max_samples_per_source * len(x) / len(source_data)),
                    random_state=42,
                )
            )
            print(f"  {source}: {len(source_data):,} → {len(source_capped):,} (capped)")
        else:
            source_capped = source_data
            print(f"  {source}: {len(source_data):,} (no cap needed)")

        final_data_list.append(source_capped)

    final_data = pd.concat(final_data_list, ignore_index=True)

    print(f"\nTraining data rebalancing completed:")
    print(f"  Original: {len(train_data):,} records")
    print(f"  Rebalanced: {len(final_data):,} records")
    print(f"  Overall spam rate: {final_data['label'].mean() * 100:.1f}%")

    # Final distribution
    print(f"\nFinal training distribution:")
    for source in final_data["source"].unique():
        source_data = final_data[final_data["source"] == source]
        spam_rate = source_data["label"].mean() * 100
        proportion = len(source_data) / len(final_data) * 100
        print(
            f"  {source}: {len(source_data):,} ({proportion:.1f}% of total, {spam_rate:.1f}% spam)"
        )

    return final_data


def create_stratified_train_test_splits(data, test_size=0.2, random_state=42):
    """
    Create train/test splits stratified by (source,label).

    Args:
        data (pd.DataFrame): Columns ['text','label','source'].
        test_size (float): Proportion for test set.
        random_state (int): RNG seed.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data).

    Notes:
        - Verifies overall spam rates and per-source distributions post split.
    """
    print("\n" + "=" * 60)
    print("CREATING STRATIFIED TRAIN/TEST SPLITS")
    print("=" * 60)

    # Create stratification key combining source and label
    # Use a separator that's unlikely to appear in source names
    data["strat_key"] = data["source"].astype(str) + "|||" + data["label"].astype(str)

    print("Original distribution by source and label:")
    strat_dist = data["strat_key"].value_counts().sort_index()
    for key in strat_dist.index:
        source, label = key.split("|||")
        label_name = "spam" if label == "1" else "ham"
        print(f"  {source}_{label_name}: {strat_dist[key]:,} records")

    # Split data with stratification
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data["strat_key"]
    )

    # Remove the temporary stratification key
    train_data = train_data.drop("strat_key", axis=1)
    test_data = test_data.drop("strat_key", axis=1)

    print(f"\nData splits created:")
    print(
        f"  Training: {len(train_data):,} records ({len(train_data)/len(data)*100:.1f}%)"
    )
    print(
        f"  Test:     {len(test_data):,} records ({len(test_data)/len(data)*100:.1f}%)"
    )

    # Verify stratification - overall spam rates
    print(f"\nOverall spam rate verification:")
    original_spam_rate = data["label"].mean() * 100
    train_spam_rate = train_data["label"].mean() * 100
    test_spam_rate = test_data["label"].mean() * 100
    print(f"  Original: {original_spam_rate:.1f}% spam")
    print(f"  Training: {train_spam_rate:.1f}% spam")
    print(f"  Test:     {test_spam_rate:.1f}% spam")

    # Verify stratification - source distribution
    print(f"\nSource distribution verification:")
    print("  Original:")
    original_source_dist = data["source"].value_counts(normalize=True) * 100
    for src, pct in original_source_dist.items():
        print(f"    {src}: {pct:.1f}%")

    print("  Training:")
    train_source_dist = train_data["source"].value_counts(normalize=True) * 100
    for src, pct in train_source_dist.items():
        print(f"    {src}: {pct:.1f}%")

    print("  Test:")
    test_source_dist = test_data["source"].value_counts(normalize=True) * 100
    for src, pct in test_source_dist.items():
        print(f"    {src}: {pct:.1f}%")

    # Verify spam rates within each source
    print(f"\nSpam rate by source verification:")
    for source in data["source"].unique():
        orig_rate = data[data["source"] == source]["label"].mean() * 100
        train_rate = train_data[train_data["source"] == source]["label"].mean() * 100
        test_rate = test_data[test_data["source"] == source]["label"].mean() * 100
        print(
            f"  {source}: Original={orig_rate:.1f}%, Train={train_rate:.1f}%, Test={test_rate:.1f}%"
        )

    print(f"\nStratified splits completed successfully!")
    return train_data, test_data


def setup_kfold_strategy(train_data, n_splits=5, random_state=42):
    """
    Build a StratifiedKFold using (source,label) as the stratification key.

    Args:
        train_data (pd.DataFrame): Rebalanced training data.
        n_splits (int): Number of folds.
        random_state (int): RNG seed.

    Returns:
        tuple: (skf, stratify_labels, fold_info)
            skf: StratifiedKFold instance
            stratify_labels: pd.Series of strat keys
            fold_info: list[dict] with train/val sizes and spam rates per fold
    """
    print("\n" + "=" * 60)
    print(f"SETTING UP {n_splits}-FOLD CROSS-VALIDATION")
    print("=" * 60)

    # Use source+label for stratification
    stratify_labels = (
        train_data["source"].astype(str) + "|||" + train_data["label"].astype(str)
    )

    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Verify fold balance
    print("Verifying fold balance:")
    fold_info = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(train_data, stratify_labels)
    ):
        fold_train = train_data.iloc[train_idx]
        fold_val = train_data.iloc[val_idx]

        train_spam_rate = fold_train["label"].mean() * 100
        val_spam_rate = fold_val["label"].mean() * 100

        fold_info.append(
            {
                "fold": fold_idx + 1,
                "train_size": len(fold_train),
                "val_size": len(fold_val),
                "train_spam_rate": train_spam_rate,
                "val_spam_rate": val_spam_rate,
            }
        )

        print(
            f"  Fold {fold_idx + 1}: Train={len(fold_train):,} ({train_spam_rate:.1f}% spam), "
            f"Val={len(fold_val):,} ({val_spam_rate:.1f}% spam)"
        )

    print(
        f"\nK-fold setup completed. Use StratifiedKFold with stratify_labels in your model training."
    )

    return skf, stratify_labels, fold_info


def save_preprocessed_data(
    train_data, test_data, kfold_info, output_dir="datasets/preprocessed"
):
    """
    Save train/test splits and metadata.

    Args:
        train_data (pd.DataFrame): Training data with 'cleaned_text','label','source'.
        test_data (pd.DataFrame): Test data with 'cleaned_text','label','source'.
        kfold_info (list[dict]): Fold stats for reference.
        output_dir (str): Target directory.

    Returns:
        None

    Files:
        - train.csv, test.csv
        - metadata.json (counts, distributions, text length stats, folds)
    """
    print("\n" + "=" * 60)
    print("SAVING PREPROCESSED DATA")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Columns to save
    columns_to_save = ["cleaned_text", "label", "source"]

    try:
        # Save main splits
        train_data[columns_to_save].to_csv(
            output_path / "train.csv", index=False, encoding="utf-8"
        )
        test_data[columns_to_save].to_csv(
            output_path / "test.csv", index=False, encoding="utf-8"
        )

        # Calculate combined statistics for metadata
        full_data = pd.concat([train_data, test_data], ignore_index=True)

        # Save metadata
        metadata = {
            "total_records": len(full_data),
            "train_records": len(train_data),
            "test_records": len(test_data),
            "overall_spam_rate": full_data["label"].mean(),
            "source_distribution": full_data["source"].value_counts().to_dict(),
            "spam_rate_by_source": full_data.groupby("source")["label"]
            .mean()
            .to_dict(),
            "text_length_stats": full_data["text_length"].describe().to_dict(),
            "kfold_info": kfold_info,
            "columns": columns_to_save,
        }

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Data saved to: {output_path}")
        print(f"Files created:")
        print(f"  - train.csv ({len(train_data):,} records)")
        print(f"  - test.csv ({len(test_data):,} records)")
        print(f"  - metadata.json (preprocessing statistics)")

    except Exception as e:
        print(f"Error saving data: {e}")


def create_balance_analysis_plots(
    data, train_data, test_data, output_dir="datasets/preprocessed"
):
    """
    Generate sanity-check plots comparing original vs balanced distributions.

    Args:
        data (pd.DataFrame): Original dataset.
        train_data (pd.DataFrame): Rebalanced training data.
        test_data (pd.DataFrame): Held-out test data.
        output_dir (str): Target directory.

    Returns:
        None
    """
    print("\n" + "=" * 60)
    print("CREATING BALANCE ANALYSIS PLOTS")
    print("=" * 60)

    output_path = Path(output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "Spam Detection Dataset - Balance Analysis After Preprocessing", fontsize=16
    )

    # 1. Source distribution comparison
    original_source_dist = data["source"].value_counts()
    balanced_source_dist = pd.concat([train_data, test_data])["source"].value_counts()

    ax1 = axes[0, 0]
    x = np.arange(len(original_source_dist))
    width = 0.35
    ax1.bar(
        x - width / 2, original_source_dist.values, width, label="Original", alpha=0.8
    )
    ax1.bar(
        x + width / 2, balanced_source_dist.values, width, label="Balanced", alpha=0.8
    )
    ax1.set_title("Source Distribution: Before vs After")
    ax1.set_xlabel("Source")
    ax1.set_ylabel("Count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(original_source_dist.index)
    ax1.legend()

    # 2. Spam rate by source comparison
    original_spam_rates = data.groupby("source")["label"].mean() * 100
    balanced_spam_rates = (
        pd.concat([train_data, test_data]).groupby("source")["label"].mean() * 100
    )

    ax2 = axes[0, 1]
    x = np.arange(len(original_spam_rates))
    ax2.bar(
        x - width / 2, original_spam_rates.values, width, label="Original", alpha=0.8
    )
    ax2.bar(
        x + width / 2, balanced_spam_rates.values, width, label="Balanced", alpha=0.8
    )
    ax2.set_title("Spam Rate by Source: Before vs After")
    ax2.set_xlabel("Source")
    ax2.set_ylabel("Spam Rate (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(original_spam_rates.index)
    ax2.legend()
    ax2.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50% line")

    # 3. Word count distribution
    full_balanced = pd.concat([train_data, test_data])
    ax3 = axes[0, 2]

    # Use log scale for better visibility of the distribution
    word_counts = full_balanced["word_count"]
    # Add 1 to avoid log(0) issues
    log_word_counts = np.log10(word_counts + 1)

    ax3.boxplot(
        log_word_counts,
        patch_artist=True,
        boxprops=dict(facecolor="skyblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", markerfacecolor="red", markersize=3, alpha=0.6),
    )
    ax3.set_title("Word Count Distribution - Log10 Scale")
    ax3.set_ylabel("Word Count")
    ax3.set_xticklabels([""])  # Remove x-axis label since box plot doesn't need it

    # Convert log scale back to original scale for y-axis labels
    log_ticks = ax3.get_yticks()
    original_ticks = 10**log_ticks - 1
    ax3.set_yticklabels([f"{int(val)}" for val in original_ticks])

    # Add text annotations for key statistics
    median_val = np.median(word_counts)
    q75_val = np.percentile(word_counts, 75)
    q25_val = np.percentile(word_counts, 25)

    ax3.text(
        0.02,
        0.98,
        f"Median: {median_val:.0f}\nQ1: {q25_val:.0f}\nQ3: {q75_val:.0f}",
        transform=ax3.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # 4. Train/Test split visualization
    ax4 = axes[1, 0]
    split_sizes = [len(train_data), len(test_data)]
    split_labels = ["Train", "Test"]
    ax4.pie(
        split_sizes,
        labels=split_labels,
        autopct="%1.1f%%",
        colors=["lightblue", "lightcoral"],
    )
    ax4.set_title("Train/Test Split Distribution")

    # 5. Spam vs Ham distribution
    ax5 = axes[1, 1]
    spam_counts = full_balanced["label"].value_counts()
    ax5.bar(["Ham (0)", "Spam (1)"], spam_counts.values, color=["lightgreen", "salmon"])
    ax5.set_title("Overall Spam vs Ham Distribution")
    ax5.set_ylabel("Count")
    for i, v in enumerate(spam_counts.values):
        ax5.text(i, v + len(full_balanced) * 0.01, f"{v:,}", ha="center", va="bottom")

    # 6. Source-Label heatmap
    ax6 = axes[1, 2]
    source_label_crosstab = (
        pd.crosstab(full_balanced["source"], full_balanced["label"], normalize="index")
        * 100
    )
    sns.heatmap(source_label_crosstab, annot=True, fmt=".1f", cmap="RdYlBu_r", ax=ax6)
    ax6.set_title("Spam Rate by Source (% within source)")
    ax6.set_xlabel("Label")
    ax6.set_ylabel("Source")

    plt.tight_layout()
    plt.savefig(output_path / "balance_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Balance analysis plots saved: {output_path / 'balance_analysis.png'}")


def run_base_preprocessing_pipeline(
    input_path="datasets/combined.csv",
    target_spam_rates=None,
    soft_cap_ratio=0.42,
    test_size=0.2,
    n_folds=5,
    output_dir="datasets/preprocessed",
):
    """
    End-to-end preprocessing: load → clean → split → rebalance → CV setup → save → plots.

    Args:
        input_path (str): Path to combined CSV.
        target_spam_rates (dict|None): Per-source targets (None=keep original).
        soft_cap_ratio (float): Per-source cap after rebalancing.
        test_size (float): Test split proportion.
        n_folds (int): K-folds for CV.
        output_dir (str): Output directory.

    Returns:
        bool: True on success, False on failure.

    Outputs:
        - datasets/preprocessed/train.csv, test.csv, metadata.json, balance_analysis.png
    """
    if target_spam_rates is None:
        target_spam_rates = {
            "sms": 0.375,  # 37.5% spam (middle of 35-40%)
            "email": 0.375,  # 37.5% spam (middle of 35-40%)
            "mix_email_sms": None,  # Keep original ratio
            "youtube": None,  # Keep original ratio
        }

    print("STARTING BASE PREPROCESSING PIPELINE")
    print("=" * 60)
    print("Strategy:")
    print("  1. Stratified split by source and label (80/20)")
    print("  2. Rebalance training data only:")
    print("     - sms/email: undersample ham to achieve 35-40% spam")
    print("     - mix_email_sms/youtube: keep original spam/ham ratio")
    print("  3. Apply soft cap: each source ≤40-45% of total training samples")
    print("=" * 60)

    # Step 1: Load and analyze
    data = load_and_analyze_data(input_path)
    if data is None:
        return False

    # Step 2: Clean text
    data = clean_text_functional(data)

    # Step 3: Create stratified train/test splits (BEFORE any rebalancing)
    train_data, test_data = create_stratified_train_test_splits(data, test_size)

    # Step 4: Rebalance training data only
    balanced_train_data = rebalance_train_data(
        train_data, target_spam_rates, soft_cap_ratio
    )

    # Step 5: Setup k-fold strategy
    skf, stratify_labels, fold_info = setup_kfold_strategy(balanced_train_data, n_folds)

    # Step 6: Save everything
    save_preprocessed_data(balanced_train_data, test_data, fold_info, output_dir)

    # Step 7: Create visualizations
    create_balance_analysis_plots(data, balanced_train_data, test_data, output_dir)

    print("\n" + "=" * 60)
    print("BASE PREPROCESSING PIPELINE COMPLETED!")
    print("=" * 60)
    print(f"Original dataset: {len(data):,} records")
    print(
        f"Training set: {len(balanced_train_data):,} records (rebalanced, ready for {n_folds}-fold CV)"
    )
    print(f"Test set: {len(test_data):,} records (original distribution preserved)")
    print(f"Output directory: {output_dir}")

    # Summary statistics
    print(f"\nFinal training data composition:")
    for source in balanced_train_data["source"].unique():
        source_data = balanced_train_data[balanced_train_data["source"] == source]
        spam_rate = source_data["label"].mean() * 100
        proportion = len(source_data) / len(balanced_train_data) * 100
        print(
            f"  {source}: {len(source_data):,} samples ({proportion:.1f}% of total, {spam_rate:.1f}% spam)"
        )

    overall_train_spam = balanced_train_data["label"].mean() * 100
    overall_test_spam = test_data["label"].mean() * 100
    print(f"\nOverall spam rates:")
    print(f"  Training: {overall_train_spam:.1f}%")
    print(f"  Test: {overall_test_spam:.1f}%")

    return True
