"""
Dataset Combination Utilities for Spam Detection

Loads multiple sources, normalizes labels, combines, validates, summarizes, and
writes a unified CSV with source tracking.
"""

import pandas as pd
import json
import os
from pathlib import Path


def normalize_label(label):
    """
    Normalize labels to binary 0 (ham) / 1 (spam).

    Args:
        label: Any primitive convertible to string/float.

    Returns:
        int: 0 or 1.

    Notes:
        - Recognizes common strings ("spam","ham","true","false", etc.).
        - Numeric > 0.5 → 1; else 0; unknown → 0 with warning.
    """
    # Convert to string first to handle various types
    label_str = str(label).lower().strip()

    # Map spam indicators to 1
    if label_str in ["spam", "1", "1.0", "true", "yes", 1]:
        return 1
    # Map ham indicators to 0
    elif label_str in ["not_spam", "ham", "0", "0.0", "false", "no", 0]:
        return 0
    else:
        # If numeric, try to convert
        try:
            numeric_val = float(label)
            return 1 if numeric_val > 0.5 else 0
        except:
            print(f"Warning: Unknown label '{label}', defaulting to 0")
            return 0


def load_sms_data(file_path):
    """
    Load SMS CSV with columns 'v1' (label) and 'v2' (text).

    Args:
        file_path (str): Path to CSV.

    Returns:
        pd.DataFrame: Columns ['text','label','source'='sms'].

    Notes:
        - Uses latin-1 encoding.
    """
    try:
        # Load with latin-1 encoding to handle special characters
        sms_df = pd.read_csv(file_path, encoding="latin-1")

        # Keep only v1 (label) and v2 (text) columns
        if "v1" in sms_df.columns and "v2" in sms_df.columns:
            sms_df = sms_df[["v2", "v1"]].copy()  # v2 is text, v1 is label
            sms_df.columns = ["text", "label"]
        else:
            print(f"Warning: Expected columns 'v1' and 'v2' not found in {file_path}")
            return pd.DataFrame(columns=["text", "label"])

        # Normalize SMS labels (ham/spam -> 0/1)
        sms_df["label"] = sms_df["label"].apply(normalize_label)

        # Add source column to track data origin
        sms_df["source"] = "sms"

        print(f"Loaded {len(sms_df)} SMS records")
        return sms_df

    except Exception as e:
        print(f"Error loading SMS data: {e}")
        return pd.DataFrame(columns=["text", "label"])


def load_email_data(file_path):
    """
    Load email CSV with 'text' and 'spam' or 'label'.

    Args:
        file_path (str): Path to CSV.

    Returns:
        pd.DataFrame: Columns ['text','label','source'='email'].
    """
    try:
        email_df = pd.read_csv(file_path, encoding="utf-8")

        # Check if columns are text and spam
        if "text" in email_df.columns and "spam" in email_df.columns:
            email_df = email_df[["text", "spam"]].copy()
            email_df.columns = ["text", "label"]  # Rename spam to label
        elif "text" in email_df.columns and "label" in email_df.columns:
            email_df = email_df[["text", "label"]].copy()
        else:
            print(
                f"Warning: Expected columns 'text' and 'spam'/'label' not found in {file_path}"
            )
            return pd.DataFrame(columns=["text", "label"])

        # Normalize email labels
        email_df["label"] = email_df["label"].apply(normalize_label)

        # Add source column
        email_df["source"] = "email"

        print(f"Loaded {len(email_df)} email records")
        return email_df

    except Exception as e:
        print(f"Error loading email data: {e}")
        return pd.DataFrame(columns=["text", "label"])


def load_mix_data(file_path):
    """
    Load mixed email/SMS CSV with 'text' and 'label'.

    Args:
        file_path (str): Path to CSV.

    Returns:
        pd.DataFrame: Columns ['text','label','source'='mix_email_sms'].
    """
    try:
        mix_df = pd.read_csv(file_path, encoding="utf-8")

        # Check if columns are text and label
        if "text" in mix_df.columns and "label" in mix_df.columns:
            mix_df = mix_df[["text", "label"]].copy()
        else:
            print(
                f"Warning: Expected columns 'text' and 'label' not found in {file_path}"
            )
            return pd.DataFrame(columns=["text", "label"])

        # Normalize mix labels
        mix_df["label"] = mix_df["label"].apply(normalize_label)

        # Add source column
        mix_df["source"] = "mix_email_sms"

        print(f"Loaded {len(mix_df)} mixed email/SMS records")
        return mix_df

    except Exception as e:
        print(f"Error loading mix data: {e}")
        return pd.DataFrame(columns=["text", "label"])


def load_youtube_data(file_path):
    """
    Load YouTube comments CSV with 'CONTENT' and 'CLASS'.

    Args:
        file_path (str): Path to CSV.

    Returns:
        pd.DataFrame: Columns ['text','label','source'='youtube'].
    """
    try:
        youtube_df = pd.read_csv(file_path, encoding="utf-8")

        # Keep only CONTENT (text) and CLASS (label) columns
        if "CONTENT" in youtube_df.columns and "CLASS" in youtube_df.columns:
            youtube_df = youtube_df[["CONTENT", "CLASS"]].copy()
            youtube_df.columns = ["text", "label"]
        else:
            print(
                f"Warning: Expected columns 'CONTENT' and 'CLASS' not found in {file_path}"
            )
            return pd.DataFrame(columns=["text", "label"])

        # Normalize YouTube labels
        youtube_df["label"] = youtube_df["label"].apply(normalize_label)

        # Add source column
        youtube_df["source"] = "youtube"

        print(f"Loaded {len(youtube_df)} YouTube comment records")
        return youtube_df

    except Exception as e:
        print(f"Error loading YouTube data: {e}")
        return pd.DataFrame(columns=["text", "label"])


def load_review_data(file_path):
    """
    Load review dataset from JSONL with fields 'text' and 'label'.

    Args:
        file_path (str): Path to JSONL.

    Returns:
        pd.DataFrame: Columns ['text','label','source'='review'].
    """
    try:
        reviews = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    if "text" in review and "label" in review:
                        reviews.append(
                            {
                                "text": review["text"],
                                "label": normalize_label(
                                    review["label"]
                                ),  # Normalize during loading
                            }
                        )
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {line[:50]}... - {e}")
                    continue

        review_df = pd.DataFrame(reviews)

        # Add source column
        if not review_df.empty:
            review_df["source"] = "review"

        print(f"Loaded {len(review_df)} review records")
        return review_df

    except Exception as e:
        print(f"Error loading review data: {e}")
        return pd.DataFrame(columns=["text", "label"])


def clean_and_validate_data(df):
    """
    Remove rows with missing/empty text or label; coerce text to string.

    Args:
        df (pd.DataFrame): Combined data.

    Returns:
        pd.DataFrame: Cleaned data.
    """

    # Remove rows with missing text or label
    initial_count = len(df)
    df = df.dropna(subset=["text", "label"])

    # Remove empty text entries
    df = df[df["text"].str.strip() != ""]

    # Convert text to string type (in case of mixed types)
    df["text"] = df["text"].astype(str)

    # Print cleaning summary
    final_count = len(df)
    removed_count = initial_count - final_count
    if removed_count > 0:
        print(f"Removed {removed_count} invalid records during cleaning")

    return df


def combine_datasets(datasets_dir="datasets"):
    """Load available sources, normalize, combine, validate, summarize, and save CSV.

    Args:
        datasets_dir (str|Path): Base datasets directory.

    Returns:
        None

    Output:
        - datasets/combined.csv (columns: text,label,source)
    """
    datasets_dir = Path(datasets_dir)

    file_paths = {
        "sms": datasets_dir / "sms" / "sms.csv",
        "email": datasets_dir / "email" / "email.csv",
        "mix": datasets_dir / "mix_email_sms" / "mix.csv",
        "youtube": datasets_dir / "comment" / "youtube_comments.csv",
        "review": datasets_dir / "review" / "review.jsonl",
    }

    # Check if datasets directory exists
    if not datasets_dir.exists():
        print(f"Error: {datasets_dir} directory not found!")
        return

    print("Starting data combination process...")
    print("=" * 50)

    # Load each dataset
    dataframes = []

    # Load SMS data
    if os.path.exists(file_paths["sms"]):
        sms_df = load_sms_data(file_paths["sms"])
        if not sms_df.empty:
            dataframes.append(sms_df)
    else:
        print(f"Warning: {file_paths['sms']} not found")

    # Load email data
    if os.path.exists(file_paths["email"]):
        email_df = load_email_data(file_paths["email"])
        if not email_df.empty:
            dataframes.append(email_df)
    else:
        print(f"Warning: {file_paths['email']} not found")

    # Load mix data
    if os.path.exists(file_paths["mix"]):
        mix_df = load_mix_data(file_paths["mix"])
        if not mix_df.empty:
            dataframes.append(mix_df)
    else:
        print(f"Warning: {file_paths['mix']} not found")

    # Load YouTube data
    if os.path.exists(file_paths["youtube"]):
        youtube_df = load_youtube_data(file_paths["youtube"])
        if not youtube_df.empty:
            dataframes.append(youtube_df)
    else:
        print(f"Warning: {file_paths['youtube']} not found")

    # Load review data
    if os.path.exists(file_paths["review"]):
        review_df = load_review_data(file_paths["review"])
        if not review_df.empty:
            dataframes.append(review_df)
    else:
        print(f"Warning: {file_paths['review']} not found")

    if not dataframes:
        print("Error: No valid datasets found!")
        return

    # Combine all dataframes
    print("\n" + "=" * 50)
    print("Combining datasets...")

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Clean and validate the combined data
    combined_df = clean_and_validate_data(combined_df)

    # Display summary statistics
    print("\n" + "=" * 50)
    print("Dataset Summary:")
    print(f"Total records: {len(combined_df)}")

    # Show distribution by source
    source_counts = combined_df["source"].value_counts()
    print("\nRecords by source:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")

    # Show label distribution (should now be 0/1)
    if "label" in combined_df.columns:
        label_counts = combined_df["label"].value_counts().sort_index()
        print("\nLabel distribution (0=ham, 1=spam):")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

        # Show spam rate by source
        print("\nSpam rate by source:")
        for source in combined_df["source"].unique():
            source_data = combined_df[combined_df["source"] == source]
            spam_rate = source_data["label"].mean() * 100
            spam_count = source_data["label"].sum()
            total_count = len(source_data)
            print(f"  {source}: {spam_rate:.1f}% ({spam_count}/{total_count})")

        # Overall spam rate
        overall_spam_rate = combined_df["label"].mean() * 100
        print(f"\nOverall spam rate: {overall_spam_rate:.1f}%")

    # Save the combined dataset
    output_path = output_path
    try:
        # Reorder columns: text, label, source
        final_columns = ["text", "label", "source"]
        combined_df = combined_df[final_columns]

        combined_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\nCombined dataset saved to: {output_path}")
        print(f"Final dataset shape: {combined_df.shape}")

        # Show sample of the data
        print("\nSample of combined data:")
        print(combined_df.head().to_string())

    except Exception as e:
        print(f"Error saving combined dataset: {e}")
