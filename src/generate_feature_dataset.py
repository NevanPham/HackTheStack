import pandas as pd
import numpy as np
import re
from pathlib import Path


def extract_basic_features(text):
    """Extract basic text statistics."""
    text = str(text)
    words = text.split()

    char_count = len(text)
    word_count = len(words)
    sentence_count = max(1, len(re.split(r'[.!?]+', text.strip())))
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length
    }


def extract_spam_indicators(text):
    """Extract spam indicator features."""
    text_lower = str(text).lower()
    text_len = max(1, len(text))

    # Spam keywords
    urgency_words = ['urgent', 'act now', 'limited time', 'expires', 'hurry', 'fast', 'don\'t wait', 'now', 'immediately']
    money_words = ['free', 'prize', 'win', 'winner', 'cash', 'money', 'claim', 'reward', '£', '$', 'pounds', 'bonus']
    action_words = ['call', 'click', 'text', 'txt', 'reply', 'send', 'subscribe', 'unsubscribe', 'download']
    suspicious_words = ['congratulations', 'selected', 'guaranteed', 'offer', 'promotion', 'discount']

    urgency_count = sum(word in text_lower for word in urgency_words)
    money_count = sum(word in text_lower for word in money_words)
    action_count = sum(word in text_lower for word in action_words)
    suspicious_count = sum(word in text_lower for word in suspicious_words)

    # Pattern detection
    has_url = int(bool(re.search(r'http[s]?://|www\.|\w+\.(com|org|net|co\.uk)', text_lower)))
    has_email = int(bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)))
    has_phone = int(bool(re.search(r'\b\d{5,}\b|(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3,4}[\s.-]?\d{4}', text)))

    currency_count = text.count('£') + text.count('$')
    has_price = int(bool(re.search(r'£\d+|\$\d+|\d+\s*(pound|dollar|gbp|usd)', text_lower)))
    number_sequences = len(re.findall(r'\b\d{4,}\b', text))

    return {
        'urgency_count': urgency_count,
        'money_count': money_count,
        'action_count': action_count,
        'suspicious_count': suspicious_count,
        'has_url': has_url,
        'has_email': has_email,
        'has_phone': has_phone,
        'currency_count': currency_count,
        'has_price': has_price,
        'number_sequences': number_sequences
    }


def extract_character_features(text):
    """Extract character-level features."""
    text = str(text)
    text_len = max(1, len(text))

    # Punctuation
    exclamation_count = text.count('!')
    question_count = text.count('?')
    dot_count = text.count('.')
    comma_count = text.count(',')

    exclamation_ratio = exclamation_count / text_len
    question_ratio = question_count / text_len
    punctuation_ratio = sum(1 for c in text if c in '!?.,:;') / text_len

    # Case
    upper_count = sum(1 for c in text if c.isupper())
    upper_ratio = upper_count / text_len
    consecutive_caps = len(re.findall(r'[A-Z]{3,}', text))

    # Digits and special characters
    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / text_len
    special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special_char_ratio = special_char_count / text_len

    # Whitespace
    space_count = text.count(' ')
    space_ratio = space_count / text_len

    return {
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'dot_count': dot_count,
        'comma_count': comma_count,
        'exclamation_ratio': exclamation_ratio,
        'question_ratio': question_ratio,
        'punctuation_ratio': punctuation_ratio,
        'upper_ratio': upper_ratio,
        'consecutive_caps': consecutive_caps,
        'digit_count': digit_count,
        'digit_ratio': digit_ratio,
        'special_char_count': special_char_count,
        'special_char_ratio': special_char_ratio,
        'space_ratio': space_ratio
    }


def extract_lexical_features(text):
    """Extract lexical diversity features."""
    words = str(text).lower().split()
    word_count = len(words)

    if word_count == 0:
        return {
            'unique_words': 0,
            'unique_ratio': 0,
            'max_word_repetition': 0
        }

    unique_words = len(set(words))
    unique_ratio = unique_words / word_count

    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    max_word_repetition = max(word_freq.values()) if word_freq else 0

    return {
        'unique_words': unique_words,
        'unique_ratio': unique_ratio,
        'max_word_repetition': max_word_repetition
    }


def extract_all_features(text):
    """Extract all features for a single text."""
    features = {}
    features.update(extract_basic_features(text))
    features.update(extract_spam_indicators(text))
    features.update(extract_character_features(text))
    features.update(extract_lexical_features(text))
    return features


def generate_feature_dataset(
    input_path="../datasets/combined.csv",
    output_path="../datasets/combined_with_features.csv"
):
    """
    Generate a new dataset with extracted features.

    Args:
        input_path: Path to input CSV (text, label, source)
        output_path: Path to output CSV with additional feature columns
    """
    print("=" * 70)
    print("GENERATING FEATURE-ENRICHED DATASET")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records")
    print(f"Spam rate: {df['label'].mean():.2%}")

    # Extract features for all texts
    print("\nExtracting features from all texts...")
    all_features = []
    for idx, text in enumerate(df['text']):
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} records...")
        features = extract_all_features(text)
        all_features.append(features)

    # Convert to DataFrame
    print("\nCreating feature DataFrame...")
    features_df = pd.DataFrame(all_features)

    # Combine with original data
    print("Combining with original data...")
    result_df = pd.concat([df, features_df], axis=1)

    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    print(f"\nSaving to {output_path}...")
    result_df.to_csv(output_path, index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("DATASET GENERATION SUMMARY")
    print("=" * 70)
    print(f"Total records: {len(result_df):,}")
    print(f"Original columns: {len(df.columns)} (text, label, source)")
    print(f"New feature columns: {len(features_df.columns)}")
    print(f"Total columns: {len(result_df.columns)}")
    print(f"\nSpam rate: {result_df['label'].mean():.2%}")
    print(f"Output saved to: {output_path}")

    print("\nFeature columns added:")
    print("  Basic Features (5):")
    print("    - char_count, word_count, sentence_count")
    print("    - avg_word_length, avg_sentence_length")
    print("\n  Spam Indicators (10):")
    print("    - urgency_count, money_count, action_count, suspicious_count")
    print("    - has_url, has_email, has_phone, currency_count")
    print("    - has_price, number_sequences")
    print("\n  Character Features (14):")
    print("    - exclamation_count, question_count, dot_count, comma_count")
    print("    - exclamation_ratio, question_ratio, punctuation_ratio")
    print("    - upper_ratio, consecutive_caps")
    print("    - digit_count, digit_ratio")
    print("    - special_char_count, special_char_ratio, space_ratio")
    print("\n  Lexical Features (3):")
    print("    - unique_words, unique_ratio, max_word_repetition")
    print("=" * 70)

    # Show sample
    print("\nSample of generated data (first 3 rows):")
    print(result_df.head(3).to_string())

    return result_df


if __name__ == "__main__":
    # Generate the feature-enriched dataset
    df = generate_feature_dataset(
        input_path="../datasets/combined.csv",
        output_path="../datasets/combined_with_features.csv"
    )

    print("\n✓ Feature dataset generation complete!")
