import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib
import json


def load_data(data_path="../datasets/combined.csv"):
    """Load the combined dataset."""
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} records")
    print(f"Spam rate: {data['label'].mean():.2%}")
    return data


def extract_basic_features(texts):
    """Extract fundamental text statistics."""
    print("Extracting basic text features...")

    features = []
    for text in texts:
        text = str(text)
        words = text.split()

        # Length features
        char_count = len(text)
        word_count = len(words)
        sentence_count = max(1, len(re.split(r'[.!?]+', text.strip())))

        # Average features
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        features.append([
            char_count,
            word_count,
            sentence_count,
            avg_word_length,
            avg_sentence_length
        ])

    feature_names = [
        'char_count',
        'word_count',
        'sentence_count',
        'avg_word_length',
        'avg_sentence_length'
    ]

    return np.array(features), feature_names


def extract_spam_indicators(texts):
    """Extract features commonly found in spam messages."""
    print("Extracting spam indicator features...")

    # Spam keywords
    urgency_words = ['urgent', 'act now', 'limited time', 'expires', 'hurry', 'fast', 'don\'t wait']
    money_words = ['free', 'prize', 'win', 'winner', 'cash', 'money', 'claim', 'reward', '£', '$', 'pounds']
    action_words = ['call', 'click', 'text', 'txt', 'reply', 'send', 'subscribe', 'unsubscribe']
    suspicious_words = ['congratulations', 'selected', 'guaranteed', 'bonus', 'offer', 'promotion']

    features = []
    for text in texts:
        text_lower = str(text).lower()
        text_len = max(1, len(text))

        # Keyword counts
        urgency_count = sum(word in text_lower for word in urgency_words)
        money_count = sum(word in text_lower for word in money_words)
        action_count = sum(word in text_lower for word in action_words)
        suspicious_count = sum(word in text_lower for word in suspicious_words)

        # Special pattern detection
        has_url = bool(re.search(r'http[s]?://|www\.|\w+\.(com|org|net|co\.uk)', text_lower))
        has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        has_phone = bool(re.search(r'\b\d{5,}\b|(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3,4}[\s.-]?\d{4}', text))

        # Money patterns
        currency_count = text.count('£') + text.count('$')
        has_price = bool(re.search(r'£\d+|\$\d+|\d+\s*(pound|dollar|gbp|usd)', text_lower))

        # Number patterns
        number_sequences = len(re.findall(r'\b\d{4,}\b', text))

        features.append([
            urgency_count,
            money_count,
            action_count,
            suspicious_count,
            int(has_url),
            int(has_email),
            int(has_phone),
            currency_count,
            int(has_price),
            number_sequences
        ])

    feature_names = [
        'urgency_count',
        'money_count',
        'action_count',
        'suspicious_count',
        'has_url',
        'has_email',
        'has_phone',
        'currency_count',
        'has_price',
        'number_sequences'
    ]

    return np.array(features), feature_names


def extract_character_features(texts):
    """Extract character-level features."""
    print("Extracting character features...")

    features = []
    for text in texts:
        text = str(text)
        text_len = max(1, len(text))

        # Punctuation counts
        exclamation_count = text.count('!')
        question_count = text.count('?')
        dot_count = text.count('.')
        comma_count = text.count(',')

        # Punctuation ratios
        exclamation_ratio = exclamation_count / text_len
        question_ratio = question_count / text_len
        punctuation_ratio = sum(1 for c in text if c in '!?.,:;') / text_len

        # Case features
        upper_count = sum(1 for c in text if c.isupper())
        lower_count = sum(1 for c in text if c.islower())
        upper_ratio = upper_count / text_len

        # Special characters
        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / text_len
        special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_char_ratio = special_char_count / text_len

        # Consecutive capitals (spam often has ALL CAPS words)
        consecutive_caps = len(re.findall(r'[A-Z]{3,}', text))

        # Whitespace features
        space_count = text.count(' ')
        space_ratio = space_count / text_len

        features.append([
            exclamation_count,
            question_count,
            dot_count,
            comma_count,
            exclamation_ratio,
            question_ratio,
            punctuation_ratio,
            upper_ratio,
            digit_count,
            digit_ratio,
            special_char_count,
            special_char_ratio,
            consecutive_caps,
            space_ratio
        ])

    feature_names = [
        'exclamation_count',
        'question_count',
        'dot_count',
        'comma_count',
        'exclamation_ratio',
        'question_ratio',
        'punctuation_ratio',
        'upper_ratio',
        'digit_count',
        'digit_ratio',
        'special_char_count',
        'special_char_ratio',
        'consecutive_caps',
        'space_ratio'
    ]

    return np.array(features), feature_names


def extract_lexical_diversity(texts):
    """Extract lexical diversity features."""
    print("Extracting lexical diversity features...")

    features = []
    for text in texts:
        words = str(text).lower().split()
        word_count = len(words)

        if word_count == 0:
            features.append([0, 0, 0])
            continue

        # Unique words
        unique_words = len(set(words))
        unique_ratio = unique_words / word_count

        # Repeated words (spam often repeats words)
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        max_word_repetition = max(word_freq.values()) if word_freq else 0

        features.append([
            unique_words,
            unique_ratio,
            max_word_repetition
        ])

    feature_names = [
        'unique_words',
        'unique_ratio',
        'max_word_repetition'
    ]

    return np.array(features), feature_names


def extract_source_features(sources):
    """One-hot encode message sources."""
    print("Extracting source features...")

    unique_sources = sorted(set(sources))
    source_to_idx = {src: idx for idx, src in enumerate(unique_sources)}

    features = []
    for source in sources:
        one_hot = [0] * len(unique_sources)
        one_hot[source_to_idx[source]] = 1
        features.append(one_hot)

    feature_names = [f'source_{src}' for src in unique_sources]

    return np.array(features), feature_names, unique_sources


def create_tfidf_features(texts, max_features=3000, ngram_range=(1, 2), vectorizer=None):
    """Create TF-IDF features."""
    print(f"Creating TF-IDF features (max_features={max_features}, ngram_range={ngram_range})...")

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            strip_accents='unicode',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
    else:
        tfidf_matrix = vectorizer.transform(texts)

    print(f"TF-IDF shape: {tfidf_matrix.shape}")
    print(f"Sparsity: {tfidf_matrix.nnz / np.prod(tfidf_matrix.shape):.4f}")

    return tfidf_matrix, vectorizer


def combine_all_features(tfidf_features, *numerical_feature_arrays):
    """Combine TF-IDF with numerical features."""
    print("Combining all features...")

    # Stack all numerical features
    numerical_features = np.hstack(numerical_feature_arrays)

    # Convert to sparse and combine with TF-IDF
    numerical_sparse = sparse.csr_matrix(numerical_features)
    combined = sparse.hstack([tfidf_features, numerical_sparse])

    print(f"Combined shape: {combined.shape}")
    print(f"Combined sparsity: {combined.nnz / np.prod(combined.shape):.4f}")

    return combined


def engineer_features(
    data_path="../datasets/combined.csv",
    output_dir="../datasets/engineered_features",
    test_size=0.2,
    random_state=42,
    max_tfidf_features=3000,
    ngram_range=(1, 2)
):
    """
    Main feature engineering pipeline.

    Args:
        data_path: Path to combined.csv
        output_dir: Output directory for features
        test_size: Test split ratio
        random_state: Random seed
        max_tfidf_features: Max TF-IDF features
        ngram_range: N-gram range for TF-IDF

    Returns:
        Path to output directory
    """
    print("=" * 70)
    print("SPAM DETECTION FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    # Load data
    data = load_data(data_path)

    # Split into train/test
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data['label']
    )

    print(f"\nTrain size: {len(train_data)} ({len(train_data)/len(data):.1%})")
    print(f"Test size: {len(test_data)} ({len(test_data)/len(data):.1%})")
    print(f"Train spam rate: {train_data['label'].mean():.2%}")
    print(f"Test spam rate: {test_data['label'].mean():.2%}")

    # Extract features from training data
    train_basic, basic_names = extract_basic_features(train_data['text'])
    train_spam_ind, spam_ind_names = extract_spam_indicators(train_data['text'])
    train_char, char_names = extract_character_features(train_data['text'])
    train_lex, lex_names = extract_lexical_diversity(train_data['text'])
    train_source, source_names, unique_sources = extract_source_features(train_data['source'])

    # Extract features from test data
    test_basic, _ = extract_basic_features(test_data['text'])
    test_spam_ind, _ = extract_spam_indicators(test_data['text'])
    test_char, _ = extract_character_features(test_data['text'])
    test_lex, _ = extract_lexical_diversity(test_data['text'])
    test_source, _, _ = extract_source_features(test_data['source'])

    # Create TF-IDF features
    train_tfidf, tfidf_vectorizer = create_tfidf_features(
        train_data['text'],
        max_features=max_tfidf_features,
        ngram_range=ngram_range
    )
    test_tfidf, _ = create_tfidf_features(
        test_data['text'],
        vectorizer=tfidf_vectorizer
    )

    # Combine all features
    X_train = combine_all_features(
        train_tfidf,
        train_basic,
        train_spam_ind,
        train_char,
        train_lex,
        train_source
    )

    X_test = combine_all_features(
        test_tfidf,
        test_basic,
        test_spam_ind,
        test_char,
        test_lex,
        test_source
    )

    y_train = train_data['label'].values
    y_test = test_data['label'].values

    # Create feature names
    tfidf_names = [f'tfidf_{name}' for name in tfidf_vectorizer.get_feature_names_out()]
    all_feature_names = tfidf_names + basic_names + spam_ind_names + char_names + lex_names + source_names

    # Save features
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("\nSaving features...")
    sparse.save_npz(output_path / 'X_train.npz', X_train)
    sparse.save_npz(output_path / 'X_test.npz', X_test)
    joblib.dump(y_train, output_path / 'y_train.pkl')
    joblib.dump(y_test, output_path / 'y_test.pkl')
    joblib.dump(tfidf_vectorizer, output_path / 'tfidf_vectorizer.pkl')

    # Save metadata
    metadata = {
        'n_train_samples': len(train_data),
        'n_test_samples': len(test_data),
        'n_features': X_train.shape[1],
        'n_tfidf_features': train_tfidf.shape[1],
        'n_basic_features': len(basic_names),
        'n_spam_indicator_features': len(spam_ind_names),
        'n_character_features': len(char_names),
        'n_lexical_features': len(lex_names),
        'n_source_features': len(source_names),
        'feature_names': all_feature_names,
        'basic_feature_names': basic_names,
        'spam_indicator_feature_names': spam_ind_names,
        'character_feature_names': char_names,
        'lexical_feature_names': lex_names,
        'source_feature_names': source_names,
        'unique_sources': unique_sources,
        'train_spam_rate': float(train_data['label'].mean()),
        'test_spam_rate': float(test_data['label'].mean()),
        'test_size': test_size,
        'random_state': random_state,
        'max_tfidf_features': max_tfidf_features,
        'ngram_range': ngram_range
    }

    with open(output_path / 'feature_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"Total features: {X_train.shape[1]:,}")
    print(f"  - TF-IDF features: {train_tfidf.shape[1]:,}")
    print(f"  - Basic text features: {len(basic_names)}")
    print(f"  - Spam indicator features: {len(spam_ind_names)}")
    print(f"  - Character features: {len(char_names)}")
    print(f"  - Lexical diversity features: {len(lex_names)}")
    print(f"  - Source features: {len(source_names)}")
    print(f"\nTraining samples: {len(train_data):,}")
    print(f"Test samples: {len(test_data):,}")
    print(f"Matrix sparsity: {X_train.nnz / np.prod(X_train.shape):.4f}")
    print(f"\nFeatures saved to: {output_path}")
    print("=" * 70)

    return output_path


if __name__ == "__main__":
    # Run feature engineering
    output_dir = engineer_features(
        data_path="../datasets/combined.csv",
        output_dir="../datasets/engineered_features",
        test_size=0.2,
        random_state=42,
        max_tfidf_features=3000,
        ngram_range=(1, 2)
    )

    print(f"\nFeature engineering complete! Output saved to: {output_dir}")
