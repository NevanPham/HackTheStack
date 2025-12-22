import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import json
from scipy import sparse


def load_preprocessed_data(data_path="datasets/preprocessed"):
    """
    Load preprocessed train/test CSVs for feature engineering.

    Args:
        data_path (str|Path): Directory containing train.csv and test.csv.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data)
    """
    data_dir = Path(data_path)

    print("Loading preprocessed data...")
    train_data = pd.read_csv(data_dir / "train.csv")
    test_data = pd.read_csv(data_dir / "test.csv")

    print(f"Training data: {len(train_data)} records")
    print(f"Test data: {len(test_data)} records")

    return train_data, test_data


def clean_email_artifacts(text):
    """
    Remove common email headers/signatures that bias features.

    Args:
        text (str): Raw text.

    Returns:
        str: Cleaned text.
    """
    text = str(text)
    
    # Remove subject lines (the suspected issue)
    text = re.sub(r'^subject:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsubject:\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'^subject\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsubject\s*', ' ', text, flags=re.IGNORECASE)
    
    # Remove other email headers that might appear
    text = re.sub(r'^(from|to|cc|bcc|date|reply-to):\s*.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove email signatures (common patterns)
    text = re.sub(r'--\s*\n.*', '', text, flags=re.DOTALL)
    text = re.sub(r'sent from my \w+', '', text, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def extract_text_features(texts):
    """
    Compute engineered text features focused on spam indicators.

    Args:
        texts (Iterable[str]): Cleaned texts.

    Returns:
        tuple[np.ndarray, list[str]]: (features, feature_names)
    """
    print("Extracting comprehensive text features...")

    features = []
    for text in texts:
        # First clean email artifacts
        text = clean_email_artifacts(text)
        
        # Basic length features
        char_count = len(text)
        word_count = len(text.split()) if text.split() else 0
        avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0

        # Existing preprocessing token features
        url_count = text.count("<URL>")
        email_count = text.count("<EMAIL>")
        phone_count = text.count("<PHONE>")
        money_count = text.count("<MONEY>")
        excited_count = text.count("<EXCITED>")

        # Basic punctuation ratios
        exclamation_ratio = text.count("!") / char_count if char_count > 0 else 0
        question_ratio = text.count("?") / char_count if char_count > 0 else 0
        upper_ratio = sum(1 for c in text if c.isupper()) / char_count if char_count > 0 else 0

        # Preprocessing for advanced features
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        words = text.split()

        # 1. LINGUISTIC PATTERNS
        # Grammar and syntax anomalies
        grammar_errors = len(re.findall(r'\b(you\'re|your|there|their|its|it\'s|to|too|two)\b', text_lower))
        incomplete_sentences = len([s for s in sentences if s.strip() and not s.strip().endswith(('.', '!', '?'))])
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        
        # 2. URGENCY AND SCARCITY INDICATORS
        # Time pressure words
        time_pressure = len(re.findall(r'\b(limited|expires?|deadline|hurry|quick|immediate|now|today|tonight|asap)\b', text_lower))
        scarcity_words = len(re.findall(r'\b(last|final|only|exclusive|rare|unique|secret|special)\b', text_lower))
        superlatives = len(re.findall(r'\b(best|greatest|amazing|incredible|unbelievable|guaranteed|perfect)\b', text_lower))
        
        # 3. CALL TO ACTION PATTERNS
        call_to_action = len(re.findall(r'\b(click|call|visit|download|install|register|sign up|buy now|order|purchase)\b', text_lower))
        imperative_verbs = len(re.findall(r'\b(get|take|grab|claim|receive|earn|make|win|start|try)\b', text_lower))
        
        # 4. FINANCIAL AND PROMOTIONAL PATTERNS
        # Enhanced money detection
        percentage_mentions = len(re.findall(r'\d+%', text))
        price_patterns = len(re.findall(r'\$\d+|\d+\s*dollars?|\d+\s*bucks|cost|price', text_lower))
        promotional_words = len(re.findall(r'\b(sale|discount|coupon|voucher|rebate|cashback|bonus|reward)\b', text_lower))
        
        # Free and offer patterns
        free_count = len(re.findall(r'\bfree\b', text_lower))
        offer_count = len(re.findall(r'\b(offer|deal|save|win|prize|gift)\b', text_lower))
        
        # 5. EMOTIONAL MANIPULATION
        positive_emotions = len(re.findall(r'\b(amazing|awesome|fantastic|incredible|wonderful|congratulations)\b', text_lower))
        fear_words = len(re.findall(r'\b(warning|alert|danger|risk|lose|miss|mistake|problem|urgent)\b', text_lower))
        excitement_words = len(re.findall(r'\b(winner|lucky|selected|chosen|opportunity|chance)\b', text_lower))
        
        # 6. STRUCTURAL SPAM INDICATORS
        # Character-level patterns
        non_alphabetic_ratio = sum(1 for c in text if not c.isalpha() and not c.isspace()) / char_count if char_count > 0 else 0
        consecutive_capitals = len(re.findall(r'[A-Z]{3,}', text))
        special_char_density = sum(1 for c in text if c in '!@#$%^&*()+={}[]|\\:";\'<>?,./') / char_count if char_count > 0 else 0
        
        # Formatting anomalies
        repeated_chars = len(re.findall(r'(.)\1{2,}', text))
        line_breaks = text.count('\n')
        excessive_spaces = len(re.findall(r'\s{4,}', text))
        
        # 7. CAPS AND EMPHASIS ANALYSIS
        caps_words = len([word for word in words if word.isupper() and len(word) > 2])
        caps_ratio = caps_words / word_count if word_count > 0 else 0
        
        # 8. DIGIT AND NUMERIC ANALYSIS
        digit_ratio = sum(c.isdigit() for c in text) / char_count if char_count > 0 else 0
        number_sequences = len(re.findall(r'\d{4,}', text))  # Long number sequences
        
        # 9. COMMUNICATION PATTERNS
        personal_pronouns = len(re.findall(r'\b(you|your|yours)\b', text_lower))
        generic_greetings = len(re.findall(r'\b(dear|hello|hi|greetings)\s+(sir|madam|friend|customer)\b', text_lower))
        first_person_claims = len(re.findall(r'\bi\s+(am|have|will|can|guarantee)\b', text_lower))
        
        # 10. TECHNICAL SPAM INDICATORS
        suspicious_domains = len(re.findall(r'\.(tk|ml|ga|cf|click|download)', text_lower))
        url_shorteners = len(re.findall(r'\b(bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly)\b', text_lower))
        suspicious_attachments = len(re.findall(r'\.(exe|zip|rar|bat|scr|com)\b', text_lower))
        
        # 11. SPAM PHRASE SCORING
        # Weighted scoring for common spam phrases
        spam_phrases = [
            ('free money', 3), ('make money', 2), ('work from home', 2),
            ('click here', 2), ('act now', 3), ('limited time', 2),
            ('you have won', 3), ('claim now', 3), ('no obligation', 2),
            ('risk free', 2), ('satisfaction guaranteed', 2)
        ]
        
        spam_phrase_score = 0
        for phrase, weight in spam_phrases:
            spam_phrase_score += len(re.findall(phrase, text_lower)) * weight
        
        # 12. SENTENCE STRUCTURE ANALYSIS
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentence_count > 0 else 0
        short_sentences = len([s for s in sentences if s.strip() and len(s.split()) < 5])
        long_sentences = len([s for s in sentences if s.strip() and len(s.split()) > 20])
        
        # 13. WORD COMPLEXITY
        complex_words = len([word for word in words if len(word) > 6])
        simple_words = len([word for word in words if len(word) <= 3])
        complexity_ratio = complex_words / word_count if word_count > 0 else 0
        
        # 14. READABILITY INDICATORS
        # Simple readability approximation
        syllable_count = sum([max(1, len(re.findall(r'[aeiouAEIOU]', word))) for word in words])
        flesch_approx = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * (syllable_count / word_count)) if word_count > 0 else 50
        
        # Compile all features
        features.append([
            # Basic features (0-10)
            char_count, word_count, avg_word_length, sentence_count, avg_sentence_length,
            url_count, email_count, phone_count, money_count, excited_count,
            
            # Punctuation and formatting (11-20)
            exclamation_ratio, question_ratio, upper_ratio, caps_words, caps_ratio,
            non_alphabetic_ratio, special_char_density, repeated_chars, line_breaks, excessive_spaces,
            
            # Linguistic patterns (21-25)
            grammar_errors, incomplete_sentences, avg_words_per_sentence, personal_pronouns, first_person_claims,
            
            # Urgency and scarcity (26-30)
            time_pressure, scarcity_words, superlatives, call_to_action, imperative_verbs,
            
            # Financial and promotional (31-35)
            free_count, offer_count, promotional_words, percentage_mentions, price_patterns,
            
            # Emotional manipulation (36-40)
            positive_emotions, fear_words, excitement_words, generic_greetings, spam_phrase_score,
            
            # Structural indicators (41-45)
            consecutive_capitals, digit_ratio, number_sequences, short_sentences, long_sentences,
            
            # Technical spam indicators (46-50)
            suspicious_domains, url_shorteners, suspicious_attachments, complex_words, simple_words,
            
            # Advanced metrics (51-53)
            complexity_ratio, flesch_approx, syllable_count
        ])

    feature_names = [
        # Basic features (0-10)
        "char_count", "word_count", "avg_word_length", "sentence_count", "avg_sentence_length",
        "url_count", "email_count", "phone_count", "money_count", "excited_count",
        
        # Punctuation and formatting (11-20)
        "exclamation_ratio", "question_ratio", "upper_ratio", "caps_words", "caps_ratio",
        "non_alphabetic_ratio", "special_char_density", "repeated_chars", "line_breaks", "excessive_spaces",
        
        # Linguistic patterns (21-25)
        "grammar_errors", "incomplete_sentences", "avg_words_per_sentence", "personal_pronouns", "first_person_claims",
        
        # Urgency and scarcity (26-30)
        "time_pressure", "scarcity_words", "superlatives", "call_to_action", "imperative_verbs",
        
        # Financial and promotional (31-35)
        "free_count", "offer_count", "promotional_words", "percentage_mentions", "price_patterns",
        
        # Emotional manipulation (36-40)
        "positive_emotions", "fear_words", "excitement_words", "generic_greetings", "spam_phrase_score",
        
        # Structural indicators (41-45)
        "consecutive_capitals", "digit_ratio", "number_sequences", "short_sentences", "long_sentences",
        
        # Technical spam indicators (46-50)
        "suspicious_domains", "url_shorteners", "suspicious_attachments", "complex_words", "simple_words",
        
        # Advanced metrics (51-53)
        "complexity_ratio", "flesch_approx", "syllable_count"
    ]

    print(f"Extracted {len(feature_names)} comprehensive text features per sample")
    return np.array(features), feature_names


def create_source_artifact_filter():
    """
    Return a set of tokens to remove as source-specific artifacts.

    Returns:
        set[str]: Artifact tokens.
    """
    # From diagnostic analysis
    source_artifacts = [
        # Enron-specific terms
        'enron', 'vince', 'kaminski', 'ect', 'hou', 'cc',
        # Single/double character tokens (often meaningless)
        '0', '00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
        '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
        '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
        '3', '30', '31', '4', '5', '6', '7', '8', '9',
        # Single letters (often artifacts)
        'j', 's', 't', 'm', 'n', 'r', 'l', 'k', 'p', 'b', 'c', 'd', 'f', 'g', 'h',
        # Common email artifacts
        'fwd', 'fw', 're', 'msg', 'orig', 'mailto'
    ]
    return set(source_artifacts)


def filter_source_artifacts_from_text(text):
    """
    Remove source artifacts and short/digit-only tokens from text.

    Args:
        text (str): Input text.

    Returns:
        str: Filtered text.
    """
    import re
    
    # Get source artifacts to filter
    source_artifacts = create_source_artifact_filter()
    
    # Split into words and filter
    words = text.lower().split()
    filtered_words = []
    
    for word in words:
        # Clean word of punctuation for checking
        clean_word = re.sub(r'[^\w]', '', word)
        
        # Keep word if it's not a source artifact and has reasonable length
        if (clean_word not in source_artifacts and 
            len(clean_word) > 2 and 
            not clean_word.isdigit()):
            filtered_words.append(word)
    
    return ' '.join(filtered_words)


def create_tfidf_features(train_texts, test_texts, max_features=500):
    """
    Build TF-IDF features after artifact filtering.

    Args:
        train_texts (Iterable[str]): Training texts.
        test_texts (Iterable[str]): Test texts.
        max_features (int): TF-IDF vocabulary size.

    Returns:
        tuple: (train_tfidf, test_tfidf, vectorizer)
            train_tfidf/test_tfidf are scipy.sparse matrices.
    """
    print(f"Creating SOURCE-FILTERED TF-IDF features (max_features={max_features})...")

    # Clean texts first to remove email artifacts
    train_texts_cleaned = [clean_email_artifacts(text) for text in train_texts]
    test_texts_cleaned = [clean_email_artifacts(text) for text in test_texts]

    # Pre-filter source artifacts from texts
    print("Pre-filtering source artifacts from texts...")
    train_texts_filtered = [filter_source_artifacts_from_text(text) for text in train_texts_cleaned]
    test_texts_filtered = [filter_source_artifacts_from_text(text) for text in test_texts_cleaned]

    # Standard TF-IDF with optimized parameters (no custom analyzer needed)
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),            # Unigrams and bigrams only
        stop_words='english',          # Remove common English words
        lowercase=True,
        token_pattern=r"\b\w+\b|<\w+>",  # Include special tokens
        min_df=5,                      # Increased from 3 to filter rare tokens
        max_df=0.6,                    # Decreased from 0.8 to remove common tokens
        sublinear_tf=True
    )

    # Fit and transform on filtered texts
    train_tfidf = tfidf.fit_transform(train_texts_filtered)
    test_tfidf = tfidf.transform(test_texts_filtered)

    print(f"TF-IDF shape - Train: {train_tfidf.shape}, Test: {test_tfidf.shape}")
    
    # Debug: Show top TF-IDF features to verify artifact removal
    feature_names = tfidf.get_feature_names_out()
    print(f"Sample TF-IDF features (post-filtering): {feature_names[:10].tolist()}")
    
    # Check if source artifacts were successfully filtered
    source_artifacts = create_source_artifact_filter()
    artifacts_found = [token for token in feature_names if token in source_artifacts]
    if artifacts_found:
        print(f"WARNING: {len(artifacts_found)} source artifacts still present: {artifacts_found[:5]}")
    else:
        print(f"SUCCESS: Source artifacts successfully filtered from vocabulary")
    
    return train_tfidf, test_tfidf, tfidf


def create_source_features(sources, all_sources):
    """
    One-hot encode source labels in fixed order.

    Args:
        sources (Iterable[str]): Sample sources.
        all_sources (Iterable[str]): Reference list of sources.

    Returns:
        tuple[np.ndarray, list[str]]: (features, feature_names)
    """
    print("Creating source features...")
    
    source_features = []
    for source in sources:
        source_vector = [1 if source == s else 0 for s in all_sources]
        source_features.append(source_vector)

    feature_names = [f"source_{source}" for source in all_sources]
    return np.array(source_features), feature_names


def create_visualization_features(scaled_features):
    """
    Reduce features to 2D with PCA for visualization.

    Args:
        scaled_features (np.ndarray): Scaled features.

    Returns:
        tuple[np.ndarray, PCA]: (features_2d, pca_model)
    """
    print("Creating 2D features for visualization...")
    
    # Use PCA to reduce to 2D for visualization
    pca_2d = PCA(n_components=2, random_state=42)
    features_2d = pca_2d.fit_transform(scaled_features)
    
    print(f"2D PCA explained variance: {pca_2d.explained_variance_ratio_.sum():.3f}")
    
    return features_2d, pca_2d


def engineer_features(
    data_path="datasets/preprocessed",
    output_dir="preprocessed/kmeans",
    max_tfidf_features=50,
    source_weight=0
):
    """
    End-to-end feature engineering for K-Means.

    Args:
        data_path (str|Path): Preprocessed data directory.
        output_dir (str|Path): Output directory for features.
        max_tfidf_features (int): TF-IDF vocabulary size.
        source_weight (float): Weight applied to source features.

    Returns:
        tuple[Path, dict]: (output_path, metadata)

    Outputs:
        - X_train.npy, X_test.npy, X_train_2d.npy, X_test_2d.npy
        - y_train.npy, y_test.npy, train_sources.npy, test_sources.npy
        - X_train_tfidf.npz, X_test_tfidf.npz, tfidf_vectorizer.pkl
        - scaler.pkl, pca_2d.pkl, feature_metadata.json
    """
    print("Starting ENHANCED feature engineering for clustering")
    print("=" * 60)
    print(f"Source feature weight: {source_weight}")
    print("NEW: Email artifact removal enabled")
    print("NEW: Enhanced spam-specific features")
    print("NEW: Diagnostic data preservation")

    # Load data
    train_data, test_data = load_preprocessed_data(data_path)
    all_sources = sorted(train_data["source"].unique())
    print(f"Sources: {all_sources}")

    # Extract enhanced text features
    train_text_features, text_feature_names = extract_text_features(train_data["cleaned_text"])
    test_text_features, _ = extract_text_features(test_data["cleaned_text"])

    # Create enhanced TF-IDF features
    train_tfidf, test_tfidf, tfidf_vectorizer = create_tfidf_features(
        train_data["cleaned_text"], test_data["cleaned_text"], max_tfidf_features
    )

    # Create source features
    train_source_features, source_feature_names = create_source_features(
        train_data["source"], all_sources
    )
    test_source_features, _ = create_source_features(test_data["source"], all_sources)

    # Apply weight to source features
    print(f"Applying weight {source_weight} to source features...")
    train_source_features_weighted = train_source_features * source_weight
    test_source_features_weighted = test_source_features * source_weight

    # Combine and scale features
    scaler = StandardScaler()
    
    train_tfidf_dense = train_tfidf.toarray()
    train_combined = np.hstack([train_tfidf_dense, train_text_features, train_source_features_weighted])
    train_scaled = scaler.fit_transform(train_combined)
    
    test_tfidf_dense = test_tfidf.toarray()
    test_combined = np.hstack([test_tfidf_dense, test_text_features, test_source_features_weighted])
    test_scaled = scaler.transform(test_combined)

    # Create 2D visualization features
    train_2d, pca_2d = create_visualization_features(train_scaled)
    test_2d = pca_2d.transform(test_scaled)

    # Create feature names
    tfidf_names = [f"tfidf_{feature}" for feature in tfidf_vectorizer.get_feature_names_out()]
    all_feature_names = tfidf_names + text_feature_names + source_feature_names

    # Save features and metadata
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Save standard features
    np.save(output_path / "X_train.npy", train_scaled)
    np.save(output_path / "X_test.npy", test_scaled)
    np.save(output_path / "X_train_2d.npy", train_2d)
    np.save(output_path / "X_test_2d.npy", test_2d)
    np.save(output_path / "y_train.npy", train_data["label"].values)
    np.save(output_path / "y_test.npy", test_data["label"].values)
    np.save(output_path / "train_sources.npy", train_data["source"].values)
    np.save(output_path / "test_sources.npy", test_data["source"].values)

    # DIAGNOSTIC: Save TF-IDF sparse matrices and vectorizer for token analysis
    sparse.save_npz(output_path / "X_train_tfidf.npz", train_tfidf)
    sparse.save_npz(output_path / "X_test_tfidf.npz", test_tfidf)
    
    # Save preprocessing objects
    joblib.dump(tfidf_vectorizer, output_path / "tfidf_vectorizer.pkl")
    joblib.dump(scaler, output_path / "scaler.pkl")
    joblib.dump(pca_2d, output_path / "pca_2d.pkl")

    # Create enhanced metadata
    metadata = {
        "n_train_samples": len(train_data),
        "n_test_samples": len(test_data),
        "n_features": train_scaled.shape[1],
        "n_tfidf_features": len(tfidf_names),
        "n_text_features": len(text_feature_names),
        "n_source_features": len(source_feature_names),
        "feature_names": all_feature_names,
        "text_feature_names": text_feature_names,
        "source_feature_names": source_feature_names,
        "tfidf_feature_names": tfidf_names,
        "sources": all_sources,
        "max_tfidf_features": max_tfidf_features,
        "source_weight": source_weight,
        "train_spam_rate": train_data["label"].mean(),
        "test_spam_rate": test_data["label"].mean(),
        "scaling_method": "StandardScaler",
        "pca_2d_explained_variance": pca_2d.explained_variance_ratio_.sum(),
        "enhancements": [
            "Email artifact removal (subject lines, headers)",
            "Enhanced spam keyword detection",
            "Capitalization and formatting analysis",
            "SOURCE ARTIFACT FILTERING - Removed Enron/organizational tokens",
            "Optimized TF-IDF (min_df=5, max_df=0.6) for content focus",
            "Custom analyzer filtering source-specific vocabulary",
            "Diagnostic TF-IDF matrices preserved"
        ],
        "diagnostic_files": [
            "X_train_tfidf.npz - Sparse TF-IDF training matrix",
            "X_test_tfidf.npz - Sparse TF-IDF test matrix",
            "tfidf_vectorizer.pkl - Fitted TF-IDF vectorizer"
        ],
        "notes": f"Enhanced features with email cleaning, source weight {source_weight}, and diagnostic data"
    }

    # Save metadata
    with open(output_path / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("=" * 60)
    print("Enhanced Feature Engineering Summary:")
    print(f"Total features: {train_scaled.shape[1]}")
    print(f"  - TF-IDF features: {len(tfidf_names)}")
    print(f"  - Text features: {len(text_feature_names)} (enhanced with spam indicators)")
    print(f"  - Source features: {len(source_feature_names)} (weighted by {source_weight})")
    print(f"Email artifacts removed: subject lines, headers, signatures")
    print(f"Enhanced spam detection: keywords, caps, urgency indicators")
    print(f"DIAGNOSTIC: TF-IDF matrices saved for token analysis")
    print(f"2D visualization features created (PCA explained variance: {pca_2d.explained_variance_ratio_.sum():.3f})")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Features saved to: {output_path}")

    return output_path, metadata