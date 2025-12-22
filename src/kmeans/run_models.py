import numpy as np
import joblib
import json
from pathlib import Path
import sys

# Use relative import since this file is in src/kmeans/
from feature_engineer import (
    clean_email_artifacts,
    filter_source_artifacts_from_text,
    extract_text_features,
    create_source_features
)


def load_model_and_preprocessors(model_dir):
    """
    Load trained K-Means model, mapping, vectorizer, scaler, and metadata.

    Args:
        model_dir (str|Path): Directory like models/kmeans/k3/<config>/.

    Returns:
        tuple: (model, cluster_mapping, tfidf_vectorizer, scaler, metadata)
    """
    model_path = Path(model_dir)
    
    # Determine features directory from model directory structure
    # Model is in: models/kmeans/k{X}/{config_name}/
    # Features are in: preprocessed/kmeans/{config_name}/
    config_name = model_path.name  # e.g., "current"
    features_dir = Path("preprocessed/kmeans") / config_name
    
    print(f"Loading model from: {model_path}")
    print(f"Loading preprocessors from: {features_dir}")
    
    # Load model and mapping
    model = joblib.load(model_path / "kmeans_model.pkl")
    cluster_mapping = joblib.load(model_path / "cluster_mapping.pkl")
    
    # Load preprocessing objects
    tfidf_vectorizer = joblib.load(features_dir / "tfidf_vectorizer.pkl")
    scaler = joblib.load(features_dir / "scaler.pkl")
    
    # Load metadata for configuration
    with open(features_dir / "feature_metadata.json", "r") as f:
        metadata = json.load(f)
    
    print(f"Model loaded: K={model.n_clusters} clusters")
    print(f"Cluster mapping: {cluster_mapping}")
    print(f"Total features: {metadata['n_features']}")
    print()
    
    return model, cluster_mapping, tfidf_vectorizer, scaler, metadata


def preprocess_single_text(text, source, tfidf_vectorizer, metadata):
    """
    Apply the same preprocessing pipeline used in training.

    Args:
        text (str): Input text.
        source (str): Source label (e.g., 'sms').
        tfidf_vectorizer: Trained TF-IDF vectorizer.
        metadata (dict): Feature metadata (includes 'sources', 'source_weight').

    Returns:
        np.ndarray: Combined features vector.
    """
    
    # Step 1: Clean email artifacts
    cleaned_text = clean_email_artifacts(text)
    
    # Step 2: Extract text features (54 features)
    text_features, _ = extract_text_features([cleaned_text])
    text_features = text_features[0]  # Get single sample
    
    # Step 3: Apply TF-IDF with source artifact filtering
    filtered_text = filter_source_artifacts_from_text(cleaned_text)
    tfidf_features = tfidf_vectorizer.transform([filtered_text])
    tfidf_features_dense = tfidf_features.toarray()[0]
    
    # Step 4: Create source features
    all_sources = metadata['sources']
    source_features, _ = create_source_features([source], all_sources)
    source_features = source_features[0]
    
    # Apply source weight
    source_weight = metadata['source_weight']
    source_features_weighted = source_features * source_weight
    
    # Step 5: Combine all features in correct order: [TF-IDF + text + source]
    combined_features = np.hstack([
        tfidf_features_dense,
        text_features,
        source_features_weighted
    ])
    
    return combined_features


def predict_spam(text, source, model, cluster_mapping, tfidf_vectorizer, scaler, metadata):
    """
    Predict spam/ham by mapping predicted cluster to majority class.

    Args:
        text (str): Input text.
        source (str): Source label.
        model: Trained K-Means.
        cluster_mapping (dict): Cluster→class mapping.
        tfidf_vectorizer: TF-IDF vectorizer.
        scaler: Fitted scaler.
        metadata (dict): Feature metadata.

    Returns:
        dict: {'text','source','cluster','prediction','label'}
    """
    
    # Preprocess the text
    features = preprocess_single_text(text, source, tfidf_vectorizer, metadata)
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Predict cluster
    cluster = model.predict(features_scaled)[0]
    
    # Map cluster to spam/ham
    prediction = cluster_mapping.get(cluster, 0)
    
    return {
        'text': text,
        'source': source,
        'cluster': int(cluster),
        'prediction': int(prediction),
        'label': 'spam' if prediction == 1 else 'ham'
    }


def get_test_messages():
    """
    Provide predefined test messages (2 spam + 2 ham per source).

    Returns:
        list[dict]: Test messages with 'text' and 'source'.
    """
    
    test_messages = [
        # SMS - 2 spam, 2 ham
        {
            "text": "URGENT! You've WON a $1000 Walmart gift card. Click here to claim NOW! Limited time offer!",
            "source": "sms",
            "expected": "spam"
        },
        {
            "text": "Congratulations! You have been selected to receive a FREE iPhone 15! Call 555-0123 immediately!",
            "source": "sms",
            "expected": "spam"
        },
        {
            "text": "Hey, are you free for dinner tonight? Let me know!",
            "source": "sms",
            "expected": "ham"
        },
        {
            "text": "Your package will arrive tomorrow between 2-4pm. Track your delivery at our website.",
            "source": "sms",
            "expected": "ham"
        },
        
        # Email - 2 spam, 2 ham
        {
            "text": "Dear valued customer, ACT NOW! Limited time offer - 90% discount on all products. Click here before midnight!",
            "source": "email",
            "expected": "spam"
        },
        {
            "text": "You have inherited $5,000,000 from a distant relative. Send your bank details to claim your fortune now!",
            "source": "email",
            "expected": "spam"
        },
        {
            "text": "Hi team, please review the attached quarterly report and provide feedback by Friday. Thanks!",
            "source": "email",
            "expected": "ham"
        },
        {
            "text": "Your subscription renewal is due next week. You can manage your account settings in the member portal.",
            "source": "email",
            "expected": "ham"
        },
        
        # YouTube - 2 spam, 2 ham
        {
            "text": "Make $5000 per week working from home! Click my profile link to learn this secret method! GUARANTEED!",
            "source": "youtube",
            "expected": "spam"
        },
        {
            "text": "FREE GIFT CARDS! Visit this website now! Limited slots available! You won't believe this opportunity!",
            "source": "youtube",
            "expected": "spam"
        },
        {
            "text": "Great video! Really helpful tutorial. I learned a lot from your explanation. Keep up the good work!",
            "source": "youtube",
            "expected": "ham"
        },
        {
            "text": "Thanks for sharing this content. Could you make a follow-up video about advanced techniques?",
            "source": "youtube",
            "expected": "ham"
        },
        
        # Review - 2 spam, 2 ham
        {
            "text": "BEST PRODUCT EVER!!! BUY NOW at www.suspiciouslink.com for 90% discount!!! Amazing deal don't miss out!!!",
            "source": "review",
            "expected": "spam"
        },
        {
            "text": "This product changed my life! Visit my website to get the secret discount code! 5 stars!",
            "source": "review",
            "expected": "spam"
        },
        {
            "text": "Good quality product. Delivered on time and works as described. Would recommend to others.",
            "source": "review",
            "expected": "ham"
        },
        {
            "text": "Decent product but shipping took longer than expected. Customer service was helpful when I contacted them.",
            "source": "review",
            "expected": "ham"
        }
    ]
    
    return test_messages


def main():
    """
    CLI to run predictions on predefined messages for a given model directory.

    Usage:
        python run_models.py models/kmeans/k3/current

    Returns:
        None
    """
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_models.py <model_directory>")
        print("\nExample:")
        print("  python run_models.py models/kmeans/k3/current")
        print("\nThis will test the model with 16 predefined messages (2 spam + 2 ham for each source)")
        return
    
    model_dir = sys.argv[1]
    
    # Load model and preprocessors
    model, cluster_mapping, tfidf_vectorizer, scaler, metadata = load_model_and_preprocessors(model_dir)
    
    # Get test messages
    test_messages = get_test_messages()
    
    print("=" * 80)
    print("TESTING K-MEANS SPAM DETECTION MODEL")
    print("=" * 80)
    print(f"Model directory: {model_dir}")
    print(f"Total test messages: {len(test_messages)}")
    print("=" * 80)
    print()
    
    # Track results
    correct = 0
    total = len(test_messages)
    results_by_source = {"sms": [], "email": [], "youtube": [], "review": []}
    
    # Test each message
    for i, test_msg in enumerate(test_messages, 1):
        text = test_msg["text"]
        source = test_msg["source"]
        expected = test_msg["expected"]
        
        # Make prediction
        result = predict_spam(text, source, model, cluster_mapping, tfidf_vectorizer, scaler, metadata)
        
        # Check if correct
        is_correct = result['label'] == expected
        if is_correct:
            correct += 1
        
        # Store result
        results_by_source[source].append({
            'text': text[:60] + "..." if len(text) > 60 else text,
            'expected': expected,
            'predicted': result['label'],
            'cluster': result['cluster'],
            'correct': is_correct
        })
        
        # Print result
        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        print(f"Test {i}/{total} [{source.upper()}] {status}")
        print(f"  Text: {text[:70]}...")
        print(f"  Expected: {expected.upper()} | Predicted: {result['label'].upper()} | Cluster: {result['cluster']}")
        print()
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Overall Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print()
    
    # Summary by source
    for source in ["sms", "email", "youtube", "review"]:
        source_results = results_by_source[source]
        source_correct = sum(1 for r in source_results if r['correct'])
        source_total = len(source_results)
        print(f"{source.upper()}: {source_correct}/{source_total} correct ({source_correct/source_total*100:.1f}%)")
    
    print("=" * 80)


if __name__ == "__main__":
    main()