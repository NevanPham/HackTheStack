import pandas as pd
import json
from pathlib import Path


def analyze_features(input_path="datasets/combined_with_features.csv"):
    """Analyze features and generate summary statistics for visualization."""
    print("Loading dataset...")
    df = pd.read_csv(input_path)

    # Separate spam and ham
    spam = df[df['label'] == 1]
    ham = df[df['label'] == 0]

    print(f"Total records: {len(df)}")
    print(f"Spam: {len(spam)} ({len(spam)/len(df)*100:.1f}%)")
    print(f"Ham: {len(ham)} ({len(ham)/len(df)*100:.1f}%)")

    # Feature groups
    basic_features = ['char_count', 'word_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length']
    spam_indicators = ['urgency_count', 'money_count', 'action_count', 'suspicious_count',
                       'has_url', 'has_email', 'has_phone', 'currency_count', 'has_price', 'number_sequences']
    character_features = ['exclamation_count', 'question_count', 'exclamation_ratio', 'question_ratio',
                          'punctuation_ratio', 'upper_ratio', 'consecutive_caps', 'digit_ratio', 'special_char_ratio']
    lexical_features = ['unique_words', 'unique_ratio', 'max_word_repetition']

    # 1. Feature importance comparison (spam vs ham means)
    feature_comparison = []
    for features, group_name in [
        (basic_features, 'Basic'),
        (spam_indicators, 'Spam Indicators'),
        (character_features, 'Character'),
        (lexical_features, 'Lexical')
    ]:
        for feat in features:
            spam_mean = spam[feat].mean()
            ham_mean = ham[feat].mean()
            difference = spam_mean - ham_mean
            ratio = spam_mean / ham_mean if ham_mean > 0 else 0

            feature_comparison.append({
                'feature': feat,
                'group': group_name,
                'spam_mean': float(spam_mean),
                'ham_mean': float(ham_mean),
                'difference': float(difference),
                'ratio': float(ratio)
            })

    # Sort by absolute difference
    feature_comparison = sorted(feature_comparison, key=lambda x: abs(x['difference']), reverse=True)

    # 2. Top discriminative features (highest spam/ham ratio)
    top_discriminative = sorted(feature_comparison, key=lambda x: x['ratio'], reverse=True)[:15]

    # 3. Distribution data for key features
    distributions = {}
    key_features = ['word_count', 'money_count', 'urgency_count', 'has_url', 'exclamation_count',
                    'upper_ratio', 'consecutive_caps', 'number_sequences']

    for feat in key_features:
        distributions[feat] = {
            'spam': {
                'min': float(spam[feat].min()),
                'max': float(spam[feat].max()),
                'mean': float(spam[feat].mean()),
                'median': float(spam[feat].median()),
                'std': float(spam[feat].std())
            },
            'ham': {
                'min': float(ham[feat].min()),
                'max': float(ham[feat].max()),
                'mean': float(ham[feat].mean()),
                'median': float(ham[feat].median()),
                'std': float(ham[feat].std())
            }
        }

    # 4. Correlation with spam label
    correlations = []
    all_features = basic_features + spam_indicators + character_features + lexical_features
    for feat in all_features:
        corr = df[feat].corr(df['label'])
        correlations.append({
            'feature': feat,
            'correlation': float(corr)
        })

    correlations = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)

    # 5. Source distribution
    source_dist = {
        'spam': spam['source'].value_counts().to_dict(),
        'ham': ham['source'].value_counts().to_dict()
    }

    # 6. Overall statistics
    overall_stats = {
        'total_messages': len(df),
        'spam_count': len(spam),
        'ham_count': len(ham),
        'spam_rate': float(len(spam) / len(df)),
        'sources': df['source'].value_counts().to_dict()
    }

    # Create output
    output = {
        'overall_stats': overall_stats,
        'feature_comparison': feature_comparison,
        'top_discriminative': top_discriminative,
        'distributions': distributions,
        'correlations': correlations[:20],  # Top 20
        'source_distribution': source_dist
    }

    # Save to JSON
    output_path = Path('spam-detection-app/public/data/feature_analysis.json')
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nAnalysis saved to: {output_path}")
    print("\nTop 10 most discriminative features (spam/ham ratio):")
    for i, item in enumerate(top_discriminative[:10], 1):
        print(f"{i}. {item['feature']}: {item['ratio']:.2f}x ({item['group']})")

    print("\nTop 10 correlated features:")
    for i, item in enumerate(correlations[:10], 1):
        print(f"{i}. {item['feature']}: {item['correlation']:.4f}")

    return output


if __name__ == "__main__":
    analyze_features()
