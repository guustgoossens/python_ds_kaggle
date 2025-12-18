"""
Submission generation module for churn prediction.

Usage:
    python submission.py                    # Generate submission with default model
    python submission.py --name sprint3     # Custom submission name -> submission_sprint3.csv
    python submission.py --threshold 0.5    # Custom probability threshold
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')


def load_features():
    """Load pre-computed feature files."""
    train = pd.read_parquet('train_features.parquet')
    test = pd.read_parquet('test_features.parquet')
    return train, test


def add_domain_interactions(df):
    """Add domain-driven interaction features based on churn domain knowledge."""
    df = df.copy()

    page_downgrade = df['page_downgrade'] if 'page_downgrade' in df.columns else 0
    cancel_visits = df['cancel_page_visits'] if 'cancel_page_visits' in df.columns else 0
    median_epd = df['events_per_day'].median()

    # Downgrade + declining engagement
    df['downgrade_declining'] = page_downgrade * (df['activity_trend'] < 0).astype(int)

    # Thumbs down + declining engagement
    df['dissatisfaction_combo'] = df['thumbs_down_ratio'] * (1 - df['thumbs_up_ratio'])

    # Paid user seeing lots of ads
    df['paid_ads_issue'] = df['is_paid'] * df['ad_ratio']

    # Cancel page visits * thumbs down
    df['cancel_dissatisfied'] = cancel_visits * df['thumbs_down_ratio']

    # Low engagement paid user
    df['low_engage_paid'] = df['is_paid'] * (df['events_per_day'] < median_epd).astype(int)

    # Error rate * session count
    df['error_sessions'] = df['error_rate'] * df['total_sessions']

    # Days active * activity trend (long-term declining users)
    df['longterm_decline'] = df['days_active'] * (df['activity_trend'] < 0).astype(int) * abs(df['activity_trend'])

    # Songs per day * thumbs down ratio
    df['listening_dissatisfaction'] = df['songs_per_day'] * df['thumbs_down_ratio']

    # Level changes * has_downgrade
    df['subscription_instability'] = df['level_changes'] * df['has_downgrade']

    # Ad ratio * declining trend
    df['ads_declining'] = df['ad_ratio'] * (df['activity_trend'] < 0).astype(int)

    return df


def add_auto_interactions(df, poly, top_features, interaction_names, selected_mask):
    """Add automated polynomial interactions."""
    df = df.copy()
    X_poly = poly.transform(df[top_features])
    for i, (name, include) in enumerate(zip(interaction_names, selected_mask)):
        if include:
            clean_name = name.replace(' ', '_')
            df[f'poly_{clean_name}'] = X_poly[:, i]
    return df


def add_transforms(df, features_to_square, features_to_log):
    """Add squared and log transform features."""
    df = df.copy()

    # Squared terms
    for feat in features_to_square:
        if feat in df.columns:
            df[f'{feat}_squared'] = df[feat] ** 2

    # Log transforms
    for feat in features_to_log:
        if feat in df.columns:
            df[f'{feat}_log'] = np.log1p(df[feat].clip(lower=0))

    return df


def engineer_features(train_df, test_df):
    """Apply all Sprint 3 feature engineering."""

    # Polynomial interaction setup
    top_features = ['days_active', 'events_per_day', 'songs_per_day', 'ad_ratio', 'thumbs_down_ratio']
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly.fit(train_df[top_features])
    interaction_names = poly.get_feature_names_out(top_features)
    selected_mask = np.array(['_' in name or ' ' in name for name in interaction_names])

    # Features for transforms
    features_to_square = ['days_active', 'events_per_day', 'songs_per_day', 'ad_ratio', 'thumbs_down_ratio']

    # Identify highly skewed features
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['userId', 'churn']]
    skewness = train_df[numeric_cols].apply(skew).sort_values(ascending=False)
    highly_skewed = skewness[skewness > 1].index.tolist()[:5]

    # Apply all feature engineering
    train_eng = train_df.copy()
    train_eng = add_domain_interactions(train_eng)
    train_eng = add_auto_interactions(train_eng, poly, top_features, interaction_names, selected_mask)
    train_eng = add_transforms(train_eng, features_to_square, highly_skewed)

    test_eng = test_df.copy()
    test_eng = add_domain_interactions(test_eng)
    test_eng = add_auto_interactions(test_eng, poly, top_features, interaction_names, selected_mask)
    test_eng = add_transforms(test_eng, features_to_square, highly_skewed)

    return train_eng, test_eng


def prepare_for_modeling(train_df, test_df):
    """Prepare features for sklearn models."""

    X_train = train_df.drop(['userId', 'churn'], axis=1, errors='ignore')
    y_train = train_df['churn']
    X_test = test_df.drop(['userId', 'churn'], axis=1, errors='ignore')

    # Encode categorical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    for col in cat_cols:
        X_train = X_train.copy()
        X_test = X_test.copy()

        X_train[col] = X_train[col].fillna('Unknown').astype(str)
        X_test[col] = X_test[col].fillna('Unknown').astype(str)

        all_values = list(set(X_train[col].unique()) | set(X_test[col].unique()))
        le = LabelEncoder()
        le.fit(all_values)

        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    # Align columns
    for col in set(X_train.columns) - set(X_test.columns):
        X_test[col] = 0
    for col in set(X_test.columns) - set(X_train.columns):
        X_train[col] = 0

    X_test = X_test[X_train.columns]

    return X_train, y_train, X_test


def train_model(X_train, y_train):
    """Train the Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def generate_submission(model, X_test, test_ids, threshold=0.5, output_name='submission'):
    """Generate and save submission file."""

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    submission = pd.DataFrame({
        'id': test_ids,
        'target': y_pred
    })

    filename = f'{output_name}.csv'
    submission.to_csv(filename, index=False)

    return submission, y_pred_proba


def main():
    parser = argparse.ArgumentParser(description='Generate churn prediction submission')
    parser.add_argument('--name', type=str, default='submission',
                        help='Output filename (without .csv)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for classification')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    args = parser.parse_args()

    print("Loading features...")
    train_df, test_df = load_features()

    print("Engineering features...")
    train_eng, test_eng = engineer_features(train_df, test_df)

    print("Preparing for modeling...")
    X_train, y_train, X_test = prepare_for_modeling(train_eng, test_eng)

    print("Training model...")
    model = train_model(X_train, y_train)

    if args.verbose:
        train_pred = model.predict(X_train)
        train_acc = (train_pred == y_train).mean()
        print(f"Training accuracy: {train_acc:.4f}")

    print("Generating submission...")
    submission, proba = generate_submission(
        model, X_test, test_eng['userId'],
        threshold=args.threshold,
        output_name=args.name
    )

    # Print summary
    print(f"\nSubmission saved to {args.name}.csv")
    print(f"Total predictions: {len(submission)}")
    print(f"Predicted churners: {submission['target'].sum()} ({submission['target'].mean()*100:.1f}%)")
    print(f"Probability range: [{proba.min():.3f}, {proba.max():.3f}]")


if __name__ == '__main__':
    main()
