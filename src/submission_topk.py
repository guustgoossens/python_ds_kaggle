"""
Top-K% Threshold Submission Generator

Uses fixed 7-day window features with ensemble model (GB + RF),
then applies top-K% threshold instead of probability > 0.5.

Usage:
    python submission_topk.py                    # Generate top 30% submission (best)
    python submission_topk.py --pct 0.22         # Generate top 22% submission
    python submission_topk.py --all              # Generate all thresholds (22%, 30%, 35%, 40%, 46%, 50%)
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def load_raw_data():
    """Load raw event-level parquet files."""
    train = pd.read_parquet('train.parquet')
    test = pd.read_parquet('test.parquet')
    train['time'] = pd.to_datetime(train['time'], unit='ms')
    test['time'] = pd.to_datetime(test['time'], unit='ms')
    return train, test


def load_churn_labels():
    """Load churn labels from pre-computed features file."""
    features = pd.read_parquet('train_features.parquet')
    return features.set_index('userId')['churn'].to_dict()


def filter_to_window(df, first_event_dict, window_days):
    """Filter events to first N days per user."""
    df = df.copy()
    df['first'] = df['userId'].map(first_event_dict)
    df['day'] = (df['time'] - df['first']).dt.total_seconds() / 86400
    return df[df['day'] <= window_days]


def create_features(df):
    """Create features from windowed event data."""
    features = df.groupby('userId').agg(
        events=('page', 'count'),
        sessions=('sessionId', 'nunique'),
        songs=('song', lambda x: x.notna().sum()),
        thumbs_up=('page', lambda x: (x == 'Thumbs Up').sum()),
        thumbs_down=('page', lambda x: (x == 'Thumbs Down').sum()),
        ads=('page', lambda x: (x == 'Roll Advert').sum()),
        downgrades=('page', lambda x: (x == 'Downgrade').sum()),
        errors=('page', lambda x: (x == 'Error').sum()),
        actual_days=('day', 'max'),
    ).reset_index()

    features['actual_days'] = features['actual_days'].clip(lower=0.1)

    # Log transforms
    features['log_events'] = np.log1p(features['events'])
    features['log_sessions'] = np.log1p(features['sessions'])
    features['log_songs'] = np.log1p(features['songs'])

    # Rate features
    features['events_per_day'] = features['events'] / features['actual_days']
    features['songs_per_day'] = features['songs'] / features['actual_days']
    features['ad_rate'] = features['ads'] / (features['events'] + 1)
    features['thumbs_down_rate'] = features['thumbs_down'] / (features['thumbs_up'] + features['thumbs_down'] + 1)
    features['ads_per_song'] = features['ads'] / (features['songs'] + 1)
    features['error_rate'] = features['errors'] / (features['events'] + 1)

    return features


def train_ensemble(X_train, y_train):
    """Train GB + RF ensemble."""
    gb = GradientBoostingClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.85,
        min_samples_leaf=3,
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=6,
        class_weight='balanced',
        random_state=43,
        n_jobs=-1
    )

    gb.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return gb, rf


def get_ensemble_proba(gb, rf, X, gb_weight=0.65):
    """Get ensemble probabilities."""
    rf_weight = 1 - gb_weight
    proba = gb_weight * gb.predict_proba(X)[:, 1] + rf_weight * rf.predict_proba(X)[:, 1]
    return proba


def generate_topk_submission(proba, test_ids, pct, output_name=None):
    """Generate submission using top K% threshold."""
    n_churn = int(len(proba) * pct)
    top_idx = np.argsort(proba)[-n_churn:]

    predictions = np.zeros(len(proba), dtype=int)
    predictions[top_idx] = 1

    submission = pd.DataFrame({
        'id': test_ids,
        'target': predictions
    })

    if output_name is None:
        output_name = f'submission_top{int(pct*100)}'

    filename = f'{output_name}.csv'
    submission.to_csv(filename, index=False)

    return submission, filename


def evaluate_cv(X, y, gb, rf, pct):
    """Evaluate top-K% threshold with CV."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Clone and fit
        gb_clone = GradientBoostingClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.03,
            subsample=0.85, min_samples_leaf=3, random_state=42
        )
        rf_clone = RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_leaf=6,
            class_weight='balanced', random_state=43, n_jobs=-1
        )

        gb_clone.fit(X_tr, y_tr)
        rf_clone.fit(X_tr, y_tr)

        proba = get_ensemble_proba(gb_clone, rf_clone, X_val)

        n_churn = int(len(proba) * pct)
        top_idx = np.argsort(proba)[-n_churn:]
        pred = np.zeros(len(proba), dtype=int)
        pred[top_idx] = 1

        scores.append(accuracy_score(y_val, pred))

    return np.mean(scores), np.std(scores)


def main():
    parser = argparse.ArgumentParser(description='Top-K% threshold submission generator')
    parser.add_argument('--pct', type=float, default=0.30,
                        help='Percentage of users to predict as churners (default: 0.30)')
    parser.add_argument('--window', type=int, default=7,
                        help='Number of days for observation window (default: 7)')
    parser.add_argument('--all', action='store_true',
                        help='Generate submissions for all thresholds (22%%, 30%%, 35%%, 40%%, 46%%, 50%%)')
    parser.add_argument('--cv', action='store_true',
                        help='Run cross-validation before generating submission')
    parser.add_argument('--name', type=str, default=None,
                        help='Output filename (without .csv)')
    args = parser.parse_args()

    print("=" * 70)
    print(f"TOP-K% THRESHOLD SUBMISSION (window={args.window} days)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train_raw, test_raw = load_raw_data()
    churn_labels = load_churn_labels()

    train_first = train_raw.groupby('userId')['time'].min().to_dict()
    test_first = test_raw.groupby('userId')['time'].min().to_dict()

    # Filter to window
    print(f"Filtering to first {args.window} days...")
    train_w = filter_to_window(train_raw, train_first, args.window)
    test_w = filter_to_window(test_raw, test_first, args.window)

    # Create features
    print("Creating features...")
    train_feat = create_features(train_w)
    test_feat = create_features(test_w)

    train_feat['churn'] = train_feat['userId'].map(churn_labels)
    train_feat = train_feat.dropna(subset=['churn'])

    feat_cols = [c for c in train_feat.columns if c not in ['userId', 'churn']]
    X_train = train_feat[feat_cols].fillna(0)
    y_train = train_feat['churn'].astype(int)
    X_test = test_feat[feat_cols].fillna(0)
    test_ids = test_feat['userId'].values

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feat_cols)}")

    # Train ensemble
    print("\nTraining ensemble (GB + RF)...")
    gb, rf = train_ensemble(X_train, y_train)

    # Get test probabilities
    proba = get_ensemble_proba(gb, rf, X_test)
    print(f"  Probability range: [{proba.min():.3f}, {proba.max():.3f}]")

    # Generate submissions
    if args.all:
        thresholds = [0.22, 0.30, 0.35, 0.40, 0.46, 0.50]
        print(f"\nGenerating submissions for all thresholds...")

        if args.cv:
            print("\nCross-validation results:")
            for pct in thresholds:
                acc, std = evaluate_cv(X_train, y_train, gb, rf, pct)
                print(f"  Top {pct*100:.0f}%: {acc:.4f} (+/- {std:.4f})")

        print("\nGenerating files:")
        for pct in thresholds:
            sub, fname = generate_topk_submission(proba, test_ids, pct)
            n_churn = sub['target'].sum()
            print(f"  {fname}: {n_churn} churners ({pct*100:.0f}%)")
    else:
        pct = args.pct

        if args.cv:
            print(f"\nCross-validation for top {pct*100:.0f}%...")
            acc, std = evaluate_cv(X_train, y_train, gb, rf, pct)
            print(f"  CV Accuracy: {acc:.4f} (+/- {std:.4f})")

        print(f"\nGenerating submission for top {pct*100:.0f}%...")
        sub, fname = generate_topk_submission(proba, test_ids, pct, args.name)
        n_churn = sub['target'].sum()
        print(f"\nSaved {fname}")
        print(f"  Predicted churners: {n_churn} ({n_churn/len(sub)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == '__main__':
    main()
