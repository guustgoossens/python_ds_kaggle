"""
Fixed-Window Churn Prediction Model

Addresses temporal leakage by using ONLY the first N days of data for all users,
ensuring equal observation time regardless of when they churned.

Usage:
    python fixed_window_model.py                      # Train and evaluate with CV
    python fixed_window_model.py --submission         # Generate submission file
    python fixed_window_model.py --window 14          # Use 14-day window instead of 7
    python fixed_window_model.py --enhanced           # Use enhanced features (v2)
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def load_raw_data():
    """Load raw event-level parquet files."""
    train = pd.read_parquet('train.parquet')
    test = pd.read_parquet('test.parquet')

    # Convert timestamps
    train['time'] = pd.to_datetime(train['time'], unit='ms')
    test['time'] = pd.to_datetime(test['time'], unit='ms')

    return train, test


def load_churn_labels():
    """Load churn labels from pre-computed features file."""
    features = pd.read_parquet('train_features.parquet')
    return features[['userId', 'churn']].set_index('userId')['churn'].to_dict()


def get_first_event_times(df):
    """Get the first event timestamp for each user."""
    return df.groupby('userId')['time'].min().to_dict()


def filter_to_window(df, first_event_dict, window_days):
    """
    Filter events to only include those within the first N days per user.

    Args:
        df: Event dataframe with 'userId' and 'time' columns
        first_event_dict: Dict mapping userId -> first event timestamp
        window_days: Number of days to include from first event

    Returns:
        Filtered dataframe with only events in the window
    """
    df = df.copy()
    df['user_first_event'] = df['userId'].map(first_event_dict)
    df['days_since_start'] = (df['time'] - df['user_first_event']).dt.total_seconds() / 86400

    # Keep only events within window
    df_window = df[df['days_since_start'] <= window_days].copy()

    return df_window


def create_window_features(df_window, window_days):
    """
    Create user-level features from windowed event data.

    All features are computed from the fixed window only, ensuring
    no temporal leakage from knowing when users churned.

    Args:
        df_window: Event dataframe filtered to window period
        window_days: Window size (for documentation/context)

    Returns:
        DataFrame with one row per user and computed features
    """
    # Basic engagement counts
    features = df_window.groupby('userId').agg(
        # Volume metrics
        events_in_window=('page', 'count'),
        sessions_in_window=('sessionId', 'nunique'),
        songs_in_window=('song', lambda x: x.notna().sum()),
        unique_songs=('song', 'nunique'),
        unique_artists=('artist', 'nunique'),

        # Interaction counts
        thumbs_up=('page', lambda x: (x == 'Thumbs Up').sum()),
        thumbs_down=('page', lambda x: (x == 'Thumbs Down').sum()),
        add_to_playlist=('page', lambda x: (x == 'Add to Playlist').sum()),
        add_friend=('page', lambda x: (x == 'Add Friend').sum()),

        # Friction/issue signals
        errors=('page', lambda x: (x == 'Error').sum()),
        help_visits=('page', lambda x: (x == 'Help').sum()),

        # Ad exposure
        ads=('page', lambda x: (x == 'Roll Advert').sum()),

        # Navigation
        home_visits=('page', lambda x: (x == 'Home').sum()),
        settings_visits=('page', lambda x: (x == 'Settings').sum()),

        # Subscription signals
        downgrades=('page', lambda x: (x == 'Downgrade').sum()),
        upgrades=('page', lambda x: (x == 'Upgrade').sum()),

        # Actual days observed (for rate calculations)
        actual_days=('days_since_start', 'max'),
    ).reset_index()

    # Ensure actual_days is at least 0.1 to avoid division by zero
    features['actual_days'] = features['actual_days'].clip(lower=0.1)

    # Rate features (normalized by observation time)
    features['events_per_day'] = features['events_in_window'] / features['actual_days']
    features['songs_per_day'] = features['songs_in_window'] / features['actual_days']
    features['sessions_per_day'] = features['sessions_in_window'] / features['actual_days']

    # Ratio features (bounded 0-1, handle division by zero)
    total_thumbs = features['thumbs_up'] + features['thumbs_down']
    features['thumbs_up_ratio'] = features['thumbs_up'] / (total_thumbs + 1)
    features['thumbs_down_ratio'] = features['thumbs_down'] / (total_thumbs + 1)

    features['error_rate'] = features['errors'] / (features['events_in_window'] + 1)
    features['ad_rate'] = features['ads'] / (features['events_in_window'] + 1)
    features['help_rate'] = features['help_visits'] / (features['events_in_window'] + 1)

    # Songs per session
    features['songs_per_session'] = features['songs_in_window'] / (features['sessions_in_window'] + 1)

    # Ads per song (frustration metric)
    features['ads_per_song'] = features['ads'] / (features['songs_in_window'] + 1)

    # Engagement diversity
    features['song_variety'] = features['unique_songs'] / (features['songs_in_window'] + 1)
    features['artist_variety'] = features['unique_artists'] / (features['songs_in_window'] + 1)

    return features


def create_trajectory_features(df_window, window_days):
    """
    Create trajectory features comparing early vs late behavior within the window.

    Captures whether user engagement is declining (churn signal) or stable.
    """
    # Split window into early and late periods
    midpoint = window_days / 2

    early = df_window[df_window['days_since_start'] <= midpoint].copy()
    late = df_window[df_window['days_since_start'] > midpoint].copy()

    # Early period aggregates
    early_agg = early.groupby('userId').agg(
        early_events=('page', 'count'),
        early_songs=('song', lambda x: x.notna().sum()),
        early_sessions=('sessionId', 'nunique'),
        early_thumbs_down=('page', lambda x: (x == 'Thumbs Down').sum()),
        early_ads=('page', lambda x: (x == 'Roll Advert').sum()),
    ).reset_index()

    # Late period aggregates
    late_agg = late.groupby('userId').agg(
        late_events=('page', 'count'),
        late_songs=('song', lambda x: x.notna().sum()),
        late_sessions=('sessionId', 'nunique'),
        late_thumbs_down=('page', lambda x: (x == 'Thumbs Down').sum()),
        late_ads=('page', lambda x: (x == 'Roll Advert').sum()),
    ).reset_index()

    # Get all users
    all_users = df_window[['userId']].drop_duplicates()
    features = all_users.merge(early_agg, on='userId', how='left')
    features = features.merge(late_agg, on='userId', how='left')

    # Fill NaN with 0 (user had no events in that period)
    features = features.fillna(0)

    # Trajectory ratios (late / early) - values < 1 indicate decline
    features['events_trajectory'] = (features['late_events'] + 1) / (features['early_events'] + 1)
    features['songs_trajectory'] = (features['late_songs'] + 1) / (features['early_songs'] + 1)
    features['sessions_trajectory'] = (features['late_sessions'] + 1) / (features['early_sessions'] + 1)

    # Absolute changes (negative = decline)
    features['events_change'] = features['late_events'] - features['early_events']
    features['songs_change'] = features['late_songs'] - features['early_songs']

    # Frustration trajectory (increasing thumbs down = bad)
    features['thumbs_down_trajectory'] = (features['late_thumbs_down'] + 1) / (features['early_thumbs_down'] + 1)

    # Did user have activity in late period at all?
    features['has_late_activity'] = (features['late_events'] > 0).astype(int)

    # Keep only trajectory features
    traj_cols = ['userId', 'events_trajectory', 'songs_trajectory', 'sessions_trajectory',
                 'events_change', 'songs_change', 'thumbs_down_trajectory', 'has_late_activity']

    return features[traj_cols]


def create_session_features(df_window):
    """
    Create session-level quality features.

    Aggregates per session first, then computes user-level statistics.
    """
    # Compute per-session metrics
    session_stats = df_window.groupby(['userId', 'sessionId']).agg(
        session_events=('page', 'count'),
        session_songs=('song', lambda x: x.notna().sum()),
        session_duration_seconds=('time', lambda x: (x.max() - x.min()).total_seconds()),
        session_thumbs_down=('page', lambda x: (x == 'Thumbs Down').sum()),
        session_errors=('page', lambda x: (x == 'Error').sum()),
        session_ads=('page', lambda x: (x == 'Roll Advert').sum()),
    ).reset_index()

    # Convert duration to minutes
    session_stats['session_duration_min'] = session_stats['session_duration_seconds'] / 60

    # Aggregate session stats to user level
    user_session_features = session_stats.groupby('userId').agg(
        # Session duration stats
        avg_session_duration=('session_duration_min', 'mean'),
        std_session_duration=('session_duration_min', 'std'),
        max_session_duration=('session_duration_min', 'max'),
        min_session_duration=('session_duration_min', 'min'),

        # Session quality
        avg_songs_per_session=('session_songs', 'mean'),
        std_songs_per_session=('session_songs', 'std'),

        # Problem sessions
        pct_sessions_with_errors=('session_errors', lambda x: (x > 0).mean()),
        pct_sessions_with_thumbs_down=('session_thumbs_down', lambda x: (x > 0).mean()),

        # Session count for weighting
        num_sessions_check=('sessionId', 'count'),
    ).reset_index()

    # Fill NaN (std is NaN if only 1 session)
    user_session_features = user_session_features.fillna(0)

    # Session consistency (low std = consistent user)
    user_session_features['session_duration_cv'] = (
        user_session_features['std_session_duration'] /
        (user_session_features['avg_session_duration'] + 0.1)
    )

    # Drop helper column
    user_session_features = user_session_features.drop('num_sessions_check', axis=1)

    return user_session_features


def create_gap_features(df_window, window_days):
    """
    Create gap/absence features measuring time between sessions.

    Long gaps indicate disengagement.
    """
    # Get session start times
    session_times = df_window.groupby(['userId', 'sessionId']).agg(
        session_start=('time', 'min'),
        session_day=('days_since_start', 'min'),
    ).reset_index()

    # Sort by time within each user
    session_times = session_times.sort_values(['userId', 'session_start'])

    # Compute gaps between sessions
    session_times['prev_session_start'] = session_times.groupby('userId')['session_start'].shift(1)
    session_times['gap_hours'] = (
        (session_times['session_start'] - session_times['prev_session_start'])
        .dt.total_seconds() / 3600
    )

    # Aggregate gap stats per user
    gap_features = session_times.groupby('userId').agg(
        avg_gap_hours=('gap_hours', 'mean'),
        max_gap_hours=('gap_hours', 'max'),
        std_gap_hours=('gap_hours', 'std'),
        num_gaps=('gap_hours', 'count'),
    ).reset_index()

    # Fill NaN (users with only 1 session have no gaps)
    gap_features = gap_features.fillna(0)

    # Count days with zero events
    events_per_day = df_window.groupby(['userId', df_window['days_since_start'].astype(int)]).size()
    events_per_day = events_per_day.reset_index(name='daily_events')
    events_per_day.columns = ['userId', 'day', 'daily_events']

    # For each user, count days in window with activity
    days_with_activity = events_per_day.groupby('userId')['day'].nunique().reset_index()
    days_with_activity.columns = ['userId', 'days_with_activity']

    # Merge
    gap_features = gap_features.merge(days_with_activity, on='userId', how='left')
    gap_features['days_with_activity'] = gap_features['days_with_activity'].fillna(0)

    # Days without activity (within window)
    gap_features['days_inactive'] = window_days - gap_features['days_with_activity']
    gap_features['pct_days_active'] = gap_features['days_with_activity'] / window_days

    return gap_features


def create_enhanced_features(df_window, window_days):
    """
    Create all enhanced features by combining base, trajectory, session, and gap features.
    """
    # Base features
    base = create_window_features(df_window, window_days)

    # Trajectory features
    trajectory = create_trajectory_features(df_window, window_days)

    # Session features
    session = create_session_features(df_window)

    # Gap features
    gap = create_gap_features(df_window, window_days)

    # Merge all
    features = base.merge(trajectory, on='userId', how='left')
    features = features.merge(session, on='userId', how='left')
    features = features.merge(gap, on='userId', how='left')

    # Fill any remaining NaN
    features = features.fillna(0)

    return features


def prepare_for_modeling(train_features, test_features, churn_labels):
    """
    Prepare features for sklearn models.

    Args:
        train_features: Training features DataFrame
        test_features: Test features DataFrame
        churn_labels: Dict mapping userId -> churn label

    Returns:
        X_train, y_train, X_test, test_ids, feature_columns
    """
    # Add churn labels to training data
    train_features = train_features.copy()
    train_features['churn'] = train_features['userId'].map(churn_labels)
    train_features = train_features.dropna(subset=['churn'])

    # Get feature columns (exclude userId and churn)
    feature_cols = [c for c in train_features.columns if c not in ['userId', 'churn']]

    X_train = train_features[feature_cols].fillna(0)
    y_train = train_features['churn'].astype(int)

    X_test = test_features[feature_cols].fillna(0)
    test_ids = test_features['userId'].values

    return X_train, y_train, X_test, test_ids, feature_cols


def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_cv(X, y, cv=5):
    """Evaluate model with stratified cross-validation."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    accuracies = []
    f1_scores = []
    roc_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        accuracies.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        roc_aucs.append(roc_auc_score(y_val, y_proba))

    return {
        'accuracy': (np.mean(accuracies), np.std(accuracies)),
        'f1': (np.mean(f1_scores), np.std(f1_scores)),
        'roc_auc': (np.mean(roc_aucs), np.std(roc_aucs)),
    }


def generate_submission(model, X_test, test_ids, threshold=0.5, output_name='submission_fixed_window'):
    """Generate and save submission file."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    submission = pd.DataFrame({
        'id': test_ids,
        'target': y_pred
    })

    filename = f'{output_name}.csv'
    submission.to_csv(filename, index=False)

    return submission, y_proba


def print_feature_importance(model, feature_cols, top_n=15):
    """Print top feature importances."""
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop {top_n} Feature Importances:")
    print("-" * 40)
    for _, row in importances.head(top_n).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Fixed-window churn prediction')
    parser.add_argument('--window', type=int, default=7,
                        help='Number of days for observation window (default: 7)')
    parser.add_argument('--submission', action='store_true',
                        help='Generate submission file')
    parser.add_argument('--name', type=str, default='submission_fixed_window',
                        help='Output filename (without .csv)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for classification')
    parser.add_argument('--enhanced', action='store_true',
                        help='Use enhanced features (trajectory, session, gap)')
    args = parser.parse_args()

    version = "ENHANCED" if args.enhanced else "BASE"
    print("=" * 70)
    print(f"FIXED-WINDOW CHURN PREDICTION (window={args.window} days, {version})")
    print("=" * 70)

    # Load data
    print("\nLoading raw event data...")
    train_raw, test_raw = load_raw_data()
    churn_labels = load_churn_labels()

    print(f"  Train events: {len(train_raw):,}")
    print(f"  Test events:  {len(test_raw):,}")
    print(f"  Train users:  {train_raw['userId'].nunique():,}")
    print(f"  Test users:   {test_raw['userId'].nunique():,}")

    # Get first event times
    train_first_events = get_first_event_times(train_raw)
    test_first_events = get_first_event_times(test_raw)

    # Filter to window
    print(f"\nFiltering to first {args.window} days per user...")
    train_window = filter_to_window(train_raw, train_first_events, args.window)
    test_window = filter_to_window(test_raw, test_first_events, args.window)

    print(f"  Train events in window: {len(train_window):,} ({len(train_window)/len(train_raw)*100:.1f}%)")
    print(f"  Test events in window:  {len(test_window):,} ({len(test_window)/len(test_raw)*100:.1f}%)")

    # Create features
    print("\nCreating features from windowed data...")
    if args.enhanced:
        print("  Using ENHANCED features (trajectory + session + gap)...")
        train_features = create_enhanced_features(train_window, args.window)
        test_features = create_enhanced_features(test_window, args.window)
    else:
        train_features = create_window_features(train_window, args.window)
        test_features = create_window_features(test_window, args.window)

    print(f"  Features created: {len(train_features.columns) - 1}")

    # Prepare for modeling
    X_train, y_train, X_test, test_ids, feature_cols = prepare_for_modeling(
        train_features, test_features, churn_labels
    )

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples:  {len(X_test)}")
    print(f"  Churn rate:    {y_train.mean()*100:.2f}%")

    # Cross-validation
    print("\nRunning 5-fold cross-validation...")
    cv_results = evaluate_cv(X_train, y_train, cv=5)

    print("\nCross-Validation Results:")
    print("-" * 40)
    print(f"  Accuracy: {cv_results['accuracy'][0]:.4f} (+/- {cv_results['accuracy'][1]:.4f})")
    print(f"  F1 Score: {cv_results['f1'][0]:.4f} (+/- {cv_results['f1'][1]:.4f})")
    print(f"  ROC-AUC:  {cv_results['roc_auc'][0]:.4f} (+/- {cv_results['roc_auc'][1]:.4f})")

    # Train final model
    print("\nTraining final model on full training data...")
    model = train_model(X_train, y_train)

    print_feature_importance(model, feature_cols)

    # Generate submission
    if args.submission:
        print("\nGenerating submission...")
        submission, y_proba = generate_submission(
            model, X_test, test_ids,
            threshold=args.threshold,
            output_name=args.name
        )

        print(f"\nSubmission saved to {args.name}.csv")
        print(f"  Total predictions:   {len(submission)}")
        print(f"  Predicted churners:  {submission['target'].sum()} ({submission['target'].mean()*100:.1f}%)")
        print(f"  Probability range:   [{y_proba.min():.3f}, {y_proba.max():.3f}]")

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == '__main__':
    main()
