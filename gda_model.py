"""
Gaussian Discriminant Analysis (GDA) model for churn prediction.

Implements LDA and QDA with appropriate feature transformations to
approximate Gaussian distributions for each feature type.

Usage:
    python gda_model.py                    # Run full comparison
    python gda_model.py --submission       # Generate submission with best model
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')


# --- Feature categorization based on distribution analysis ---

# Count features: Poisson-like, use log(x+1)
COUNT_FEATURES = [
    'total_events', 'total_sessions', 'total_songs', 'unique_songs', 'unique_artists',
    'total_listen_time', 'level_changes', 'cancel_page_visits',
    'page_add_friend', 'page_add_to_playlist', 'page_downgrade', 'page_error',
    'page_help', 'page_home', 'page_logout', 'page_nextsong', 'page_roll_advert',
    'page_settings', 'page_thumbs_down', 'page_thumbs_up', 'page_upgrade'
]

# Rate features: right-skewed, use log(x+1)
RATE_FEATURES = ['events_per_day', 'songs_per_day']

# Ratio features: bounded [0,1], use logit transform
RATIO_FEATURES = [
    'thumbs_up_ratio', 'thumbs_down_ratio', 'ad_ratio', 'error_rate',
    'paid_ratio', 'playlist_add_ratio', 'song_repeat_ratio'
]

# Session/temporal stats: already approximately Gaussian
GAUSSIAN_FEATURES = [
    'avg_session_length', 'max_session_length', 'std_session_length',
    'avg_song_length', 'std_song_length', 'days_active', 'days_since_registration'
]

# Binary features: include as-is
BINARY_FEATURES = ['is_paid', 'has_downgrade', 'has_upgrade']

# Special handling: extreme outliers
SPECIAL_FEATURES = ['activity_trend']

# Categorical features to drop
DROP_FEATURES = ['gender', 'state', 'userId', 'churn']


def logit_transform(x, eps=1e-6):
    """Apply logit transform: log(x / (1-x)) to map [0,1] -> (-inf, inf)."""
    x_clipped = np.clip(x, eps, 1 - eps)
    return np.log(x_clipped / (1 - x_clipped))


def transform_features_for_gda(df):
    """
    Transform features to approximate Gaussianity for GDA.

    Transformations:
    - Count features: log(x+1) to handle Poisson-like distributions
    - Rate features: log(x+1) to handle right-skewed distributions
    - Ratio features: logit transform to handle Beta-like [0,1] distributions
    - activity_trend: clip + shift + log to handle extreme outliers
    - Session/temporal stats: no transform (already ~Gaussian)
    - Binary features: include as-is
    """
    df_transformed = df.copy()

    # 1. Log-transform count features
    for col in COUNT_FEATURES:
        if col in df_transformed.columns:
            df_transformed[col] = np.log1p(df_transformed[col].fillna(0).clip(lower=0))

    # 2. Log-transform rate features
    for col in RATE_FEATURES:
        if col in df_transformed.columns:
            df_transformed[col] = np.log1p(df_transformed[col].fillna(0).clip(lower=0))

    # 3. Logit-transform ratio features
    for col in RATIO_FEATURES:
        if col in df_transformed.columns:
            df_transformed[col] = logit_transform(df_transformed[col].fillna(0.5))

    # 4. Handle activity_trend (clip then shift to positive, then log)
    if 'activity_trend' in df_transformed.columns:
        # Clip to [-10, 10], shift to [1, 21], then log
        clipped = df_transformed['activity_trend'].clip(-10, 10)
        df_transformed['activity_trend'] = np.log1p(clipped + 11)  # shift to make positive

    # 5. Fill NaN in Gaussian features
    for col in GAUSSIAN_FEATURES:
        if col in df_transformed.columns:
            df_transformed[col] = df_transformed[col].fillna(df_transformed[col].median())

    # 6. Binary features: fill NaN with 0
    for col in BINARY_FEATURES:
        if col in df_transformed.columns:
            df_transformed[col] = df_transformed[col].fillna(0)

    # 7. Drop categorical and identifier columns
    for col in DROP_FEATURES:
        if col in df_transformed.columns:
            df_transformed = df_transformed.drop(col, axis=1)

    return df_transformed


def load_features():
    """Load pre-computed feature files."""
    train = pd.read_parquet('train_features.parquet')
    test = pd.read_parquet('test_features.parquet')
    return train, test


def prepare_gda_data(train_df, test_df):
    """Prepare data for GDA models with transformations."""
    y_train = train_df['churn'].values
    test_ids = test_df['userId'].values

    # Apply GDA-specific transformations
    X_train = transform_features_for_gda(train_df)
    X_test = transform_features_for_gda(test_df)

    # Ensure columns match
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # Standardize features (important for GDA)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, test_ids, scaler, common_cols


def evaluate_model(model, X, y, cv=5, model_name="Model"):
    """Evaluate model with stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Collect metrics across folds
    accuracies = []
    f1_scores = []
    roc_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        accuracies.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        roc_aucs.append(roc_auc_score(y_val, y_pred_proba))

    results = {
        'model': model_name,
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'roc_auc_mean': np.mean(roc_aucs),
        'roc_auc_std': np.std(roc_aucs)
    }

    return results


def compare_gda_models(X, y):
    """Compare LDA and QDA with different regularization settings."""
    models = {
        'LDA (auto shrinkage)': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        'LDA (no shrinkage)': LinearDiscriminantAnalysis(solver='svd'),
        'QDA (reg=0.0)': QuadraticDiscriminantAnalysis(reg_param=0.0),
        'QDA (reg=0.1)': QuadraticDiscriminantAnalysis(reg_param=0.1),
        'QDA (reg=0.3)': QuadraticDiscriminantAnalysis(reg_param=0.3),
        'QDA (reg=0.5)': QuadraticDiscriminantAnalysis(reg_param=0.5),
    }

    results = []
    for name, model in models.items():
        print(f"  Evaluating {name}...")
        try:
            result = evaluate_model(model, X, y, cv=5, model_name=name)
            results.append(result)
        except Exception as e:
            print(f"    Error: {e}")
            continue

    return pd.DataFrame(results)


def print_results(results_df):
    """Print formatted comparison results."""
    print("\n" + "="*80)
    print("GDA MODEL COMPARISON RESULTS (5-Fold Stratified CV)")
    print("="*80)

    # Sort by accuracy
    results_df = results_df.sort_values('accuracy_mean', ascending=False)

    print(f"\n{'Model':<25} {'Accuracy':<18} {'F1 Score':<18} {'ROC-AUC':<18}")
    print("-"*80)

    for _, row in results_df.iterrows():
        acc = f"{row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}"
        f1 = f"{row['f1_mean']:.4f} +/- {row['f1_std']:.4f}"
        auc = f"{row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f}"
        print(f"{row['model']:<25} {acc:<18} {f1:<18} {auc:<18}")

    print("\n" + "="*80)

    # Best model
    best = results_df.iloc[0]
    print(f"\nBest model: {best['model']}")
    print(f"  Accuracy: {best['accuracy_mean']:.4f} (+/- {best['accuracy_std']:.4f})")
    print(f"  F1 Score: {best['f1_mean']:.4f} (+/- {best['f1_std']:.4f})")
    print(f"  ROC-AUC:  {best['roc_auc_mean']:.4f} (+/- {best['roc_auc_std']:.4f})")


def analyze_transformations(train_df):
    """Analyze skewness before and after transformations."""
    from scipy.stats import skew

    print("\n" + "="*80)
    print("TRANSFORMATION ANALYSIS: Skewness Before vs After")
    print("="*80)

    # Original features
    original = train_df.drop(['userId', 'churn', 'gender', 'state'], axis=1, errors='ignore')
    original = original.select_dtypes(include=[np.number])

    # Transformed features
    transformed = transform_features_for_gda(train_df)

    print(f"\n{'Feature':<30} {'Original Skew':<15} {'Transformed Skew':<15} {'Change':<10}")
    print("-"*80)

    improvements = []
    for col in transformed.columns:
        if col in original.columns:
            orig_skew = skew(original[col].dropna())
            trans_skew = skew(transformed[col].dropna())
            change = abs(orig_skew) - abs(trans_skew)
            improvements.append((col, orig_skew, trans_skew, change))

    # Sort by improvement
    improvements.sort(key=lambda x: x[3], reverse=True)

    for col, orig, trans, change in improvements[:20]:
        sign = "+" if change > 0 else ""
        print(f"{col:<30} {orig:>12.2f}   {trans:>12.2f}   {sign}{change:>8.2f}")

    avg_orig_skew = np.mean([abs(x[1]) for x in improvements])
    avg_trans_skew = np.mean([abs(x[2]) for x in improvements])
    print(f"\n{'Average |skewness|':<30} {avg_orig_skew:>12.2f}   {avg_trans_skew:>12.2f}")


def generate_submission(model, X_train, y_train, X_test, test_ids, threshold=0.5, output_name='submission_gda'):
    """Generate and save submission file."""
    # Train on full dataset
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    submission = pd.DataFrame({
        'id': test_ids,
        'target': y_pred
    })

    filename = f'{output_name}.csv'
    submission.to_csv(filename, index=False)

    print(f"\nSubmission saved to {filename}")
    print(f"Total predictions: {len(submission)}")
    print(f"Predicted churners: {submission['target'].sum()} ({submission['target'].mean()*100:.1f}%)")
    print(f"Probability range: [{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]")

    return submission, y_pred_proba


def main():
    parser = argparse.ArgumentParser(description='GDA model for churn prediction')
    parser.add_argument('--submission', action='store_true',
                        help='Generate submission with best model')
    parser.add_argument('--name', type=str, default='submission_gda',
                        help='Output filename (without .csv)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for classification')
    parser.add_argument('--analyze', action='store_true',
                        help='Show transformation analysis')
    args = parser.parse_args()

    print("Loading features...")
    train_df, test_df = load_features()
    print(f"  Train: {len(train_df)} users, {train_df.shape[1]} features")
    print(f"  Test:  {len(test_df)} users")
    print(f"  Churn rate: {train_df['churn'].mean()*100:.2f}%")

    if args.analyze:
        analyze_transformations(train_df)

    print("\nPreparing data with GDA transformations...")
    X_train, y_train, X_test, test_ids, scaler, feature_cols = prepare_gda_data(train_df, test_df)
    print(f"  Features after transformation: {len(feature_cols)}")

    print("\nComparing GDA models...")
    results_df = compare_gda_models(X_train, y_train)
    print_results(results_df)

    # Save results
    results_df.to_csv('gda_comparison_results.csv', index=False)
    print("\nResults saved to gda_comparison_results.csv")

    if args.submission:
        print("\nGenerating submission with best model...")
        # LDA (no shrinkage) performed best in evaluation
        best_model = LinearDiscriminantAnalysis(solver='svd')
        generate_submission(best_model, X_train, y_train, X_test, test_ids,
                           threshold=args.threshold, output_name=args.name)


if __name__ == '__main__':
    main()
