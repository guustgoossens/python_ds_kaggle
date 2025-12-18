"""
Advanced model comparison for churn prediction.

Tests 10 advanced models on both Sprint 2 (baseline) and Sprint 3 (enhanced) feature sets.
Outputs comparison results to CSV and generates submission files for top models.

Usage:
    python advanced_models.py
"""

from submission import load_features, engineer_features
from imblearn.ensemble import BalancedRandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Gradient boosting libraries

# Imbalanced learning

# Reuse feature engineering from submission.py


def prepare_features(train_df, test_df, apply_engineering=True):
    """Prepare features for modeling, optionally applying Sprint 3 engineering."""
    if apply_engineering:
        train_df, test_df = engineer_features(train_df, test_df)

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

    return X_train, y_train, X_test, test_df['userId']


def get_models(n_neg, n_pos):
    """Return dictionary of 10 advanced models with imbalance-aware settings."""
    scale = n_neg / n_pos  # ~3.5 for 22% churn rate

    # Base models for ensembles
    xgb = XGBClassifier(
        scale_pos_weight=scale,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    lgbm = LGBMClassifier(
        is_unbalance=True,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    catboost = CatBoostClassifier(
        auto_class_weights='Balanced',
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=0
    )

    models = {
        '1_XGBoost': XGBClassifier(
            scale_pos_weight=scale,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),

        '2_LightGBM': LGBMClassifier(
            is_unbalance=True,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),

        '3_CatBoost': CatBoostClassifier(
            auto_class_weights='Balanced',
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        ),

        '4_GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),

        '5_HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42
        ),

        '6_ExtraTrees': ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),

        '7_AdaBoost': AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),

        '8_VotingEnsemble': VotingClassifier(
            estimators=[
                ('xgb', xgb),
                ('lgbm', lgbm),
                ('catboost', catboost)
            ],
            voting='soft',
            n_jobs=-1
        ),

        '9_StackingEnsemble': StackingClassifier(
            estimators=[
                ('xgb', XGBClassifier(
                    scale_pos_weight=scale,
                    n_estimators=50,
                    max_depth=4,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )),
                ('lgbm', LGBMClassifier(
                    is_unbalance=True,
                    n_estimators=50,
                    max_depth=4,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=50,
                    max_depth=6,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ))
            ],
            final_estimator=LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            cv=3,
            n_jobs=-1
        ),

        '10_BalancedRandomForest': BalancedRandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    }

    return models


def evaluate_model(model, X, y, cv=5):
    """Evaluate model using stratified cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    acc_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

    return {
        'accuracy_mean': acc_scores.mean(),
        'accuracy_std': acc_scores.std(),
        'f1_mean': f1_scores.mean(),
        'f1_std': f1_scores.std(),
        'roc_auc_mean': auc_scores.mean(),
        'roc_auc_std': auc_scores.std()
    }


def generate_submission(model, X_train, y_train, X_test, test_ids, filename):
    """Train model on full training data and generate submission."""
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    submission = pd.DataFrame({
        'id': test_ids,
        'target': y_pred
    })
    submission.to_csv(filename, index=False)

    return submission, y_pred_proba


def main():
    print("=" * 60)
    print("Advanced Model Comparison for Churn Prediction")
    print("=" * 60)

    # Load features
    print("\nLoading features...")
    train_df, test_df = load_features()

    # Prepare both feature sets
    print("Preparing Sprint 2 features (baseline)...")
    X_sprint2, y, X_test_sprint2, test_ids = prepare_features(
        train_df.copy(), test_df.copy(), apply_engineering=False
    )
    print(f"  Sprint 2: {X_sprint2.shape[1]} features")

    print("Preparing Sprint 3 features (enhanced)...")
    X_sprint3, _, X_test_sprint3, _ = prepare_features(
        train_df.copy(), test_df.copy(), apply_engineering=True
    )
    print(f"  Sprint 3: {X_sprint3.shape[1]} features")

    # Calculate class imbalance
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    print(
        f"\nClass distribution: {n_neg} non-churn, {n_pos} churn ({n_pos/(n_neg+n_pos)*100:.1f}%)")

    # Get models
    models = get_models(n_neg, n_pos)

    # Store results
    results = []

    # Evaluate on Sprint 2 features
    print("\n" + "-" * 60)
    print("Evaluating models on Sprint 2 features...")
    print("-" * 60)

    for i, (name, model) in enumerate(models.items(), 1):
        print(f"  [{i:2d}/10] {name}...", end=" ", flush=True)
        try:
            metrics = evaluate_model(model, X_sprint2, y)
            print(f"Acc={metrics['accuracy_mean']:.4f}, "
                  f"F1={metrics['f1_mean']:.4f}, "
                  f"AUC={metrics['roc_auc_mean']:.4f}")
            results.append({
                'model': name,
                'feature_set': 'Sprint2',
                'n_features': X_sprint2.shape[1],
                **metrics
            })
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'model': name,
                'feature_set': 'Sprint2',
                'n_features': X_sprint2.shape[1],
                'accuracy_mean': np.nan,
                'accuracy_std': np.nan,
                'f1_mean': np.nan,
                'f1_std': np.nan,
                'roc_auc_mean': np.nan,
                'roc_auc_std': np.nan
            })

    # Evaluate on Sprint 3 features
    print("\n" + "-" * 60)
    print("Evaluating models on Sprint 3 features...")
    print("-" * 60)

    for i, (name, model) in enumerate(models.items(), 1):
        print(f"  [{i:2d}/10] {name}...", end=" ", flush=True)
        try:
            # Need fresh model instance for Sprint 3
            fresh_models = get_models(n_neg, n_pos)
            metrics = evaluate_model(fresh_models[name], X_sprint3, y)
            print(f"Acc={metrics['accuracy_mean']:.4f}, "
                  f"F1={metrics['f1_mean']:.4f}, "
                  f"AUC={metrics['roc_auc_mean']:.4f}")
            results.append({
                'model': name,
                'feature_set': 'Sprint3',
                'n_features': X_sprint3.shape[1],
                **metrics
            })
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'model': name,
                'feature_set': 'Sprint3',
                'n_features': X_sprint3.shape[1],
                'accuracy_mean': np.nan,
                'accuracy_std': np.nan,
                'f1_mean': np.nan,
                'f1_std': np.nan,
                'roc_auc_mean': np.nan,
                'roc_auc_std': np.nan
            })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\n" + "=" * 60)
    print("Results saved to model_comparison_results.csv")

    # Find best models
    valid_results = results_df.dropna(subset=['accuracy_mean'])

    best_overall = valid_results.loc[valid_results['accuracy_mean'].idxmax()]
    best_sprint2 = valid_results[valid_results['feature_set'] == 'Sprint2'].loc[
        valid_results[valid_results['feature_set'] == 'Sprint2']['accuracy_mean'].idxmax()
    ]
    best_sprint3 = valid_results[valid_results['feature_set'] == 'Sprint3'].loc[
        valid_results[valid_results['feature_set'] == 'Sprint3']['accuracy_mean'].idxmax()
    ]

    print("\n" + "-" * 60)
    print("Best Models:")
    print("-" * 60)
    print(f"  Overall: {best_overall['model']} on {best_overall['feature_set']} "
          f"(Acc: {best_overall['accuracy_mean']:.4f})")
    print(f"  Sprint 2:   {best_sprint2['model']} (Acc: {best_sprint2['accuracy_mean']:.4f})")
    print(f"  Sprint 3:   {best_sprint3['model']} (Acc: {best_sprint3['accuracy_mean']:.4f})")

    # Generate submissions
    print("\n" + "-" * 60)
    print("Generating submissions...")
    print("-" * 60)

    # Best overall
    print(f"  Training {best_overall['model']} for best overall submission...")
    best_overall_models = get_models(n_neg, n_pos)
    if best_overall['feature_set'] == 'Sprint2':
        generate_submission(
            best_overall_models[best_overall['model']],
            X_sprint2, y, X_test_sprint2, test_ids,
            'submission_best_overall.csv'
        )
    else:
        generate_submission(
            best_overall_models[best_overall['model']],
            X_sprint3, y, X_test_sprint3, test_ids,
            'submission_best_overall.csv'
        )
    print("    -> submission_best_overall.csv")

    # Best Sprint 2
    print(f"  Training {best_sprint2['model']} for best Sprint 2 submission...")
    best_sprint2_models = get_models(n_neg, n_pos)
    generate_submission(
        best_sprint2_models[best_sprint2['model']],
        X_sprint2, y, X_test_sprint2, test_ids,
        'submission_best_sprint2.csv'
    )
    print("    -> submission_best_sprint2.csv")

    # Best Sprint 3
    print(f"  Training {best_sprint3['model']} for best Sprint 3 submission...")
    best_sprint3_models = get_models(n_neg, n_pos)
    generate_submission(
        best_sprint3_models[best_sprint3['model']],
        X_sprint3, y, X_test_sprint3, test_ids,
        'submission_best_sprint3.csv'
    )
    print("    -> submission_best_sprint3.csv")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)

    summary = results_df.pivot_table(
        index='model',
        columns='feature_set',
        values='accuracy_mean',
        aggfunc='first'
    ).round(4)

    print(summary.to_string())

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
