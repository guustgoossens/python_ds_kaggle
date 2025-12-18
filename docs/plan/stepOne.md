# Churn Prediction Project - 4-Sprint Implementation Plan

## Key Insights from Data Exploration

### Data Quality Summary
| Issue                               | Severity | Action                                 |
| ----------------------------------- | -------- | -------------------------------------- |
| Song/artist/length missing (18%)    | None     | Expected - only on non-music pages     |
| Demographics missing in test (15%)  | Low      | Logged-out sessions - impute "Unknown" |
| Song length outliers (0.5-3024 sec) | Low      | Cap at 1200 seconds                    |
| Class imbalance (22% churn)         | Medium   | Use class_weight='balanced'            |

### Activity Levels
- Churned users: ~918 events/user average
- Non-churned users: ~913 events/user average
- Activity levels are similar - churn is not driven by low engagement

### Data Structure
- Train: 17.5M events → 19,140 unique users (4,271 churned = 22%)
- Test: 4.4M events → 2,904 unique users (predict these)
- Time range: Oct 1 - Nov 20, 2018 (50 days)
- No user overlap between train/test (good!)

### Design Decisions (Confirmed)

1. **Cancel page handling**: Keep Cancel visits that occur >12 hours before Cancellation Confirmation. This captures "considered cancelling" behavior while avoiding leakage from the final cancellation flow.

2. **Temporal slicing**: 3 slices per churned user (balanced approach)

3. **Location features**: Extract state/region from location string (reduces 875 → ~50 categories)

4. **Submission threshold**: Use 0.5 probability threshold (standard)

---

## Sprint 1: Data Pipeline & Feature Engineering Foundation

### Goal: Transform event logs → user-level features

### Step 1.1: Load Data and Create Labels

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load data
train = pd.read_parquet('train.parquet')
test = pd.read_parquet('test.parquet')

# Create user-level churn labels
churned_users = train[train['page'] == 'Cancellation Confirmation']['userId'].unique()
print(f"Churned users: {len(churned_users)}")  # Should be 4,271

# Get churn timestamps for temporal slicing later
churn_times = train[train['page'] == 'Cancellation Confirmation'].groupby('userId')['time'].first()
```

### Step 1.2: Basic Feature Engineering Function

```python
def create_user_features(df, user_ids, label_dict=None, observation_end=None):
    """
    Aggregate event-level data to user-level features.

    Parameters:
    - df: DataFrame with events
    - user_ids: list of user IDs to process
    - label_dict: {userId: label} for training data
    - observation_end: cutoff timestamp (for temporal slicing)
    """

    features = []

    for user_id in user_ids:
        user_data = df[df['userId'] == user_id].copy()

        # Apply observation window cutoff if specified
        if observation_end is not None:
            user_data = user_data[user_data['time'] <= observation_end]

        if len(user_data) == 0:
            continue

        # ===== ENGAGEMENT FEATURES =====
        f = {'userId': user_id}

        # Total activity
        f['total_events'] = len(user_data)
        f['total_songs'] = (user_data['page'] == 'NextSong').sum()
        f['total_sessions'] = user_data['sessionId'].nunique()

        # Session stats
        session_lengths = user_data.groupby('sessionId').size()
        f['avg_session_length'] = session_lengths.mean()
        f['max_session_length'] = session_lengths.max()

        # ===== PAGE TYPE COUNTS =====
        page_counts = user_data['page'].value_counts()
        important_pages = ['NextSong', 'Thumbs Up', 'Thumbs Down', 'Add to Playlist',
                          'Add Friend', 'Downgrade', 'Upgrade', 'Error', 'Help',
                          'Home', 'Settings', 'Roll Advert', 'Logout']

        for page in important_pages:
            f[f'page_{page.lower().replace(" ", "_")}'] = page_counts.get(page, 0)

        # ===== BEHAVIORAL RATIOS =====
        if f['total_songs'] > 0:
            f['thumbs_up_ratio'] = f['page_thumbs_up'] / f['total_songs']
            f['thumbs_down_ratio'] = f['page_thumbs_down'] / f['total_songs']
            f['playlist_add_ratio'] = f['page_add_to_playlist'] / f['total_songs']
        else:
            f['thumbs_up_ratio'] = 0
            f['thumbs_down_ratio'] = 0
            f['playlist_add_ratio'] = 0

        # Error rate
        f['error_rate'] = f['page_error'] / f['total_events'] if f['total_events'] > 0 else 0

        # ===== TEMPORAL FEATURES =====
        f['days_active'] = (user_data['time'].max() - user_data['time'].min()).days + 1
        f['days_since_registration'] = (user_data['time'].max() - user_data['registration'].iloc[0]).days

        # Recency: days since last activity (relative to observation end or data end)
        if observation_end:
            f['days_since_last_activity'] = (observation_end - user_data['time'].max()).days
        else:
            f['days_since_last_activity'] = 0

        # Activity trend: compare first half vs second half
        mid_time = user_data['time'].min() + (user_data['time'].max() - user_data['time'].min()) / 2
        first_half = len(user_data[user_data['time'] <= mid_time])
        second_half = len(user_data[user_data['time'] > mid_time])
        f['activity_trend'] = (second_half - first_half) / max(first_half, 1)  # positive = increasing

        # ===== SUBSCRIPTION FEATURES =====
        f['is_paid'] = 1 if user_data['level'].iloc[-1] == 'paid' else 0
        f['level_changes'] = (user_data['level'] != user_data['level'].shift()).sum() - 1

        # Has downgrade/upgrade events (strong churn signals!)
        f['has_downgrade'] = 1 if f['page_downgrade'] > 0 else 0
        f['has_upgrade'] = 1 if f['page_upgrade'] > 0 else 0

        # ===== CANCEL PAGE FEATURE (>12 hours before churn only) =====
        # Count Cancel page visits, excluding those within 12 hours of Cancellation Confirmation
        cancel_visits = user_data[user_data['page'] == 'Cancel']
        if len(cancel_visits) > 0:
            # If user churned, exclude Cancel visits within 12 hours of churn
            churn_event = user_data[user_data['page'] == 'Cancellation Confirmation']
            if len(churn_event) > 0:
                churn_time = churn_event['time'].iloc[0]
                safe_cancels = cancel_visits[cancel_visits['time'] < churn_time - pd.Timedelta(hours=12)]
                f['cancel_page_visits'] = len(safe_cancels)
            else:
                f['cancel_page_visits'] = len(cancel_visits)
        else:
            f['cancel_page_visits'] = 0

        # ===== DEMOGRAPHICS =====
        f['gender'] = user_data['gender'].iloc[0] if pd.notna(user_data['gender'].iloc[0]) else 'Unknown'

        # ===== LOCATION (Extract state) =====
        location = user_data['location'].iloc[0] if pd.notna(user_data['location'].iloc[0]) else 'Unknown'
        # Location format: "City, ST" - extract state code
        if location != 'Unknown' and ',' in location:
            f['state'] = location.split(',')[-1].strip()[:2]  # Get last 2 chars after comma
        else:
            f['state'] = 'Unknown'

        # ===== CONTENT DIVERSITY =====
        songs_played = user_data[user_data['page'] == 'NextSong']
        f['unique_songs'] = songs_played['song'].nunique() if len(songs_played) > 0 else 0
        f['unique_artists'] = songs_played['artist'].nunique() if len(songs_played) > 0 else 0

        # Song length stats (capped at 1200 seconds)
        if len(songs_played) > 0 and songs_played['length'].notna().any():
            lengths = songs_played['length'].clip(upper=1200)
            f['avg_song_length'] = lengths.mean()
            f['total_listen_time'] = lengths.sum()
        else:
            f['avg_song_length'] = 0
            f['total_listen_time'] = 0

        # ===== ADD LABEL =====
        if label_dict is not None:
            f['churn'] = label_dict.get(user_id, 0)

        features.append(f)

    return pd.DataFrame(features)
```

### Step 1.3: Create Training Dataset

```python
# Create label dictionary
all_users = train['userId'].unique()
label_dict = {user: 1 if user in churned_users else 0 for user in all_users}

# Create features for all training users
print("Creating features for training users...")
train_features = create_user_features(train, all_users, label_dict)
print(f"Training set shape: {train_features.shape}")
print(f"Churn distribution:\n{train_features['churn'].value_counts()}")

# Save intermediate result
train_features.to_parquet('train_features.parquet', index=False)
```

### Step 1.4: Create Test Dataset

```python
# Create features for test users
test_users = test['userId'].unique()
print("Creating features for test users...")
test_features = create_user_features(test, test_users)
print(f"Test set shape: {test_features.shape}")

# Save intermediate result
test_features.to_parquet('test_features.parquet', index=False)
```

### Sprint 1 Deliverables:
- [ ] Feature engineering function working
- [ ] Train features created (~19,140 rows, 30+ features)
- [ ] Test features created (~2,904 rows)
- [ ] Saved as parquet files for quick loading

---

## Sprint 2: Baseline Models & Validation Strategy

### Goal: Train baseline models, establish validation approach

### Step 2.1: Prepare Data for Modeling

```python
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load features
train_df = pd.read_parquet('train_features.parquet')

# Separate features and target
X = train_df.drop(['userId', 'churn'], axis=1)
y = train_df['churn']

# Encode categorical features
le_gender = LabelEncoder()
X['gender'] = le_gender.fit_transform(X['gender'])

le_state = LabelEncoder()
X['state'] = le_state.fit_transform(X['state'])

# Split: use temporal-aware split (users who churned earlier for train)
# Or use stratified random split for simplicity
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}")
print(f"Train churn rate: {y_train.mean():.2%}")
print(f"Val churn rate: {y_val.mean():.2%}")

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

### Step 2.2: Baseline Model - Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

# Train logistic regression with balanced class weights
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

# Evaluate
y_pred_lr = lr.predict(X_val_scaled)
print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_lr):.4f}")
print(classification_report(y_val, y_pred_lr))

# Feature importance
coef_df = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs(lr.coef_[0])
}).sort_values('importance', ascending=False)
print("\nTop 10 Features:")
print(coef_df.head(10))
```

### Step 2.3: Random Forest Model

```python
from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)  # No scaling needed for tree models

# Evaluate
y_pred_rf = rf.predict(X_val)
print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_rf):.4f}")
print(classification_report(y_val, y_pred_rf))

# Feature importance
rf_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Features:")
print(rf_importance.head(10))
```

### Step 2.4: Cross-Validation Setup

```python
from sklearn.model_selection import cross_val_score

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate models with CV
rf_cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
print(f"\nRandom Forest CV Accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std()*2:.4f})")
```

### Sprint 2 Deliverables:
- [ ] Validation split created (80/20 stratified)
- [ ] Logistic Regression baseline trained and evaluated
- [ ] Random Forest baseline trained and evaluated
- [ ] Feature importance analysis completed
- [ ] Cross-validation scores calculated

---

## Sprint 3: Advanced Models & Temporal Slicing Experiment

### Goal: Try LightGBM, experiment with temporal slicing approach

### Step 3.1: LightGBM Model

```python
import lightgbm as lgb

# LightGBM parameters
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'is_unbalance': True,  # Handle class imbalance
    'random_state': 42
}

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train with early stopping
model_lgb = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Evaluate
y_pred_lgb_prob = model_lgb.predict(X_val)
y_pred_lgb = (y_pred_lgb_prob > 0.5).astype(int)
print(f"\nLightGBM Accuracy: {accuracy_score(y_val, y_pred_lgb):.4f}")
print(classification_report(y_val, y_pred_lgb))
```

### Step 3.2: Temporal Slicing Experiment

```python
def create_sliced_training_data(train_df, churn_times, n_slices_per_churner=3):
    """
    Create training examples with temporal slicing.

    For churned users: create multiple examples at different points before churn
    For non-churned users: create examples at the end of observation window
    """
    examples = []

    all_users = train_df['userId'].unique()
    churned_set = set(churn_times.index)
    observation_end = train_df['time'].max()

    for user_id in all_users:
        user_data = train_df[train_df['userId'] == user_id]

        if user_id in churned_set:
            # Churned user: create multiple sliced examples
            churn_time = churn_times[user_id]

            for _ in range(n_slices_per_churner):
                # Random cutoff between 1-10 days before churn
                days_before = np.random.randint(1, 11)
                cutoff = churn_time - timedelta(days=days_before)

                # Only create example if user has activity before cutoff
                user_before = user_data[user_data['time'] < cutoff]
                if len(user_before) > 10:  # Minimum activity threshold
                    example = create_single_user_features(user_before, cutoff)
                    example['churn'] = 1
                    example['userId'] = user_id
                    examples.append(example)
        else:
            # Non-churned user: single example at observation end
            example = create_single_user_features(user_data, observation_end)
            example['churn'] = 0
            example['userId'] = user_id
            examples.append(example)

    return pd.DataFrame(examples)

def create_single_user_features(user_data, observation_end):
    """Create features for a single user up to observation_end."""
    f = {}

    # Same feature engineering as before, but respecting the cutoff
    f['total_events'] = len(user_data)
    f['total_songs'] = (user_data['page'] == 'NextSong').sum()
    f['total_sessions'] = user_data['sessionId'].nunique()

    # ... (same features as the Sprint 1 function)

    # Recency feature is now meaningful
    f['days_since_last_activity'] = (observation_end - user_data['time'].max()).days

    return f
```

### Step 3.3: Compare Sliced vs Non-Sliced

```python
# Create both datasets
print("Creating non-sliced dataset...")
train_simple = pd.read_parquet('train_features.parquet')  # Already created

print("Creating sliced dataset...")
train_sliced = create_sliced_training_data(train, churn_times, n_slices_per_churner=3)

print(f"Non-sliced: {len(train_simple)} examples")
print(f"Sliced: {len(train_sliced)} examples")

# Train same model on both
def evaluate_dataset(df, name):
    X = df.drop(['userId', 'churn'], axis=1)
    y = df['churn']

    # Handle categorical
    if 'gender' in X.columns and X['gender'].dtype == 'object':
        X['gender'] = LabelEncoder().fit_transform(X['gender'].fillna('Unknown'))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)

    accuracy = accuracy_score(y_val, rf.predict(X_val))
    print(f"{name} - Validation Accuracy: {accuracy:.4f}")
    return accuracy

acc_simple = evaluate_dataset(train_simple, "Non-sliced")
acc_sliced = evaluate_dataset(train_sliced, "Sliced")
```

### Step 3.4: Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV

# Define search space for LightGBM
param_dist = {
    'num_leaves': [15, 31, 63, 127],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, -1],
    'min_child_samples': [10, 20, 50],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

lgb_clf = lgb.LGBMClassifier(
    objective='binary',
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

# Random search with cross-validation
search = RandomizedSearchCV(
    lgb_clf,
    param_dist,
    n_iter=30,  # Try 30 random combinations
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
search.fit(X, y)

print(f"Best parameters: {search.best_params_}")
print(f"Best CV accuracy: {search.best_score_:.4f}")
```

### Sprint 3 Deliverables:
- [ ] LightGBM model trained and evaluated
- [ ] Temporal slicing implemented and compared to simple approach
- [ ] Hyperparameter tuning completed
- [ ] Best model selected

---

## Sprint 4: Final Model, Submission & Report Start

### Goal: Generate predictions, create submission, start report

### Step 4.1: Train Final Model on Full Training Data

```python
# Use best model from the Sprint 3 experiments
# Example: LightGBM with tuned parameters

best_params = search.best_params_  # From the Sprint 3

final_model = lgb.LGBMClassifier(
    **best_params,
    objective='binary',
    class_weight='balanced',
    random_state=42
)

# Train on FULL training set
X_full = train_df.drop(['userId', 'churn'], axis=1)
y_full = train_df['churn']

# Encode gender
X_full['gender'] = le_gender.transform(X_full['gender'])

final_model.fit(X_full, y_full)
print("Final model trained on full dataset")
```

### Step 4.2: Generate Test Predictions

```python
# Load and prepare test features
test_df = pd.read_parquet('test_features.parquet')
X_test = test_df.drop(['userId'], axis=1)

# Encode gender (handle unseen categories)
X_test['gender'] = X_test['gender'].apply(
    lambda x: le_gender.transform([x])[0] if x in le_gender.classes_ else -1
)

# Predict
test_predictions = final_model.predict(X_test)
test_probabilities = final_model.predict_proba(X_test)[:, 1]

# Check prediction distribution
print(f"Predicted churners: {test_predictions.sum()} ({test_predictions.mean():.2%})")
```

### Step 4.3: Create Submission File

```python
# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['userId'].astype(int),
    'target': test_predictions
})

# Verify format matches example
print(submission.head())
print(f"Submission shape: {submission.shape}")

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
```

### Step 4.4: Analyze Results for Report

```python
# Feature importance for report
importance_df = pd.DataFrame({
    'feature': X_full.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== TOP FEATURES FOR REPORT ===")
print(importance_df.head(15))

# Save feature importance plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'].head(15), importance_df['importance'].head(15))
plt.xlabel('Importance')
plt.title('Top 15 Features for Churn Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
```

### Sprint 4 Deliverables:
- [ ] Final model trained on full training data
- [ ] Test predictions generated
- [ ] Submission file created and verified
- [ ] Feature importance analysis for report
- [ ] Report outline started

---

## Report Outline (4 Pages)

### Page 1: Introduction & Data Exploration
- Problem description: predict churn from streaming service logs
- Data overview: 17.5M events, 19K users, 22% churn rate
- Key finding: downgrade events strongly associated with churn (73% vs 61%)
- Data quality: minimal issues, structural missing values explained

### Page 2: Feature Engineering
- Approach: aggregate event logs to user-level features
- Feature categories: engagement, behavioral ratios, temporal, subscription
- Top predictive features (from importance analysis)
- Temporal slicing experiment: sliced vs non-sliced comparison

### Page 3: Modeling Approach
- Models tested: Logistic Regression, Random Forest, LightGBM
- Validation strategy: stratified 5-fold CV
- Hyperparameter tuning approach
- Model comparison table with accuracy scores

### Page 4: Results & Conclusions
- Final model performance
- Key insights:
  - Downgrade events are strongest predictor
  - Activity trend matters (declining = churn risk)
  - Thumbs down ratio indicates dissatisfaction
- Limitations and future work
- What would you do with more time?

---

## File Structure

```
final_project/
├── train.parquet              # Original training data
├── test.parquet               # Original test data
├── explore_data.ipynb         # EDA notebook (existing)
├── churn_predictor.ipynb      # Main modeling notebook (to create)
├── train_features.parquet     # Engineered training features
├── test_features.parquet      # Engineered test features
├── submission.csv             # Final submission
├── feature_importance.png     # Visualization for report
├── plan/
│   └── stepOne.md             # This plan file
└── report.pdf                 # Final report (4 pages)
```
