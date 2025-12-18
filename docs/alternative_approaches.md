# Alternative Approaches for Churn Prediction

Now that we've confirmed temporal leakage was the issue (50% → 55% by fixing it), here are approaches that properly handle the time-to-event nature of churn.

---

## 1. Survival Analysis (Cox Proportional Hazards)

### What it is
Instead of predicting "will they churn? (yes/no)", model "what is their hazard rate of churning at any given time?"

### Why it fits this problem
- **Handles censoring**: Non-churners aren't "non-events" - they just haven't churned YET
- **Time-aware**: Explicitly models when events happen, not just if
- **No leakage by design**: Features are computed at time t to predict hazard after t

### How it works
```
h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ...)

- h(t|X): hazard of churning at time t given features X
- h₀(t): baseline hazard (same for everyone)
- exp(...): how features multiply the baseline risk
```

### Implementation sketch
```python
from lifelines import CoxPHFitter

# Data format: one row per user
# - duration: days until churn (or end of observation)
# - event: 1 if churned, 0 if censored (still active)
# - features: computed at start of observation

cph = CoxPHFitter()
cph.fit(df, duration_col='duration', event_col='churned')

# Predict survival probability at Sprint 30
cph.predict_survival_function(X_test, times=[30])
```

### Prediction for submission
- Predict survival probability at end of window
- Convert to binary: if P(survive) < threshold → predict churn

### Difficulty: Medium
### Expected gain: Moderate (proper statistical framework)

---

## 2. Fixed-Window Feature Engineering (What we just tested)

### What it is
Use ONLY data from a fixed early period (e.g., first 7 days) for ALL users, regardless of when they churned.

### Why it works
- **Equal observation time**: Everyone gets same 7-day window
- **No future leakage**: Can't encode "how long they stayed"
- **Mimics production**: In real world, you'd predict from early behavior

### Variations to try
| Window           | Prediction target              |
| ---------------- | ------------------------------ |
| First 3 days     | Churn in days 4-50             |
| First 7 days     | Churn in days 8-50             |
| First 14 days    | Churn in days 15-50            |
| Days 1-7 vs 8-14 | Behavior change predicts churn |

### Better feature ideas for fixed window
```python
# Engagement trajectory within window
day_1_3_events = events in days 1-3
day_4_7_events = events in days 4-7
early_late_ratio = day_4_7_events / (day_1_3_events + 1)  # declining = bad

# Session patterns
avg_gap_between_sessions  # increasing gaps = disengaging
longest_gap_in_window     # long absence = warning sign
weekend_vs_weekday_ratio  # usage pattern

# Frustration signals
errors_per_session
ads_per_song
thumbs_down_per_session
```

### Difficulty: Easy
### Expected gain: Already showed improvement (50% → 55%)

---

## 3. Sequence Modeling (RNN/LSTM/Transformer)

### What it is
Instead of aggregating events into features, feed the raw sequence of events into a neural network that learns temporal patterns.

### Why it might work
- **Captures order**: "thumbs down → thumbs down → logout" is different from "thumbs up → thumbs up → logout"
- **Learns patterns**: Network discovers what sequences precede churn
- **No manual features**: Let the model find signal

### Data format
```
User 1: [NextSong, NextSong, ThumbsUp, NextSong, Ad, NextSong, ThumbsDown, Logout]
User 2: [Home, NextSong, Error, Help, Settings, Logout]
```

### Architecture options
| Model       | Pros                          | Cons                    |
| ----------- | ----------------------------- | ----------------------- |
| LSTM        | Good at sequences, proven     | Slow to train           |
| GRU         | Faster than LSTM              | Slightly less powerful  |
| Transformer | State-of-art, attention       | Needs more data         |
| 1D CNN      | Fast, captures local patterns | Less temporal awareness |

### Implementation sketch
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Encode page types as integers
page_vocab = {'NextSong': 1, 'ThumbsUp': 2, 'ThumbsDown': 3, ...}

# Pad sequences to fixed length
X = pad_sequences(sequences, maxlen=500)

model = Sequential([
    Embedding(input_dim=len(page_vocab)+1, output_dim=32),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
```

### Difficulty: Hard
### Expected gain: Potentially high if patterns exist in sequences

---

## 4. Session-Level Features + Aggregation

### What it is
Instead of user-level aggregation, first create features per SESSION, then aggregate sessions.

### Why it might work
- **Sessions are natural units**: A session = one "visit" to the app
- **Captures session quality**: Bad sessions might predict churn
- **Temporal patterns**: How do sessions evolve over time?

### Session-level features
```python
per_session = df.groupby(['userId', 'sessionId']).agg(
    session_length=('time', lambda x: (x.max() - x.min()).seconds / 60),
    songs_played=('song', lambda x: x.notna().sum()),
    thumbs_up=('page', lambda x: (x == 'Thumbs Up').sum()),
    thumbs_down=('page', lambda x: (x == 'Thumbs Down').sum()),
    errors=('page', lambda x: (x == 'Error').sum()),
    ended_with_logout=('page', lambda x: x.iloc[-1] == 'Logout'),
)
```

### User-level aggregation of sessions
```python
per_user = per_session.groupby('userId').agg(
    num_sessions=('session_length', 'count'),
    avg_session_length=('session_length', 'mean'),
    std_session_length=('session_length', 'std'),  # consistency
    avg_songs_per_session=('songs_played', 'mean'),
    pct_sessions_with_thumbs_down=('thumbs_down', lambda x: (x > 0).mean()),
    pct_sessions_with_errors=('errors', lambda x: (x > 0).mean()),

    # Trajectory features
    first_3_sessions_avg_length=...,
    last_3_sessions_avg_length=...,
    session_length_trend=...,  # regression slope
)
```

### Difficulty: Medium
### Expected gain: Moderate (better feature granularity)

---

## 5. Change-Point / Regime Detection

### What it is
Identify when a user's behavior CHANGED. The moment of change might predict churn better than absolute levels.

### Why it might work
- **Churn is a process**: Users don't suddenly churn; they disengage gradually
- **Change is signal**: A user going from 10 songs/day to 2 songs/day is concerning
- **Absolute values mislead**: Low-volume users who are consistent might not churn

### Features to compute
```python
# Split user timeline into early/late periods
early_period = first 40% of their events
late_period = last 40% of their events

# Compute change metrics
songs_per_day_change = late_songs_per_day - early_songs_per_day
session_frequency_change = late_sessions_per_week - early_sessions_per_week
thumbs_ratio_change = late_thumbs_up_ratio - early_thumbs_up_ratio

# Relative change (handles different baseline levels)
songs_per_day_pct_change = (late - early) / (early + 1)
```

### Statistical change detection
```python
from ruptures import Pelt

# Detect changepoints in daily event counts
signal = user_daily_events.values
algo = Pelt(model="rbf").fit(signal)
changepoints = algo.predict(pen=10)
```

### Difficulty: Medium
### Expected gain: Potentially high (captures disengagement process)

---

## 6. Cohort-Based Approach

### What it is
Group users by when they started, compare behavior at the SAME tenure point.

### Why it might work
- **Controls for time**: Day-7 behavior is comparable across users
- **Reveals true patterns**: What distinguishes churners at day 7?
- **Handles seasonality**: Different cohorts might behave differently

### Implementation
```python
# Compute features at specific tenure milestones
for user in users:
    features_day_7 = compute_features(user_events[user_events['tenure_days'] <= 7])
    features_day_14 = compute_features(user_events[user_events['tenure_days'] <= 14])

# Compare churners vs non-churners at day 7
churner_day7_features = features_day_7[churners]
non_churner_day7_features = features_day_7[non_churners]
```

### Difficulty: Easy-Medium
### Expected gain: Moderate (cleaner comparison)

---

## 7. Propensity Score / Matching

### What it is
Match each churner with a similar non-churner, then find what differentiates them.

### Why it might work
- **Reduces confounding**: Compare apples to apples
- **Highlights causal factors**: What's different between matched pairs?
- **Handles selection bias**: Similar users, different outcomes

### Implementation
```python
from sklearn.neighbors import NearestNeighbors

# Match on "baseline" features (things that shouldn't predict churn)
baseline_features = ['days_since_registration', 'is_paid', 'gender', ...]

# For each churner, find most similar non-churner
nn = NearestNeighbors(n_neighbors=1)
nn.fit(non_churner_baseline_features)
matches = nn.kneighbors(churner_baseline_features)

# Now compare BEHAVIORAL features between matched pairs
```

### Difficulty: Medium
### Expected gain: Insight into causal factors

---

## 8. Time-Series Classification

### What it is
Treat each user's daily engagement as a time series, use specialized time-series methods.

### Why it might work
- **Shape matters**: The "shape" of engagement over time might be predictive
- **Specialized algorithms**: Methods like ROCKET, DTW designed for this
- **Pattern matching**: Find users whose curves look like past churners

### Methods
| Method                     | Description                                   |
| -------------------------- | --------------------------------------------- |
| DTW (Dynamic Time Warping) | Similarity measure for time series            |
| ROCKET                     | Random convolutional kernels, very fast       |
| TSFresh                    | Automatic feature extraction from time series |
| Catch22                    | 22 canonical time-series features             |

### Implementation
```python
from sktime.classification.kernel_based import RocketClassifier

# Format: each user is a time series of daily event counts
X = np.array([user_daily_events for user in users])  # shape: (n_users, n_days)
y = churn_labels

rocket = RocketClassifier()
rocket.fit(X_train, y_train)
```

### Difficulty: Medium-Hard
### Expected gain: Potentially high for pattern-based churn

---

## 9. Multi-Task Learning

### What it is
Predict multiple related outcomes simultaneously: churn, downgrade, reduced engagement.

### Why it might work
- **Shared representations**: Related tasks help each other
- **More signal**: Downgrade events are more common than churn
- **Regularization**: Prevents overfitting to noise

### Tasks to predict
1. Will user churn? (main task)
2. Will user downgrade?
3. Will user's engagement drop >50%?
4. Will user have a long absence (>7 days)?

### Difficulty: Hard
### Expected gain: Moderate (if tasks are related)

---

## 10. Ensemble of Time Windows

### What it is
Train separate models on different time windows, combine predictions.

### Why it might work
- **Different windows capture different signals**
- **Reduces variance**: Ensemble is more robust
- **Covers early and late churners**

### Implementation
```python
models = {
    'day_1_7': train_model(features_day_1_7),
    'day_1_14': train_model(features_day_1_14),
    'day_8_14': train_model(features_day_8_14),  # second week only
}

# Ensemble prediction
final_pred = np.mean([m.predict_proba(X)[:, 1] for m in models.values()], axis=0)
```

### Difficulty: Easy
### Expected gain: Moderate (variance reduction)

---

## Recommended Priority

| Priority | Approach                      | Why                                  |
| -------- | ----------------------------- | ------------------------------------ |
| 1        | **Fixed-window improvements** | Already working, easy to iterate     |
| 2        | **Session-level features**    | Natural granularity, moderate effort |
| 3        | **Change-point detection**    | Captures disengagement process       |
| 4        | **Survival analysis**         | Proper statistical framework         |
| 5        | **Time-series (ROCKET)**      | If above don't work, try this        |
| 6        | **Sequence modeling**         | Last resort, needs more data/compute |
