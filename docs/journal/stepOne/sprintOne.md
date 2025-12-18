# Sprint 1: Data Pipeline & Feature Engineering

## Summary
Transformed 17.5M event logs into 19,140 user-level feature vectors (43 features each) for churn prediction.

## Key Decision: Vectorized vs Loop-Based Feature Engineering

### Initial Approach (Rejected)
The plan specified a row-by-row loop approach:
```python
for user_id in tqdm(user_ids):
    user_data = df[df['userId'] == user_id]
    # compute features for this user
```

**Problem**: Estimated runtime ~2 hours (2.7 it/s for 19,140 users). Each iteration filters the entire 17.5M row DataFrame.

### Optimized Approach (Implemented)
Replaced with vectorized `groupby` operations:
```python
basic_agg = df.groupby('userId').agg(
    total_events=('page', 'count'),
    total_sessions=('sessionId', 'nunique'),
    ...
)
page_counts = pd.crosstab(page_df['userId'], page_df['page'])
```

**Result**: Completed in ~35 seconds (100x speedup). Same output, compiled C/Cython under the hood.

### Why This Works
1. `groupby().agg()` - single pass over data, compiled operations
2. `pd.crosstab()` - efficient pivot for page counts
3. Vectorized joins instead of per-user dictionary building
4. Derived features computed on entire DataFrame at once

## Feature Categories Implemented

| Category          | Features                                                             | Rationale                     |
| ----------------- | -------------------------------------------------------------------- | ----------------------------- |
| Engagement        | total_events, total_songs, total_sessions, session_length stats      | Activity volume indicators    |
| Page counts       | 13 important pages (NextSong, Thumbs Up/Down, Downgrade, etc.)       | Behavioral signals            |
| Behavioral ratios | thumbs_up/down_ratio, playlist_add_ratio, error_rate, ad_ratio       | Normalized engagement quality |
| Temporal          | days_active, days_since_registration, events_per_day, activity_trend | Usage patterns over time      |
| Subscription      | is_paid, level_changes, has_downgrade, has_upgrade, paid_ratio       | Subscription lifecycle        |
| Content           | unique_songs/artists, song_repeat_ratio, listen_time                 | Content engagement depth      |
| Demographics      | gender, state                                                        | User attributes               |

## Data Leakage Prevention

### Cancel Page Handling
Per the plan's design decision, Cancel page visits within 12 hours of Cancellation Confirmation are excluded:
```python
cancel_df['is_safe'] = (
    cancel_df['churn_time'].isna() |
    (cancel_df['time'] < cancel_df['churn_time'] - pd.Timedelta(hours=12))
)
```
This captures "considered cancelling" behavior without leaking the final cancellation flow.

## Bug Fix: Mixed Types in Gender Column

### Issue
Test set has logged-out/guest sessions with missing demographics. After `fillna('Unknown')`, the column still had mixed types (strings + original NaN-derived integers), causing Arrow serialization to fail when saving to parquet.

### Solution
Explicit string cast:
```python
features['gender'] = features['gender'].fillna('Unknown').astype(str)
```

## Output Files

| File                   | Rows   | Columns                           | Size  |
| ---------------------- | ------ | --------------------------------- | ----- |
| train_features.parquet | 19,140 | 45 (43 features + userId + churn) | 2.5MB |
| test_features.parquet  | 2,904  | 44 (43 features + userId)         | 492KB |

## Verification
- No missing values in either dataset
- Feature columns match between train and test (excluding churn label)
- Churn rate preserved: 22.31% (4,271 / 19,140)
