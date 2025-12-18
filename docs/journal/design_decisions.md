# Design Decisions - Churn Prediction Project

This document captures the key design decisions made for the churn prediction model, with supporting data and reasoning from our exploratory data analysis.

---

## 1. Churn Definition

**Decision**: Define churn as visiting the `Cancellation Confirmation` page.

**Data Evidence** (cell-23, cell-27):
```
Total users in training set: 19,140
Churned users: 4,271
Non-churned users: 14,869
Churn rate: 22.31%

Users with 'Cancel' page visit: 4,271
Users with 'Cancellation Confirmation': 4,271
Perfect match: True
```

**Reasoning**:
- Clear, unambiguous event marking account cancellation
- Cancel and Cancellation Confirmation have a perfect 1:1 relationship
- The `Cancel` page always leads to confirmation (median 33 seconds between them)

---

## 2. Cancel Page Feature Handling

**Decision**: Keep Cancel page visits that occur >12 hours before Cancellation Confirmation.

**Data Evidence** (cell-27):
```
Time between Cancel and Confirmation:
  Median: 33.0 seconds
  Mean: 89.8 seconds
  Max: 3,720.0 seconds (62.0 minutes)
```

**Reasoning**:
- The final Cancel click is essentially the same event as churning - including it would be data leakage
- However, if a user visited Cancel weeks ago but didn't confirm, that's a valuable signal ("considered leaving")
- 12-hour threshold is conservative: max observed gap is 62 minutes, so 12 hours (720 minutes) provides a 11x safety margin
- Earlier Cancel visits indicate dissatisfaction without leaking the outcome

**Alternative considered**: Exclude all Cancel visits
- Rejected because early Cancel visits (days/weeks before churn) capture "considering cancellation" behavior

---

## 3. Temporal Slicing Strategy

**Decision**: Create 3 training examples per churned user at random points 1-10 days before churn.

**Data Evidence** (cell-23):
```
Churned users: 4,271
Non-churned users: 14,869
Churn rate: 22.31%
```

**Reasoning**:

| Approach | Training Examples | Pros | Cons |
|----------|-------------------|------|------|
| No slicing | 19,140 | Simple | Limited temporal variation |
| 3 slices/churner | ~27,000 | Balanced dataset, captures pre-churn patterns | Moderate complexity |
| 5+ slices/churner | ~35,000+ | More data | Risk of overfitting to specific users |

- With 3 slices: ~12,813 churned examples vs ~14,869 non-churned = more balanced
- 1-10 day window captures behavior at different stages before churn
- Minimum 10 events threshold ensures meaningful feature extraction
- Mirrors prediction task: "given behavior up to time T, will user churn within 10 days?"

---

## 4. Class Imbalance Handling

**Decision**: Use `class_weight='balanced'` in sklearn models and `is_unbalance=True` in LightGBM.

**Data Evidence** (cell-23):
```
Total users: 19,140
Churned: 4,271 (22.31%)
Non-churned: 14,869 (77.69%)
Class ratio: ~1:3.5
```

**Reasoning**:
- 22% churn rate is moderate imbalance (not severe)
- Without weighting: model might predict "no churn" too often to minimize overall error
- Balanced weights penalize misclassifying churners more heavily
- SMOTE not needed: 22% minority is substantial enough for learning
- Computationally efficient, avoids creating synthetic data or losing information

---

## 5. Location Features

**Decision**: Extract 2-letter state code from location string (reduces 875 → ~50 categories).

**Data Evidence** (cell-18):
```
Unique users (train): 19,140
```
Sample location format: `"Dallas-Fort Worth-Arlington, TX"`

**Reasoning**:
- 875 unique locations for 19,140 users = ~22 users per location on average
- Many locations have very few users → sparse features and overfitting risk
- State-level aggregation:
  - Reduces dimensionality 17.5x
  - Captures regional patterns (if any exist)
  - Sufficient data per category for reliable patterns
  - Handles unseen cities in test set gracefully

**Alternative considered**: Skip location entirely
- Rejected because regional patterns might exist (e.g., competition from local services)
- State extraction is low-cost to implement

---

## 6. Key Predictive Features: Downgrade/Upgrade Events

**Decision**: Include `has_downgrade` and `has_upgrade` as binary features, plus raw counts.

**Data Evidence** (cell-29):
```
Churned users behavior:
  Had Downgrade event: 3,113 (72.9%)
  Had Upgrade event: 2,981 (69.8%)

Non-churned users behavior:
  Had Downgrade event: 9,073 (61.0%)
  Had Upgrade event: 10,217 (68.7%)
```

**Reasoning**:
- Downgrade is a strong discriminator: 72.9% of churners vs 61.0% of non-churners
- 11.9 percentage point difference (19.5% relative increase) is significant
- Typical churn journey: Upgrade (try paid) → Downgrade (dissatisfied) → Cancel
- Upgrade is less discriminative (69.8% vs 68.7%) but still worth including

---

## 7. Subscription Level Features

**Decision**: Track `is_paid` (current level) and `level_changes` (number of transitions).

**Data Evidence** (cell-31):
```
Subscription level at churn:
  paid: 2,879 (67.4%)
  free: 1,392 (32.6%)
```

**Reasoning**:
- 67% of churners were still on paid when they cancelled
- Counterintuitive insight: paid users who are unhappy cancel; free users might stay indefinitely
- `level_changes` captures volatility: users who switched multiple times may be indecisive

---

## 8. Critical Insight: Churned Users Are MORE Active

**Decision**: Focus on behavioral dissatisfaction signals rather than disengagement metrics.

**Data Evidence** (cell-25):
```
Events per user:
  Churned users:     mean = 917.8, median = 557.0
  Non-churned users: mean = 913.3, median = 531.0

Churned users have 1.0x more events on average!
```

**Reasoning**:
- Churn is driven by **dissatisfaction**, not disengagement
- Feature engineering prioritizes:
  - Page interactions (Downgrade, Thumbs Down, Error) - dissatisfaction signals
  - Behavioral ratios (thumbs up/down per song)
  - Activity trends (declining activity may precede churn)
- Demographics deprioritized: gender split is balanced (53% M / 47% F among churners)

---

## 9. Missing Value Handling

**Decision**: No imputation for song/artist/length; use "Unknown" for missing demographics.

**Data Evidence** (cell-33):
```
Missing song values by page type:
  NextSong pages with missing song: 0
  Other pages with missing song: 3,208,203 / 3,208,203 (100.0%)

Test set missing demographics by auth status:
  Guest: 3,363
  Logged In: 0
  Logged Out: 650,318
```

**Reasoning**:
- Song/artist/length are NULL on non-music pages by design (structural, not data quality issue)
- We aggregate to user level, so these NULLs don't affect features
- Demographics missing in test = logged-out/guest sessions
- "Unknown" category preserves this information as a signal

---

## 10. Song Length Outlier Handling

**Decision**: Cap song length at 1,200 seconds (20 minutes).

**Data Evidence** (cell-12):
```
Song length statistics:
  min: 0.522 seconds
  max: 3,024.67 seconds (50+ minutes)
  mean: 248.7 seconds (~4 min)
  75th percentile: 276.9 seconds
```

**Reasoning**:
- Typical songs are 3-5 minutes
- Values >20 minutes are likely podcasts, audiobooks, or data errors
- 1,200 seconds = ~4.3x the 75th percentile, preserving legitimate long songs
- Capping prevents outliers from skewing averages while preserving event counts

---

## 11. Model Selection

**Decision**: Use LightGBM as primary model with Logistic Regression and Random Forest as baselines.

| Model | Why Included |
|-------|--------------|
| Logistic Regression | Interpretable baseline, fast training, feature importance via coefficients |
| Random Forest | Handles non-linear relationships, robust to outliers, no scaling needed |
| LightGBM | State-of-art gradient boosting, handles imbalance natively, fast with early stopping |

**Why not neural networks**:
- Tabular data with ~30 features
- ~19K training examples - not enough for deep learning advantage
- Tree-based models typically outperform NNs on structured tabular data
- Interpretability requirements for report

---

## 12. Validation Strategy

**Decision**: Use stratified 80/20 split for validation, 5-fold stratified CV for model selection.

**Data Evidence** (cell-35):
```
Train users: 19,140
Test users: 2,904
Overlap: 0 users (0.0% of test)
```

**Reasoning**:
- Stratification preserves 22% churn rate in all folds
- 80/20 provides ~3,800 validation examples (statistically significant)
- 5-fold CV reduces variance in performance estimates
- No temporal split needed because train/test users don't overlap
- Model must generalize to unseen users → features must capture behavior patterns, not user identity

---

## 13. Prediction Threshold

**Decision**: Use 0.5 probability threshold for binary classification.

**Reasoning**:
- Standard threshold for balanced accuracy
- With `class_weight='balanced'`, model already adjusts for class imbalance
- No business context suggesting asymmetric costs (false positive vs false negative)
- Can be tuned post-hoc if evaluation reveals issues

---

## 14. Feature Exclusions

**Decision**: Exclude `userId`, `sessionId`, and raw timestamps from model features.

**Reasoning**:
- `userId`: No overlap between train/test (cell-35) - would be useless or harmful
- `sessionId`: Sequential identifiers with no predictive meaning
- Raw `ts`/`time`: Absolute timestamps don't generalize

**Temporal features kept instead**:
- `days_active`: Duration of user activity
- `days_since_registration`: Account age
- `days_since_last_activity`: Recency (meaningful with temporal slicing)
- `activity_trend`: First half vs second half comparison

---

## Summary Table

| Decision | Choice | Key Data Point |
|----------|--------|----------------|
| Churn definition | Cancellation Confirmation page | 4,271 users, 22.31% rate |
| Cancel page handling | Keep if >12h before churn | Max observed gap: 62 min |
| Temporal slicing | 3 slices/churner, 1-10 days before | Balances 4,271 vs 14,869 |
| Class imbalance | class_weight='balanced' | 3.5:1 ratio (moderate) |
| Location encoding | State extraction | 875 → ~50 categories |
| Key features | Downgrade/Upgrade events | 72.9% vs 61.0% downgrade rate |
| Subscription tracking | is_paid + level_changes | 67% churners were paid |
| Critical insight | Churners are MORE active | Dissatisfaction, not disengagement |
| Missing values | Structural - no imputation | 100% missing = non-song pages |
| Song length | Cap at 1,200 sec | Max 3,024 sec, p75 = 277 sec |
| Models | LightGBM + LR + RF baselines | Tabular data best practices |
| Validation | Stratified 80/20 + 5-fold CV | 0% user overlap train/test |
| Threshold | 0.5 | Standard with balanced weights |
