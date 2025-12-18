"""Generate figures for the LaTeX report."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']

# Load data
train_features = pd.read_parquet('train_features.parquet')
test_features = pd.read_parquet('test_features.parquet')

n_train_users = len(train_features)
n_churned = train_features['churn'].sum()
churn_rate = n_churned / n_train_users * 100

print(f"Training users: {n_train_users:,}")
print(f"Churned users:  {n_churned:,} ({churn_rate:.2f}%)")

# =============================================================================
# Figure 1: Churn Distribution
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

churn_counts = train_features['churn'].value_counts().sort_index()
bars = ax.bar(['Active Users', 'Churned Users'],
              [churn_counts[0], churn_counts[1]],
              color=[colors[0], colors[1]],
              edgecolor='white', linewidth=2)

for bar, count in zip(bars, [churn_counts[0], churn_counts[1]]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{count:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.text(1, churn_counts[1]/2, f'{churn_rate:.1f}%', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')
ax.text(0, churn_counts[0]/2, f'{100-churn_rate:.1f}%', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')

ax.set_ylabel('Number of Users', fontsize=12)
ax.set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
ax.set_ylim(0, churn_counts[0] * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/churn_distribution.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/churn_distribution.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: churn_distribution")

# =============================================================================
# Figure 2: Feature Importance
# =============================================================================
feature_cols = [c for c in train_features.columns if c not in ['userId', 'churn']]
X = train_features[feature_cols].copy()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_cols].fillna(0)
y = train_features['churn'].astype(int)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)

importance_df = pd.DataFrame({
    'feature': numeric_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(importance_df['feature'], importance_df['importance'], color=colors[2])

for i, (feat, imp) in enumerate(zip(importance_df['feature'], importance_df['importance'])):
    if feat == 'days_active':
        bars[i].set_color(colors[1])

ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Top 15 Features (Initial Model)\nNote: days_active (red) is the dominant feature',
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/feature_importance.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/feature_importance.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: feature_importance")

# =============================================================================
# Figure 3: Leakage Pattern (days_active distribution + CV vs Test)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: days_active distribution
ax1 = axes[0]
churned = train_features[train_features['churn'] == 1]['days_active']
active = train_features[train_features['churn'] == 0]['days_active']

ax1.hist(active, bins=30, alpha=0.7, label=f'Active (mean={active.mean():.1f})', color=colors[0])
ax1.hist(churned, bins=30, alpha=0.7, label=f'Churned (mean={churned.mean():.1f})', color=colors[1])
ax1.axvline(churned.mean(), color=colors[1], linestyle='--', linewidth=2)
ax1.axvline(active.mean(), color=colors[0], linestyle='--', linewidth=2)

ax1.set_xlabel('Days Active', fontsize=12)
ax1.set_ylabel('Number of Users', fontsize=12)
ax1.set_title('The Leakage Pattern\nChurners have shorter observation windows', fontsize=12, fontweight='bold')
ax1.legend()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right plot: CV vs Test accuracy
ax2 = axes[1]
approaches = ['Initial\n(leaked)', 'Fixed Window\n(7-day)', 'Enhanced\nFeatures', 'Gradient\nBoosting']
cv_scores = [88, 75, 74, 80]
test_scores = [50, 61.3, 59.2, 55.7]

x = np.arange(len(approaches))
width = 0.35

bars1 = ax2.bar(x - width/2, cv_scores, width, label='CV Accuracy', color=colors[2])
bars2 = ax2.bar(x + width/2, test_scores, width, label='Test Accuracy', color=colors[4])

for i, (cv, test) in enumerate(zip(cv_scores, test_scores)):
    gap = cv - test
    ax2.annotate(f'-{gap:.0f}pts', xy=(i, max(cv, test) + 2), ha='center', fontsize=9, color='gray')

ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('CV vs Test Performance\nHigher CV often meant worse test!', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(approaches)
ax2.legend()
ax2.set_ylim(0, 100)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/leakage_pattern.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/leakage_pattern.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: leakage_pattern")

# =============================================================================
# Figure 4: All Submissions Comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

submissions = [
    ('Initial (leaked)', 50),
    ('Fixed 7-day RF', 61.3),
    ('Fixed 7-day GB', 55.7),
    ('Enhanced features', 59.2),
    ('GDA (LDA)', 50),
    ('Ensemble top-50%', 60.1),
    ('Ensemble top-46%', 61.3),
    ('Ensemble top-30%', 62.3),
]

names, scores = zip(*submissions)
y_pos = np.arange(len(names))

bar_colors = [colors[1] if s < 55 else (colors[0] if s > 61 else colors[2]) for s in scores]

bars = ax.barh(y_pos, scores, color=bar_colors, edgecolor='white', linewidth=1)

bars[-1].set_edgecolor('black')
bars[-1].set_linewidth(3)

ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.set_xlabel('Test Accuracy (%)', fontsize=12)
ax.set_title('All Submission Attempts\nBest: 62.3% with Ensemble + Threshold Calibration',
             fontsize=13, fontweight='bold')
ax.axvline(50, color='gray', linestyle='--', alpha=0.5, label='Random guessing')
ax.set_xlim(45, 65)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, score in zip(bars, scores):
    ax.text(score + 0.3, bar.get_y() + bar.get_height()/2,
            f'{score}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figures/submissions.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/submissions.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: submissions")

# =============================================================================
# Figure 5: Correlation Heatmap
# =============================================================================
key_features = ['days_active', 'total_events', 'total_sessions', 'events_per_day',
                'thumbs_up_ratio', 'thumbs_down_ratio', 'ad_ratio', 'error_rate',
                'has_downgrade', 'is_paid', 'activity_trend', 'churn']

available_features = [f for f in key_features if f in train_features.columns]
corr_matrix = train_features[available_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8})

ax.set_title('Feature Correlations\nNote: days_active has strongest correlation with churn (-0.40)',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/correlation_heatmap.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/correlation_heatmap.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved: correlation_heatmap")

print("\nAll figures saved to figures/ directory")
