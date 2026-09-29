import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
ML_FOLDER = os.path.join(project_root, 'data', 'ml')
MODEL_FOLDER = os.path.join(project_root, 'data', 'models')
FIG_FOLDER = os.path.join(project_root, 'reports', 'figures')
os.makedirs(FIG_FOLDER, exist_ok=True)

pipeline = joblib.load(os.path.join(MODEL_FOLDER, 'ridge_pipeline.joblib'))
X_test = pd.read_csv(os.path.join(ML_FOLDER, 'X_test.csv'))
Y_test = pd.read_csv(os.path.join(ML_FOLDER, 'Y_test.csv')).squeeze('columns')

id_cols = ['player_id', 'player', 'season', 'team', 'secondary_team', 'nation']
feature_cols = [c for c in X_test.columns if c not in id_cols]

pred_log1p = pipeline.predict(X_test[feature_cols])
actual_eur = np.expm1(Y_test.values)
pred_eur = np.expm1(pred_log1p)


# PLOT I: predicted vs actual
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(actual_eur, pred_eur, alpha=0.5, s=20, edgecolors='none')
# y=x reference line
lims = [min(actual_eur.min(), pred_eur.min()),
        max(actual_eur.max(), pred_eur.max())]
ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect prediction')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Actual Market Value (€, log scale)')
ax.set_ylabel('Predicted Market Value (€, log scale)')
ax.set_title('Predicted vs Actual Market Value (Test Set 2025-26)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_FOLDER, '01_pred_vs_actual.png'), dpi=150)
plt.close()


# PLOT II: Residuals vs Predicted
residuals = actual_eur - pred_eur  # actual − predicted

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(pred_eur, residuals, alpha=0.5, s=20, edgecolors='none')
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.set_xscale('log')
ax.set_xlabel('Predicted Market Value (€, log scale)')
ax.set_ylabel('Residual (Actual − Predicted, €)')
ax.set_title('Residuals vs Predicted Value (Test Set)')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Zero error (perfect prediction)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_FOLDER, '02_residuals.png'), dpi=150)
plt.close()


# PLOT III: Coefficient bar chart (top 20 by magnitude)
ridge = pipeline.named_steps['ridge']
coefs = pd.DataFrame({
    'feature': feature_cols,
    'coef': ridge.coef_
})
coefs['abs_coef'] = coefs['coef'].abs()
top20 = coefs.sort_values('abs_coef', ascending=False).head(20)
top20 = top20.sort_values('coef')  # sort ascending for horizontal bar chart

colors = ['#d62728' if c < 0 else '#2ca02c' for c in top20['coef']]
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top20['feature'], top20['coef'], color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient (standardised features)')
ax.set_title('Top 20 Ridge Coefficients by Magnitude')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_FOLDER, '03_coefficients.png'), dpi=150)
plt.close()


# PLOT IV: relative error distribution
rel_err = np.abs((actual_eur - pred_eur) / actual_eur) * 100

fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(rel_err, bins=40, color='steelblue', edgecolor='white')
ax.axvline(np.median(rel_err), color='red', linestyle='--',
           linewidth=1.5, label=f'Median = {np.median(rel_err):.1f}%')
ax.axvline(np.mean(rel_err), color='orange', linestyle='--',
           linewidth=1.5, label=f'Mean = {np.mean(rel_err):.1f}%')
ax.set_xlabel('Absolute Relative Error (%)')
ax.set_ylabel('Number of players')
ax.set_title('Distribution of Relative Prediction Error (Test Set)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_FOLDER, '04_error_distribution.png'), dpi=150)
plt.close()

print(f"Saved 4 figures to {FIG_FOLDER}")
print("  01_pred_vs_actual.png")
print("  02_residuals.png")
print("  03_coefficients.png")
print("  04_error_distribution.png")