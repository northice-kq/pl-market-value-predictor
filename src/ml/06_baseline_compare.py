import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
ML_FOLDER = os.path.join(project_root, 'data', 'ml')
MODEL_FOLDER = os.path.join(project_root, 'data', 'models')

X_train = pd.read_csv(os.path.join(ML_FOLDER, 'X_train.csv'))
X_test  = pd.read_csv(os.path.join(ML_FOLDER, 'X_test.csv'))
Y_train = pd.read_csv(os.path.join(ML_FOLDER, 'Y_train.csv')).squeeze('columns')
Y_test  = pd.read_csv(os.path.join(ML_FOLDER, 'Y_test.csv')).squeeze('columns')
pipeline = joblib.load(os.path.join(MODEL_FOLDER, 'ridge_pipeline.joblib'))

id_cols = ['player_id', 'player', 'season', 'team', 'secondary_team', 'nation']
feature_cols = [c for c in X_train.columns if c not in id_cols]

def report(name, y_true_log1p, y_pred_log1p):# gen report
    rmse_log1p = np.sqrt(mean_squared_error(y_true_log1p, y_pred_log1p))
    mae_log1p = mean_absolute_error(y_true_log1p, y_pred_log1p)
    r2_log1p = r2_score(y_true_log1p, y_pred_log1p)

    y_true_eur = np.expm1(y_true_log1p)
    y_pred_eur = np.expm1(y_pred_log1p)
    rmse_eur = np.sqrt(mean_squared_error(y_true_eur, y_pred_eur))
    mae_eur = mean_absolute_error(y_true_eur, y_pred_eur)
    rel_err = np.abs((y_true_eur - y_pred_eur) / y_true_eur) * 100

    return {
        'model': name,
        'rmse_log1p': rmse_log1p,
        'mae_log1p': mae_log1p,
        'r2_log1p': r2_log1p,
        'rmse_eur': rmse_eur,
        'mae_eur': mae_eur,
        'median_rel_err_pct': np.median(rel_err),
    }
# baseline I: predict the training mean
pred_mean_log1p = np.full(len(Y_test), Y_train.mean())
results_mean = report('Predict mean', Y_test.values, pred_mean_log1p)

#  baseline II: predict = pre-season value (no change from pre to end)
pred_pre_log1p = X_test['log1p_pre_mv'].values
results_pre = report('Predict = pre-season value', Y_test.values, pred_pre_log1p)

#  baseline III: predict = mid-season value
pred_mid_log1p = X_test['log1p_mid_mv'].values
results_mid = report('Predict = mid-season value', Y_test.values, pred_mid_log1p)

# model
pred_ridge_log1p = pipeline.predict(X_test[feature_cols])
results_ridge = report('Ridge model', Y_test.values, pred_ridge_log1p)

df = pd.DataFrame([results_mean, results_pre, results_mid, results_ridge])

print("\n" + "=" * 90)
print("MODEL COMPARISON ON TEST SET (2025-26)")
print("=" * 90)
print(df.to_string(index=False, float_format=lambda x: f'{x:,.4f}' if abs(x) < 100 else f'{x:,.0f}'))

# compute over the strongest baseline
ridge_rmse_log1p = results_ridge['rmse_log1p']
mid_rmse_log1p   = results_mid['rmse_log1p']
improvement_pct = (mid_rmse_log1p - ridge_rmse_log1p) / mid_rmse_log1p * 100

ridge_mae_eur = results_ridge['mae_eur']
mid_mae_eur   = results_mid['mae_eur']
mae_improvement_pct = (mid_mae_eur - ridge_mae_eur) / mid_mae_eur * 100

print(f"\nRidge vs strongest baseline (mid-season value):")
print(f"  RMSE (log1p) improvement: {improvement_pct:+.2f}%")
print(f"  MAE  (€)   improvement: {mae_improvement_pct:+.2f}%")

df.to_csv(os.path.join(MODEL_FOLDER, 'baseline_comparison.csv'), index=False)
print(f"\nSaved comparison to {MODEL_FOLDER}/baseline_comparison.csv")