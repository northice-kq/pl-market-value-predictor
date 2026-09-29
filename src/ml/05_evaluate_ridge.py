import pandas as pd
import numpy as np
import os
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
ML_FOLDER = os.path.join(project_root, 'data', 'ml')
MODEL_FOLDER = os.path.join(project_root, 'data', 'models')

pipeline = joblib.load(os.path.join(MODEL_FOLDER, 'ridge_pipeline.joblib'))

X_test = pd.read_csv(os.path.join(ML_FOLDER, 'X_test.csv'))
Y_test = pd.read_csv(os.path.join(ML_FOLDER, 'Y_test.csv')).squeeze('columns')

id_cols = ['player_id', 'player', 'season', 'team', 'secondary_team', 'nation']
feature_cols = [c for c in X_test.columns if c not in id_cols]

# predict
preds_log = pipeline.predict(X_test[feature_cols])


results = X_test[['player_id', 'player', 'team', 'season', 'age']].copy()
results['actual_log']    = Y_test.values
results['pred_log']      = preds_log
results['actual_mv']     = np.expm1(results['actual_log'])
results['pred_mv']       = np.expm1(results['pred_log'])
results['error_eur']     = results['actual_mv'] - results['pred_mv']
results['abs_error_eur'] = results['error_eur'].abs()
results['error_log']     = results['actual_log'] - results['pred_log']
results['abs_error_log'] = results['error_log'].abs()
results['rel_error_pct'] = (results['error_eur'] / results['actual_mv']) * 100
results['abs_rel_error_pct'] = results['rel_error_pct'].abs()

# sort by abs err
worst = results.sort_values('abs_error_eur', ascending=False).head(15)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("\n15 WORST TEST PREDICTIONS (by absolute euro error)\n")
print(worst[[
    'player', 'team', 'season', 'age',
    'actual_mv', 'pred_mv', 'error_eur', 'abs_error_eur', 'rel_error_pct'
]].to_string(index=False, float_format=lambda x: f'{x:,.1f}'))

print("\nError breakdown")
print(f"Total test rows:            {len(results)}")
print(f"Players with euro error > €5M:   {(results['abs_error_eur'] >  5_000_000).sum()}")
print(f"Players with euro error > €10M:  {(results['abs_error_eur'] > 10_000_000).sum()}")
print(f"Players with euro error > €20M:  {(results['abs_error_eur'] > 20_000_000).sum()}")
print(f"Players with rel. error > 20%:   {(results['abs_rel_error_pct'] >  20).sum()}")
print(f"Players with rel. error > 50%:   {(results['abs_rel_error_pct'] >  50).sum()}")
print(f"Players with rel. error > 100%:  {(results['abs_rel_error_pct'] > 100).sum()}")

print(f"\nMean absolute error:            €{results['abs_error_eur'].mean():,.0f}")
print(f"Mean absolute relative error:   {results['abs_rel_error_pct'].mean():.1f}%")
print(f"Median absolute relative error: {results['abs_rel_error_pct'].median():.1f}%")

# FULL table
results.to_csv(os.path.join(MODEL_FOLDER, 'ridge_test_predictions.csv'), index=False)
print(f"\nSaved full predictions to {MODEL_FOLDER}/ridge_test_predictions.csv")