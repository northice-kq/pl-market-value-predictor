import pandas as pd
import numpy as np
import os
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
ML_FOLDER = os.path.join(project_root, 'data', 'ml')
MODEL_FOLDER = os.path.join(project_root, 'data', 'models')
OUTPUT_FOLDER = os.path.join(project_root, 'reports', 'model_output')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

pipeline = joblib.load(os.path.join(MODEL_FOLDER, 'ridge_pipeline.joblib'))
X_test = pd.read_csv(os.path.join(ML_FOLDER, 'X_test.csv'))
Y_test = pd.read_csv(os.path.join(ML_FOLDER, 'Y_test.csv')).squeeze('columns')

id_cols = ['player_id', 'player', 'season', 'team', 'secondary_team', 'nation']
feature_cols = [c for c in X_test.columns if c not in id_cols]

pred_log1p = pipeline.predict(X_test[feature_cols])
pred_eur = np.expm1(pred_log1p)

def build_position_label(row):
    roles = []
    if row['is_DF']: roles.append('DF')
    if row['is_MF']: roles.append('MF')
    if row['is_FW']: roles.append('FW')
    return '/'.join(roles) if roles else '?'

# build the valuation table
df = X_test[['player_id', 'player', 'team', 'season', 'age']].copy()
df['position'] = X_test[['is_DF', 'is_MF', 'is_FW']].apply(build_position_label, axis=1)
df['actual_mv'] = np.expm1(Y_test.values)
df['predicted_mv'] = pred_eur
df['gap_eur'] = df['predicted_mv'] - df['actual_mv']   # + = undervalued, − = overvalued
df['gap_pct'] = df['gap_eur'] / df['actual_mv'] * 100
df['abs_gap_eur'] = df['gap_eur'].abs()

top_undervalued = df.sort_values('gap_eur', ascending=False).head(10) # top 10 undervalued
top_overvalued = df.sort_values('gap_eur', ascending=True).head(10)# top 10 overvalued

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

cols_to_show = ['player', 'team', 'age', 'position',
                'actual_mv', 'predicted_mv', 'gap_eur', 'gap_pct']

print("\n" + "=" * 70)
print("TOP 10 UNDERVALUED PLAYERS (model thinks they should be worth more)")
print("=" * 70)
print(top_undervalued[cols_to_show].to_string(
    index=False, float_format=lambda x: f'{x:,.0f}'))

print("\n" + "=" * 70)
print("TOP 10 OVERVALUED PLAYERS (model thinks they should be worth less)")
print("=" * 100)
print(top_overvalued[cols_to_show].to_string(
    index=False, float_format=lambda x: f'{x:,.0f}'))

df.to_csv(os.path.join(OUTPUT_FOLDER, 'valuations_all.csv'), index=False)
top_undervalued.to_csv(os.path.join(OUTPUT_FOLDER, 'top10_undervalued.csv'), index=False)
top_overvalued.to_csv(os.path.join(OUTPUT_FOLDER, 'top10_overvalued.csv'), index=False)

print(f"\nSaved reports to {OUTPUT_FOLDER}")