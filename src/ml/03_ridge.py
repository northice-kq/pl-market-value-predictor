import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
ML_FOLDER = os.path.join(project_root, 'data', 'ml')
MODEL_FOLDER = os.path.join(project_root, 'data', 'models')
os.makedirs(MODEL_FOLDER, exist_ok=True)

X_train = pd.read_csv(os.path.join(ML_FOLDER, 'X_train.csv'))
X_val   = pd.read_csv(os.path.join(ML_FOLDER, 'X_val.csv'))
X_test  = pd.read_csv(os.path.join(ML_FOLDER, 'X_test.csv'))
Y_train = pd.read_csv(os.path.join(ML_FOLDER, 'Y_train.csv')).squeeze('columns')
Y_val   = pd.read_csv(os.path.join(ML_FOLDER, 'Y_val.csv')).squeeze('columns')
Y_test  = pd.read_csv(os.path.join(ML_FOLDER, 'Y_test.csv')).squeeze('columns')

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# drop identify cols
id_cols = ['player_id', 'player', 'season', 'team', 'secondary_team', 'nation']
feature_cols = [c for c in X_train.columns if c not in id_cols]

print(f"\nUsing {len(feature_cols)} features:")
for c in feature_cols:
    print(f"  - {c}")

X_train_f = X_train[feature_cols]
X_val_f   = X_val[feature_cols]
X_test_f  = X_test[feature_cols]

# pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # safety net for any leftover NaN, but should not happen as it is filled in the prev .py
    ('scaler', StandardScaler()), # Ridge needs scaled inputs
    ('ridge', Ridge(alpha=1.0))
])

# test
alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
best_alpha = None
best_val_rmse = float('inf')

print("\n--- Alpha tuning on validation set ---")
for alpha in alphas:
    pipeline.set_params(ridge__alpha=alpha)
    pipeline.fit(X_train_f, Y_train)
    preds = pipeline.predict(X_val_f)
    rmse = np.sqrt(mean_squared_error(Y_val, preds))
    print(f"  alpha={alpha:<6}  val RMSE (log): {rmse:.5f}")
    if rmse < best_val_rmse:
        best_val_rmse = rmse
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha} (val RMSE log: {best_val_rmse:.5f})")

# refit model w best fit alpha
pipeline.set_params(ridge__alpha=best_alpha)
pipeline.fit(X_train_f, Y_train)

def evaluate(name, X, y):
    preds = pipeline.predict(X)
    rmse_log = np.sqrt(mean_squared_error(y, preds))
    mae_log  = mean_absolute_error(y, preds)
    r2       = r2_score(y, preds)

    # back to euro scale for interpretability
    preds_eur = np.expm1(preds)
    y_eur     = np.expm1(y)
    rmse_eur  = np.sqrt(mean_squared_error(y_eur, preds_eur))
    mae_eur   = mean_absolute_error(y_eur, preds_eur)

    print(f"\n{name}:")
    print(f"RMSE (log):  {rmse_log:.4f}")
    print(f"MAE  (log):  {mae_log:.4f}")
    print(f"R²   (log):  {r2:.4f}")
    print(f"RMSE (€):    {rmse_eur:,.0f}")
    print(f"MAE  (€):    {mae_eur:,.0f}")

evaluate("TRAIN", X_train_f, Y_train)
evaluate("VAL  ", X_val_f,   Y_val)
evaluate("TEST ", X_test_f,  Y_test)

# inspect coef
ridge_model = pipeline.named_steps['ridge']
coefs = pd.DataFrame({
    'feature': feature_cols,
    'coef': ridge_model.coef_
}).sort_values('coef', key=abs, ascending=False)

print("\n Coefficients (by magnitude) ")
print(coefs.to_string(index=False))

#save
joblib.dump(pipeline, os.path.join(MODEL_FOLDER, 'ridge_pipeline.joblib')) #model
with open(os.path.join(MODEL_FOLDER, 'feature_cols.txt'), 'w', encoding='utf-8') as f: # feature list
    f.write('\n'.join(feature_cols))
coefs.to_csv(os.path.join(MODEL_FOLDER, 'ridge_coefficients.csv'), index=False) # coef
metrics_rows = [] # metric
for name, X, y in [('train', X_train_f, Y_train),('val',   X_val_f,   Y_val),('test',  X_test_f,  Y_test)]:
    preds = pipeline.predict(X)
    metrics_rows.append({
        'split': name,
        'rmse_log': np.sqrt(mean_squared_error(y, preds)),
        'mae_log':  mean_absolute_error(y, preds),
        'r2_log':   r2_score(y, preds),
        'rmse_eur': np.sqrt(mean_squared_error(np.expm1(y), np.expm1(preds))),
        'mae_eur':  mean_absolute_error(np.expm1(y), np.expm1(preds)),
    })
pd.DataFrame(metrics_rows).to_csv(
    os.path.join(MODEL_FOLDER, 'ridge_metrics.csv'), index=False)

print(f"\nSaved model to {MODEL_FOLDER}")