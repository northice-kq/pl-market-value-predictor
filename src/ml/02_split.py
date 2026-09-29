import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
ML_FOLDER = os.path.join(project_root, 'data', 'ml')

X = pd.read_csv(os.path.join(ML_FOLDER, 'X_raw.csv'))
Y = pd.read_csv(os.path.join(ML_FOLDER, 'Y_log1p_target.csv')).squeeze('columns')

print(f"Loaded X: {X.shape}")
print(f"Loaded Y: {Y.shape}")

# season split
train_seasons = [2122, 2223, 2324]
val_seasons   = [2425]
test_seasons  = [2526]

train_mask = X['season'].isin(train_seasons)
val_mask   = X['season'].isin(val_seasons)
test_mask  = X['season'].isin(test_seasons)

print(f"\nTrain rows: {train_mask.sum()}")
print(f"Val rows:   {val_mask.sum()}")
print(f"Test rows:  {test_mask.sum()}")
print(f"Total:      {train_mask.sum() + val_mask.sum() + test_mask.sum()}")

X_train, Y_train = X[train_mask].copy(), Y[train_mask].copy()
X_val,   Y_val   = X[val_mask].copy(),   Y[val_mask].copy()
X_test,  Y_test  = X[test_mask].copy(),  Y[test_mask].copy()

X_train.to_csv(os.path.join(ML_FOLDER, 'X_train.csv'), index=False)
X_val.to_csv(os.path.join(ML_FOLDER, 'X_val.csv'), index=False)
X_test.to_csv(os.path.join(ML_FOLDER, 'X_test.csv'), index=False)
Y_train.to_csv(os.path.join(ML_FOLDER, 'Y_train.csv'), index=False)
Y_val.to_csv(os.path.join(ML_FOLDER, 'Y_val.csv'), index=False)
Y_test.to_csv(os.path.join(ML_FOLDER, 'Y_test.csv'), index=False)

print("\nSaved train/val/test splits to", ML_FOLDER)