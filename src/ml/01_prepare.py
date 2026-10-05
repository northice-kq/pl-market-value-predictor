from multiprocessing.reduction import duplicate

import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
INPUT_FILE = os.path.join(project_root, 'data', 'final', 'final_data_v1.1.csv')
OUTPUT_FOLDER = os.path.join(project_root, 'data', 'ml')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

df = df[~df['pos'].fillna('').str.upper().str.contains('GK')].copy() # remove GK
print(f"After removing GK: {len(df)} rows") #Note, please check if the resultant row number is correct

# feature
df['log1p_target_mv'] = np.log1p(df['target_mv']) # target
df['age_2'] = df['age'] ** 2 # age^2
df['log1p_pre_mv'] = np.log1p(df['pre_season_mv'])
df['log1p_mid_mv'] = np.log1p(df['mid_season_mv'])


df['pos_upper'] = df['pos'].fillna('').str.upper()
df['is_DF'] = df['pos_upper'].str.contains('DF').astype(int)
df['is_MF'] = df['pos_upper'].str.contains('MF').astype(int)
df['is_FW'] = df['pos_upper'].str.contains('FW').astype(int)
df = df.drop(columns=['pos', 'pos_upper'])  # drop original pos

count_cols = [
    'Playing Time_MP', 'Playing Time_Starts', 'Playing Time_Min',
    'Performance_Gls', 'Performance_Ast', 'Performance_G-PK',
    'Performance_PK', 'Performance_PKatt',
    'Standard_Sh', 'Standard_SoT',
    'Performance_CrdY', 'Performance_CrdR', 'Performance_2CrdY',
    'Performance_Fls', 'Performance_Fld', 'Performance_Off',
    'Performance_Crs', 'Performance_Int', 'Performance_TklW', 'Performance_OG',
    'KeyP', 'AvgP', 'LongB', 'ThrB',
]

rate_cols = [
    'Per 90 Minutes_Gls', 'Per 90 Minutes_Ast', 'Per 90 Minutes_G-PK', 'Per 90 Minutes_G+A-PK',
    'Standard_SoT%', 'Standard_Sh/90', 'Standard_SoT/90',
    'Standard_G/Sh', 'Standard_G/SoT', 'PS%', 'Rating',
]

for col in count_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0) # fill NaN with 0

for col in rate_cols:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val) # filled with median

# Drop rows where market value log features are missing
before = len(df)
df = df.dropna(subset=['log1p_pre_mv', 'log1p_mid_mv', 'mid_season_log1p_acceleration'])
after = len(df)
print(f"Dropped {before - after} rows without 'log1p_pre_mv', 'log1p_mid_mv', 'mid_season_log1p_acceleration'")

target_col = 'log1p_target_mv'
leakage_cols = ['target_mv', 'end_season_acceleration', 'end_season_log1p_acceleration']
duplicate_cols = ['pre_season_mv','mid_season_mv','mid_season_acceleration']

drop_from_X =  leakage_cols + [target_col] + duplicate_cols

X = df.drop(columns=drop_from_X)
Y = df['log1p_target_mv']

print(f"\nFeatures X: {X.shape}")
print(f"Target Y:   {Y.shape}")
print(f"\nFeature columns:")
for col in X.columns:
    print(f"  - {col}")

X.to_csv(os.path.join(OUTPUT_FOLDER, 'X_raw.csv'), index=False)
Y.to_csv(os.path.join(OUTPUT_FOLDER, 'Y_log1p_target.csv'), index=False)
print(f"\nSaved X and Y to {OUTPUT_FOLDER}")