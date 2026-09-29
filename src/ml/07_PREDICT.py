import pandas as pd
import numpy as np
import os
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
ML_FOLDER = os.path.join(project_root, 'data', 'ml')
MODEL_FOLDER = os.path.join(project_root, 'data', 'models')
INPUT_CSV = os.path.join(project_root, 'data', 'input', 'players_template.csv')
OUTPUT_CSV = os.path.join(project_root, 'reports', 'model_output', 'predictions.csv')

pipeline = joblib.load(os.path.join(MODEL_FOLDER, 'ridge_pipeline.joblib'))
X_train = pd.read_csv(os.path.join(ML_FOLDER, 'X_train.csv'))

id_cols = ['player_id', 'player', 'season', 'team', 'secondary_team', 'nation']
feature_cols = [c for c in X_train.columns if c not in id_cols]
medians = X_train[feature_cols].median()

# training range for validation & hard limits
AGE_LO, AGE_HI = X_train['age'].quantile([0.05, 0.95])
PRE_MV_LO = np.expm1(X_train['log1p_pre_mv']).quantile(0.05)
PRE_MV_HI = np.expm1(X_train['log1p_pre_mv']).quantile(0.95)
MID_MV_LO = np.expm1(X_train['log1p_mid_mv']).quantile(0.05)
MID_MV_HI = np.expm1(X_train['log1p_mid_mv']).quantile(0.95)

AGE_HARD_LO, AGE_HARD_HI = 15, 45
MV_HARD_LO, MV_HARD_HI = 50000, 300_000_000

def parse_number(value, field_name, player_label):
    """
    Try to convert a value to float. Returns (float_value, error_message).
    On success: (value, None)
    On failure: (None, 'reason')
    """
    if pd.isna(value):
        return None, None  # blank cell

    if isinstance(value, str):
        value = value.strip().replace(',', '').replace('€', '')

    try:
        return float(value), None
    except (ValueError, TypeError):
        return None, f"'{field_name}' = '{value}' is not a valid number"

def validate_and_build(player_label, player_data):
    """
    Validate required inputs, warn on out-of-range values, return feature row or None.
    """
    warnings = []
    soft_warnings = False
    POS_KEY = 'position (DF/MF/FW - comma-separated if multiple)'

    # required fields
    required = ['age', POS_KEY, 'pre_season_mv', 'mid_season_mv']
    missing = [f for f in required if pd.isna(player_data.get(f))]
    if missing:
        print(f"{player_label}: missing required field(s): {', '.join(missing)}. SKIPPED.")
        print('=' * 30)
        return None, warnings, None, False, None

    parsed = {}
    for field in ['age', 'pre_season_mv', 'mid_season_mv']:
        value, err = parse_number(player_data[field], field, player_label)
        if err:
            print(f"{player_label}: {err}. SKIPPED.")
            print('=' * 30)
            return None, warnings, None, False, None
        parsed[field] = value

    age = parsed['age']
    pre_mv = parsed['pre_season_mv']
    mid_mv = parsed['mid_season_mv']


    position = str(player_data[POS_KEY]).strip().upper()
    parsed['position'] = position
    valid_positions = {'DF', 'MF', 'FW'}
    detected = {p for p in valid_positions if p in position}

    # hard rej.
    if not (AGE_HARD_LO <= age <= AGE_HARD_HI):
        print(f"{player_label}: age {age} outside plausible range ({AGE_HARD_LO}–{AGE_HARD_HI}). SKIPPED.")
        print('=' * 30)
        return None, warnings, None, False, None
    if not (MV_HARD_LO <= pre_mv <= MV_HARD_HI):
        print(f"{player_label}: pre-season value €{pre_mv:,.0f} outside plausible range(€{MV_HARD_LO:,}–€{MV_HARD_HI:,}). SKIPPED.")
        print('=' * 30)
        return None, warnings, None, False, None
    if not (MV_HARD_LO <= mid_mv <= MV_HARD_HI):
        print(f"{player_label}: mid-season value €{mid_mv:,.0f} outside plausible range (€{MV_HARD_LO:,}–€{MV_HARD_HI:,}). SKIPPED.")
        print('=' * 30)
        return None, warnings, None, False, None
    if not detected:
        print(f"{player_label}: position '{player_data[POS_KEY]}' invalid. Must contain at least one of DF, MF, or FW. SKIPPED.")
        print('=' * 30)
        return None, warnings, None, False, None

    # soft warning: outside optimal range
    if age < AGE_LO or age > AGE_HI:
        warnings.append(
            f"age {age:.0f} is in the top/bottom 5% of the training distribution (5–95% spans {AGE_LO:.1f}–{AGE_HI:.1f}), prediction involves extrapolation")
        soft_warnings = True
    if pre_mv < PRE_MV_LO or pre_mv > PRE_MV_HI:
        warnings.append(
            f"pre-season value €{pre_mv:,.0f} is in the top/bottom 5% of the training distribution (5–95% spans €{PRE_MV_LO:,.0f}–€{PRE_MV_HI:,.0f}), prediction involves extrapolation")
        soft_warnings = True
    if mid_mv < MID_MV_LO or mid_mv > MID_MV_HI:
        warnings.append(
            f"mid-season value €{mid_mv:,.0f} is in the top/bottom 5% of the training distribution (5–95% spans €{MID_MV_LO:,.0f}–€{MID_MV_HI:,.0f}), prediction involves extrapolation")
        soft_warnings = True


    log_pre = np.log1p(pre_mv)
    log_mid = np.log1p(mid_mv)
    prev_mid_mv_raw = player_data.get('prev_mid_season_mv')
    prev_mid_mv, err = parse_number(prev_mid_mv_raw, 'prev_mid_season_mv', player_label)
    if err:
        print(f"{player_label}: {err}. Acceleration set to 0.")
        warnings.append(f"invalid prev_mid_season_mv; acceleration set to 0")
        prev_mid_mv = None
    if pd.isna(prev_mid_mv):
        accel = 0.0
    else:
        prev_mid_mv = float(prev_mid_mv)
        if prev_mid_mv <= 0:
            warnings.append("previous mid-season value ≤ 0; acceleration set to 0")
            accel = 0.0
        else:
            accel = log_mid - 2 * log_pre + np.log1p(prev_mid_mv)
    parsed['prev_mid_season_mv'] = prev_mid_mv

    # feature row
    row = medians.copy()
    row['age'] = age
    row['age_2'] = age ** 2
    row['log1p_pre_mv'] = log_pre
    row['log1p_mid_mv'] = log_mid
    row['mid_season_log1p_acceleration'] = accel
    row['is_DF'] = 1 if 'DF' in detected else 0
    row['is_MF'] = 1 if 'MF' in detected else 0
    row['is_FW'] = 1 if 'FW' in detected else 0

    # op stat overide
    derived = {'age', 'age_2', 'log1p_pre_mv', 'log1p_mid_mv',
               'mid_season_log1p_acceleration', 'is_DF', 'is_MF', 'is_FW'}

    user_provided = set() #used to track what data user gave
    for key, value in player_data.items():
        if pd.isna(value):
            continue
        if key in derived or key in {'player', 'team', 'season', 'prev_mid_season_mv'}:
            continue
        if key in feature_cols:
            stat_value, err = parse_number(value, key, player_label)
            if err:
                print(f"{player_label}: {err}. Using median for '{key}'.")
                soft_warnings = True
                continue
            row[key] = stat_value
            user_provided.add(key)

    # derive per-90 and ratio stats where possible
    minutes = None
    if 'Playing Time_Min' in user_provided:
        minutes = row['Playing Time_Min']

    if minutes is not None and minutes > 0:
        def derive(target, required_inputs, formula): #only derive if user provided all required inputs
            if not all(k in user_provided for k in required_inputs):
                return
            try:
                row[target] = formula()
            except (ZeroDivisionError, ValueError, TypeError):
                pass  # leave median

        derive('Per 90 Minutes_Gls',
               ['Performance_Gls'],
               lambda: row['Performance_Gls'] / minutes * 90)

        derive('Per 90 Minutes_Ast',
               ['Performance_Ast'],
               lambda: row['Performance_Ast'] / minutes * 90)

        derive('Per 90 Minutes_G-PK',
               ['Performance_G-PK'],
               lambda: row['Performance_G-PK'] / minutes * 90)

        derive('Per 90 Minutes_G+A-PK',
               ['Performance_G-PK', 'Performance_Ast'],
               lambda: (row['Performance_G-PK'] + row['Performance_Ast']) / minutes * 90)

        derive('Standard_Sh/90',
               ['Standard_Sh'],
               lambda: row['Standard_Sh'] / minutes * 90)

        derive('Standard_SoT/90',
               ['Standard_SoT'],
               lambda: row['Standard_SoT'] / minutes * 90)

    # non-per-90 ratios
    def derive_ratio(target, numerator, denominator):
        if numerator not in user_provided or denominator not in user_provided:
            return
        try:
            d = row[denominator]
            if d == 0:
                return
            row[target] = row[numerator] / d
        except (ZeroDivisionError, ValueError, TypeError):
            pass

    derive_ratio('Standard_SoT%', 'Standard_SoT', 'Standard_Sh')
    derive_ratio('Standard_G/Sh', 'Performance_Gls', 'Standard_Sh')
    derive_ratio('Standard_G/SoT', 'Performance_Gls', 'Standard_SoT')

    return row, warnings, accel, soft_warnings, parsed


print('=' * 30)
print(f"Training age range (5–95%):  {AGE_LO:.1f} – {AGE_HI:.1f}")
print(f"Training pre-MV range (5–95%): €{PRE_MV_LO:,.0f} – €{PRE_MV_HI:,.0f}")
print(f"Training mid-MV range (5–95%): €{MID_MV_LO:,.0f} – €{MID_MV_HI:,.0f}")
print('=' * 30)

raw = pd.read_csv(INPUT_CSV, index_col='parameter')
print(f"Loaded template with {raw.shape[1]} player column(s).")

# transpose so each row is a player
players = raw.T
print(f"Transposed to {players.shape[0]} players x {players.shape[1]} parameters.")
print('=' * 30)
results = []
for player_label, player_data in players.iterrows():
    output = validate_and_build(player_label, player_data)
    if output[0] is None:
        continue
    row, warnings, accel, soft_warnings, parsed = output

    for w in warnings:
        print(f"{player_label}: {w}")
    if soft_warnings:
        print('=' * 30)
    X_input = pd.DataFrame([row])[feature_cols]
    pred_log1p = pipeline.predict(X_input)[0]
    pred_eur = np.expm1(pred_log1p)

    age = parsed['age']
    pre_mv = parsed['pre_season_mv']
    mid_mv = parsed['mid_season_mv']
    position = parsed['position']

    change_eur = pred_eur - mid_mv
    change_pct = (change_eur / mid_mv) * 100 if mid_mv > 0 else 0

    results.append({
        'player_label': player_label,
        'player': player_data.get('player', ''),
        'age': age,
        'position': position,
        'pre_season_mv': pre_mv,
        'mid_season_mv': mid_mv,
        'prev_mid_season_mv': parsed.get('prev_mid_season_mv'),
        'computed_accel': round(accel, 4),
        'predicted_mv': round(pred_eur),
        'predicted_change_eur': round(change_eur),
        'predicted_change_pct': round(change_pct, 2),
    })
if not results:
    print("\nNo valid players found. Fill in the required fields and rerun.")
else:
    df_out = pd.DataFrame(results)
    print("\n=== Predictions ===")
    print(df_out.to_string(index=False))

    # Colored summary (after the clean table)
    print("\n=== Summary ===")
    for r in results:
        pred = f"€{r['predicted_mv']:,.0f}"
        change = r['predicted_change_pct']
        color = "\033[32m" if change > 0 else "\033[31m" if change < 0 else "\033[0m"

        name = r.get('player', '')
        label = f"{r['player_label']} ({name})" if name and not pd.isna(name) else r['player_label']
        label_padded = f"{label:<40}"
        print(f"  {label_padded} → {color}{pred:>16}\033[0m  ({change:+.2f}%)")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved predictions to {OUTPUT_CSV}")