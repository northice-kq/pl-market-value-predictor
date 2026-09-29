import pandas as pd
import os
from datetime import datetime
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
PLAYER_ID_FILE = os.path.join(project_root, 'data', 'raw', 'raw_transfer_data', 'prem_player_id.csv')
MV_FILE = os.path.join(project_root, 'data', 'raw', 'raw_transfer_data', 'prem_market_value.csv')
OUTPUT_FOLDER = os.path.join(project_root, 'data', 'transfer_value')
OUTPUT_ALL = os.path.join(OUTPUT_FOLDER, 'processed_mv.csv')
OUTPUT_USABLE = os.path.join(OUTPUT_FOLDER, 'processed_USABLE_mv.csv')

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%d/%m/%Y')
    except:
        return None

def main():
    # name mapping
    if not os.path.exists(PLAYER_ID_FILE):
        print(f" ID file not found: {PLAYER_ID_FILE}")
        return
    df_id = pd.read_csv(PLAYER_ID_FILE)
    if 'player_id' not in df_id.columns or 'player' not in df_id.columns:
        print("prem_player_id.csv must have columns: player_id, player")
        return
    id_to_name = df_id.drop_duplicates('player_id').set_index('player_id')['player'].to_dict()
    print(f"Loaded {len(id_to_name)} player IDs with names.")

    # read market value
    if not os.path.exists(MV_FILE):
        print(f"Market value file not found: {MV_FILE}")
        return
    df_mv = pd.read_csv(MV_FILE)
    required_mv = ['player_id', 'date', 'market_value_in_eur', 'current_club_name']
    for col in required_mv:
        if col not in df_mv.columns:
            print(f"Missing column in market_value.csv: {col}")
            return
    print(f"Loaded {len(df_mv)} market value rows.")

    # filter to only players in the id mapping
    df_mv = df_mv[df_mv['player_id'].isin(id_to_name.keys())]
    print(f"Filtered to {len(df_mv)} rows for known players.")

    if df_mv.empty:
        print("No matching rows found. Check player_id consistency.")
        return

    # parse dates and turn into year-month and take their latest value
    df_mv['date_parsed'] = df_mv['date'].apply(parse_date)
    df_mv = df_mv.dropna(subset=['date_parsed'])
    df_mv['year_month'] = df_mv['date_parsed'].dt.strftime('%Y-%m')

    df_mv = df_mv.sort_values(['player_id', 'date_parsed'])
    df_mv_monthly = df_mv.groupby(['player_id', 'year_month'], as_index=False).last()

    records = []
    for player_id, group in df_mv_monthly.groupby('player_id'):
        player_name = id_to_name.get(player_id, '')

        # chronological order
        full_player_data = df_mv[df_mv['player_id'] == player_id].sort_values('date_parsed')
        clubs = []
        seen = set()
        for col, row in full_player_data.iterrows():
            club = row['current_club_name'].strip()
            if club not in seen:
                seen.add(club)
                clubs.append(club)
        clubs_str = ','.join(clubs)
        record = {'player_id': player_id, 'player': player_name, 'clubs': clubs_str}
        group_sorted = group.sort_values('year_month')
        for _, row in group_sorted.iterrows():
            month = row['year_month']
            col_name = f"value_at_{month}"
            record[col_name] = row['market_value_in_eur']
        records.append(record)

    df_wide = pd.DataFrame(records)

    # sort columns chronologically / round to integers
    value_cols = [col for col in df_wide.columns if col.startswith('value_at_')]
    sorted_value_cols = sorted(value_cols, key=lambda x: x.split('_')[-1])
    final_cols = ['player_id', 'player', 'clubs'] + sorted_value_cols
    df_wide = df_wide[final_cols]
    for col in sorted_value_cols:
        df_wide[col] = df_wide[col].round(0).astype('Int64')

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    df_wide.to_csv(OUTPUT_ALL, index=False)
    print(f"Saved full monthly data to {OUTPUT_ALL}")

    # build USABLE VERSION
    # Select only June/December columns from the full date
    df_wide[sorted_value_cols] = df_wide[sorted_value_cols].ffill(axis=1) #use forward fill

    # only June/December columns
    june_dec_cols = [col for col in sorted_value_cols if col.endswith('-06') or col.endswith('-12')]
    df_usable = df_wide[['player_id', 'player', 'clubs'] + june_dec_cols].copy()
    for col in june_dec_cols:
        df_usable[col] = df_usable[col].round(0).astype('Int64')

    #compute ABSOLUTE acceleration
    acceleration_cols = {}
    if len(june_dec_cols) >= 3:
        for i in range(2, len(june_dec_cols)):
            prev1 = june_dec_cols[i-1]
            prev2 = june_dec_cols[i-2]
            current = june_dec_cols[i]
            change_prev = df_usable[prev1] - df_usable[prev2]
            change_curr = df_usable[current] - df_usable[prev1]
            accel_col = f"acceleration_{current.split('_')[-1]}"
            acceleration_cols[accel_col] = (change_curr - change_prev).round(0).astype('Int64')

    for col_name, values in acceleration_cols.items():
        df_usable[col_name] = values

    # LOG ACCELERATION
    #Formula: log1p(current) - 2*log1p(prev1) + log1p(prev2)

    log1p_acceleration_cols = {}
    if len(june_dec_cols) >= 3:
        for i in range(2, len(june_dec_cols)):
            prev1 = june_dec_cols[i - 1]
            prev2 = june_dec_cols[i - 2]
            current = june_dec_cols[i]

            prev2_vals = df_usable[prev2].astype(float)
            prev1_vals = df_usable[prev1].astype(float)
            current_vals = df_usable[current].astype(float)

            log1p_accel = (
                    np.log1p(current_vals)
                    - 2 * np.log1p(prev1_vals)
                    + np.log1p(prev2_vals)
            )
            
            accel_col = f"log1p_acceleration_{current.split('_')[-1]}"
            log1p_acceleration_cols[accel_col] = log1p_accel.round(4)

    for col_name, values in log1p_acceleration_cols.items():
        df_usable[col_name] = values



    #player info + June/December columns + acceleration columns
    final_usable_cols = (['player_id', 'player', 'clubs'] + june_dec_cols + list(acceleration_cols.keys()) + list(log1p_acceleration_cols.keys()))
    df_usable = df_usable[final_usable_cols]


    df_usable.to_csv(OUTPUT_USABLE, index=False)
    print(f"Saved USABLE (June/December + acceleration) to {OUTPUT_USABLE}")
    print(f"   Number of June/December columns: {len(june_dec_cols)}")
    print(f"   Number of acceleration columns: {len(acceleration_cols)}")
    print(f"   Number of log acceleration columns: {len(log1p_acceleration_cols)}")

if __name__ == "__main__":
    main()