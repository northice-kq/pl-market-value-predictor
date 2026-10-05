import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
INPUT_FOLDER = os.path.join(project_root, 'data', 'merged_data')
OUTPUT_FOLDER = os.path.join(project_root, 'data', 'final')
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'final_data_v1.1.csv')

SEASONS = ['2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026']
MV_COLS = ['pre_season_mv', 'mid_season_mv', 'target_mv', 'mid_season_acceleration', 'end_season_acceleration', 'mid_season_log1p_acceleration', 'end_season_log1p_acceleration']

def main():
    print("Combining merged data into final dataset...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    df_list = []

    for season in SEASONS:
        input_file = os.path.join(INPUT_FOLDER, f'merged_data_{season}.csv')
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            continue

        df = pd.read_csv(input_file)
        print(f"Loaded {season}: {len(df)} rows")

        cols_to_check = [col for col in MV_COLS if col in df.columns] # drop player w/o mv data
        df_clean = df.dropna(subset=cols_to_check)
        dropped = len(df) - len(df_clean)
        if dropped > 0:
            print(f"   Dropped {dropped} rows with missing MV data")

        df_list.append(df_clean)

    if not df_list:
        print("No data loaded.")
        return

    # concat all seasons
    final_df = pd.concat(df_list, ignore_index=True)
    final_df = final_df.sort_values(['season', 'player_id']).reset_index(drop=True)

    mv_int_cols = ['pre_season_mv', 'mid_season_mv', 'target_mv', 'mid_season_acceleration', 'end_season_acceleration'] #change it to int
    for col in mv_int_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].round(0).astype('Int64')  # Int64 allows NaN

    mv_float_cols = ['mid_season_log1p_acceleration', 'end_season_log1p_acceleration']
    for col in mv_float_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].astype(float)

    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved final dataset to {OUTPUT_FILE}\n")
    print(f"   Total rows: {len(final_df)}")
    print(f"   Columns: {list(final_df.columns)}")

if __name__ == "__main__":
    main()