import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
ID_DATA_FOLDER = os.path.join(project_root, 'data', 'id_data')
MV_FILE = os.path.join(project_root, 'data', 'transfer_value', 'processed_USABLE_mv.csv')
OUTPUT_FOLDER = os.path.join(project_root, 'data', 'merged_data')

SEASONS = ['2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026']

def get_season_years(season):
    start_year = int(season.split('-')[0])
    return start_year, start_year + 1

def main():
    # Load MV data once
    if not os.path.exists(MV_FILE):
        print(f"MV file not found: {MV_FILE}")
        return
    df_mv = pd.read_csv(MV_FILE)
    df_mv['player_id'] = df_mv['player_id'].astype('Int64')
    print(f"Loaded MV data: {len(df_mv)} players")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for season in SEASONS:
        print(f"\nProcessing {season}...")
        id_file = os.path.join(ID_DATA_FOLDER, f'id_data_{season}.csv')
        if not os.path.exists(id_file):
            print(f"ID file not found: {id_file}")
            continue

        df_id = pd.read_csv(id_file)
        print(f"   Loaded {len(df_id)} rows from ID data")

        start_year, end_year = get_season_years(season)
        pre_col = f"value_at_{start_year}-06"
        mid_col = f"value_at_{start_year}-12"
        target_col = f"value_at_{end_year}-06"
        mid_accel_col = f"acceleration_{start_year}-12"
        end_accel_col = f"acceleration_{end_year}-06"
        mid_log1p_col = f"log1p_acceleration_{start_year}-12"
        end_log1p_col = f"log1p_acceleration_{end_year}-06"

        needed_cols = [pre_col, mid_col, target_col, mid_accel_col, end_accel_col, mid_log1p_col, end_log1p_col]
        existing_cols = set(df_mv.columns)
        available_cols = [col for col in needed_cols if col in existing_cols]
        missing_cols = [col for col in needed_cols if col not in existing_cols]
        if missing_cols:
            print(f"   Missing columns in MV: {missing_cols} – will fill with NaN")

        cols_to_merge = ['player_id'] + available_cols
        df_mv_subset = df_mv[cols_to_merge].copy()

        rename_map = {
            pre_col: 'pre_season_mv', mid_col: 'mid_season_mv', target_col: 'target_mv',
            mid_accel_col: 'mid_season_acceleration', end_accel_col: 'end_season_acceleration',
            mid_log1p_col: 'mid_season_log1p_acceleration', end_log1p_col: 'end_season_log1p_acceleration',
        }
        df_mv_subset = df_mv_subset.rename(columns=rename_map)

        df_merged = df_id.merge(df_mv_subset, on='player_id', how='left')

        # id missing patterns
        mv_cols = ['pre_season_mv', 'mid_season_mv', 'target_mv', 'mid_season_acceleration', 'end_season_acceleration', 'mid_season_log1p_acceleration', 'end_season_log1p_acceleration']
        mv_cols = [col for col in mv_cols if col in df_merged.columns] # keep cols that actually existst
        mv_int_cols = ['pre_season_mv', 'mid_season_mv', 'target_mv', 'mid_season_acceleration','end_season_acceleration']
        for col in mv_int_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].round(0).astype('Int64')

        mv_float_cols = ['mid_season_log1p_acceleration', 'end_season_log1p_acceleration']
        for col in mv_float_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].astype(float)

        df_merged['nan_count'] = df_merged[mv_cols].isna().sum(axis=1) # no. of mv columns are null per row

        completely_missing = df_merged[df_merged['nan_count'] == len(mv_cols)]['player_id'].tolist() #all null
        partially_missing = df_merged[(df_merged['nan_count'] > 0) & (df_merged['nan_count'] < len(mv_cols))]['player_id'].tolist() #at least one is null

        # leftout file
        if completely_missing or partially_missing:
            leftout_file = os.path.join(OUTPUT_FOLDER, f'leftout_{season}.txt')
            with open(leftout_file, 'w', encoding='utf-8') as f:
                f.write(f"Leftout report for {season}\n")
                f.write("=" * 60 + "\n\n")

                if completely_missing:
                    f.write(f"COMPLETELY MISSING (all {len(mv_cols)} MV columns are NaN) - {len(completely_missing)} players\n")
                    for pid in completely_missing:
                        f.write(f"{pid}\n")
                    f.write("\n")

                if partially_missing:
                    f.write(f"PARTIALLY MISSING (some MV columns are NaN) - {len(partially_missing)} players\n")
                    f.write("-" * 60 + "\n")
                    for pid in partially_missing:
                        # Get the specific missing columns for this player
                        row = df_merged[df_merged['player_id'] == pid].iloc[0]
                        name = row['player']
                        missing_cols_list = [col for col in mv_cols if pd.isna(row[col])]
                        f.write(f"{pid} ({name}): {', '.join(missing_cols_list)}\n")
                    f.write("\n")

            print(f"        Leftout report saved: {leftout_file}")
            print(f"        Completely missing: {len(completely_missing)}")
            print(f"        Partially missing: {len(partially_missing)}")
        else:
            print(f"All players have complete MV data!")

        df_merged = df_merged.drop(columns=['nan_count'], errors='ignore')# drop the temporary nan_count column
        output_file = os.path.join(OUTPUT_FOLDER, f'merged_data_{season}.csv')
        df_merged.to_csv(output_file, index=False)
        print(f"Saved merged data to {output_file}")

    print("\nAll seasons processed!")

if __name__ == "__main__":
    main()