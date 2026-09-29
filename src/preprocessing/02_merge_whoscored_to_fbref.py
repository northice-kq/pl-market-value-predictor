import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # up 2 levels

FBREF_FOLDER = os.path.join(project_root, 'data', 'combined')  # FBref integrated files
WHOSCORED_FOLDER = os.path.join(project_root, 'data', 'raw', 'whoscored')  # WhoScored raw files
OUTPUT_FOLDER = os.path.join(project_root, 'data', 'combined')  # output files

SEASONS = ['2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026']

# columns to extract
WHOSCORED_COLS = ['KeyP', 'AvgP', 'PS%', 'LongB', 'ThrB', 'Rating']


def merge_season(season, fill_missing):
    """Merge FBref integrated data with WhoScored passing stats for one season."""
    print(f"\nProcessing {season}...")

    #  file paths
    fbref_path = os.path.join(FBREF_FOLDER, f'fbref_{season}_integrated.csv')
    whoscored_path = os.path.join(WHOSCORED_FOLDER, f'whoscored_{season}_passing.csv')
    output_path = os.path.join(OUTPUT_FOLDER, f'players_stats_combined_{season}.csv')
    leftout_path = os.path.join(OUTPUT_FOLDER, f'leftout_{season}.txt')

    # will skip if output already exists
    if os.path.exists(output_path):
        print(f"Output already exists: {output_path}, skipping")
        return

    if not os.path.exists(fbref_path):
        print(f"FBref integrated file not found: {fbref_path}")
        return
    if not os.path.exists(whoscored_path):
        print(f"WhoScored file not found: {whoscored_path}")
        return

    # Loading data
    df_fbref = pd.read_csv(fbref_path)
    df_whoscored = pd.read_csv(whoscored_path)

    # Standardize player and team/club name column
    df_whoscored = df_whoscored.rename(columns={'Player': 'player'})
    df_whoscored = df_whoscored.rename(columns={'Club': 'club'})
    df_fbref['team'] = df_fbref['team'].astype(str).str.strip()
    df_whoscored['club'] = df_whoscored['club'].astype(str).str.strip()

    # ensure columns are strings and trimmed
    df_fbref['player'] = df_fbref['player'].astype(str).str.strip()
    df_whoscored['player'] = df_whoscored['player'].astype(str).str.strip()

    # create lower case version for case insensitive matching
    df_fbref['player_lower'] = df_fbref['player'].str.lower()
    df_whoscored['player_lower'] = df_whoscored['player'].str.lower()

    #create merge key
    df_fbref['merge_key'] = df_fbref['player_lower'] + '|' + df_fbref['team']
    df_whoscored['merge_key'] = df_whoscored['player_lower'] + '|' + df_whoscored['club']

    # select needed cols
    whoscored_cols_present = [col for col in WHOSCORED_COLS if col in df_whoscored.columns]
    if len(whoscored_cols_present) != len(WHOSCORED_COLS):
        missing = set(WHOSCORED_COLS) - set(whoscored_cols_present)
        print(f"Missing WhoScored columns: {missing}, will skip")

    df_whoscored_subset = df_whoscored[['merge_key'] + whoscored_cols_present]

    #  merge: left join on player name (keep all FBref players)
    df_merged = df_fbref.merge(df_whoscored_subset, on='merge_key', how='left')

    df_merged = df_merged.drop(columns=['player_lower', 'merge_key'], errors='ignore') # drop temp keys

    # Find players that were NOT FOUND in WhoScored
    # check if any of the whoScored columns are NaN, and it means no match
    merged_whoscored_cols = [col for col in whoscored_cols_present if col in df_merged.columns]
    if merged_whoscored_cols:
        missing_mask = df_merged[merged_whoscored_cols].isna().all(axis=1) | (df_merged[merged_whoscored_cols] == '').all(axis=1) # Rows where all whoscored columns are missing (NaN or empty string)
        missing_players = df_merged[missing_mask]['player'].tolist()
    else:
        missing_players = df_merged['player'].tolist()
        # Write in to leftout file
        with open(leftout_path, 'w', encoding='utf-8') as f:
            if missing_players: # bool value
                f.write(f"Players from FBref not found in WhoScored for {season}:\n")
                f.write("=" * 60 + "\n")
                for p in missing_players:
                    f.write(f"{p}\n")
                print(f"{len(missing_players)} players not found, logged to {leftout_path}")
            else:
                f.write(f"All players from FBref were found in WhoScored for {season}.\n")
                print(f"All players matched for {season}")

    if fill_missing and merged_whoscored_cols:
        # replace NaN && empty strings with 0, becuase WhoScored use it as 0
        for col in merged_whoscored_cols:
            df_merged[col] = df_merged[col].fillna(0).replace('', 0)
        print(f"Filled missing WhoScored values with 0 for {season}")

    df_merged = df_merged.drop(columns=['player_lower'], errors='ignore') # drop lower case col
    # Ensure output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # save merged data
    df_merged.to_csv(output_path, index=False)
    print(f"Saved merged file: {output_path} (columns: {len(df_merged.columns)})")


def main():
    print("Merging FBref integrated data with WhoScored passing stats...")
    print("=" * 60)
    fill_choice = input("Do you want to fill missing WhoScored values (NaN) with 0? (Y/N): ").strip().upper()
    fill_missing = fill_choice == 'Y'
    for season in SEASONS:
        merge_season(season, fill_missing)
    print("\nAll seasons processed! Files are in 'data/combined/'. Check 'leftout' files for unmatched names")


if __name__ == "__main__":
    main()