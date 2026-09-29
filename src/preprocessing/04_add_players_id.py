import pandas as pd
import os
import unicodedata
import re
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
ID_FILE = os.path.join(project_root, 'data', 'raw_transfer_data', 'prem_player_id.csv')
MANUAL_MAP_FILE = os.path.join(project_root, 'data', 'raw', 'raw_transfer_data', 'player_name_mappings.txt')
INPUT_FOLDER = os.path.join(project_root, 'data', 'combined')
OUTPUT_FOLDER = os.path.join(project_root, 'data', 'id_data')

SEASONS = ['2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026']

def normalise_name(name):
    if not isinstance(name, str):
        return ''
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_only = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_only.lower().strip()

def parse_manual_mappings(filepath):
    """Build two dictionaries from the manual mapping file."""
    name_to_id = {}
    name_to_correct = {}
    if not os.path.exists(filepath):
        print(f"Manual mappings file not found: {filepath}")
        return name_to_id, name_to_correct

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ' - ' in line:
                left, right = line.split(' - ', 1)
            elif ' – ' in line:
                left, right = line.split(' – ', 1)
            else:
                continue

            left = left.strip()
            right = right.strip()

            numbers = re.findall(r'\d+', right)
            if numbers:
                player_id = int(numbers[0])
                name_to_id[left] = player_id
            else:
                name_to_correct[left] = right

    print(f"Built {len(name_to_id)} direct ID mappings and {len(name_to_correct)} name corrections.")
    return name_to_id, name_to_correct

def main():

    if not os.path.exists(ID_FILE):
        print(f"ID file not found: {ID_FILE}")
        return

    df_ids = pd.read_csv(ID_FILE)
    df_ids['name_norm'] = df_ids['player'].apply(normalise_name)
    id_map = dict(zip(df_ids['name_norm'], df_ids['player_id']))
    print(f"Loaded {len(id_map)} player IDs from main file.")

    manual_name_to_id, manual_name_to_correct = parse_manual_mappings(MANUAL_MAP_FILE) #manuel directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True) #ensure output folder exists

    # save dictionaries to JSON for inspection (debug)
    with open(os.path.join(OUTPUT_FOLDER, 'manual_name_to_id.json'), 'w') as f:
        json.dump(manual_name_to_id, f, indent=2)
    with open(os.path.join(OUTPUT_FOLDER, 'manual_name_to_correct.json'), 'w') as f:
        json.dump(manual_name_to_correct, f, indent=2)
    print(f"Saved manual dictionaries to {OUTPUT_FOLDER}")

    # Process each
    for season in SEASONS:
        input_file = os.path.join(INPUT_FOLDER, f'players_stats_aggregated_{season}.csv')
        output_file = os.path.join(OUTPUT_FOLDER, f'id_data_{season}.csv')   # <-- changed filename

        # skip if output  exists
        if os.path.exists(output_file):
            print(f"Skipping {season} – output already exists: {output_file}")
            continue

        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            continue

        df = pd.read_csv(input_file)
        print(f"\nProcessing {season} – {len(df)} rows")

        # a column for the final player_id
        df['player_id'] = None

        # manual mappings
        for idx, row in df.iterrows():
            name = row['player']
            if name in manual_name_to_id:
                df.at[idx, 'player_id'] = manual_name_to_id[name]
                continue
            if name in manual_name_to_correct:
                corrected = manual_name_to_correct[name]
                norm = normalise_name(corrected)
                if norm in id_map:
                    df.at[idx, 'player_id'] = id_map[norm]
                continue
            norm = normalise_name(name)
            if norm in id_map:
                df.at[idx, 'player_id'] = id_map[norm]

        # count unmatched
        unmatched = df['player_id'].isna().sum()
        if unmatched:
            print(f"{unmatched} players not matched. See unmatched_{season}.txt")
            unmatched_names = df[df['player_id'].isna()]['player'].unique()
            with open(os.path.join(OUTPUT_FOLDER, f'unmatched_{season}.txt'), 'w', encoding='utf-8') as f:
                for name in unmatched_names:
                    f.write(name + '\n')
        else:
            print(f"All players matched!")

        # ensure player_id is integer type (not with decimal)
        df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce').astype('Int64')

        #player_id first
        cols = ['player_id'] + [c for c in df.columns if c != 'player_id']
        df = df[cols]

        df.to_csv(output_file, index=False)
        print(f"💾 Saved to {output_file}")

    print("\nAll seasons processed!")

if __name__ == "__main__":
    main()