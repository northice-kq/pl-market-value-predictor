import soccerdata as sd
import pandas as pd
import os
import time

SEASONS = ['2021-2022', '2022-2023', '2023-2024',  '2024-2025', '2025-2026'] # Season list
STAT_TYPES = ['standard', 'shooting', 'misc', 'keeper'] # Stat types


def flatten_columns(df):
    """Flatten multi-level columns from soccerdata into single-level."""
    if isinstance(df.columns, pd.MultiIndex):
        new_columns = []
        for col in df.columns:
            parts = [str(p).strip() for p in col if str(p).strip() and 'Unnamed' not in str(p)]
            if len(parts) > 1:
                new_columns.append('_'.join(parts))
            elif len(parts) == 1:
                new_columns.append(parts[0])
            else:
                new_columns.append('_'.join([str(p) for p in col]))
        df.columns = new_columns
    return df


def scrape_fbref_season(season, stat_type, max_retries=3):
    """Scrape a single season and stat type, with retries."""
    print(f"\nScraping {stat_type} stats for {season}...")

    for attempt in range(1, max_retries + 1):
        try:
            fbref = sd.FBref(leagues="ENG-Premier League", seasons=season, no_cache=True)
            df = fbref.read_player_season_stats(stat_type=stat_type)
            df = df.reset_index()
            df = flatten_columns(df)

            unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)

            print(f"{season} {stat_type}: {len(df)} players scraped")
            return df

        except Exception as e:
            print(f"Attempt {attempt} failed for {season} {stat_type}: {e}")
            if attempt < max_retries:
                wait_time = 5 * attempt
                print(f"   Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed for {season} {stat_type}")
                return None


def scrape_all_data():
    """Scrape all seasons and stat types, skipping existing files."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # make sure to save at data dir
    project_root = os.path.dirname(script_dir)
    data_folder = os.path.join(project_root, 'data', 'raw','fbref')
    os.makedirs(data_folder, exist_ok=True)

    for season in SEASONS:
        for stat_type in STAT_TYPES:
            filename = os.path.join(data_folder, f'fbref_{season}_{stat_type}.csv')

            if os.path.exists(filename): # Skip if file already exists
                print(f"Skipping {season} {stat_type} - file already exists")
                continue

            df = scrape_fbref_season(season, stat_type)

            if df is not None:
                df.to_csv(filename, index=False)
                print(f"Saved to {filename}")

            time.sleep(1)

    print("\nAll scraping complete!")


if __name__ == "__main__":
    print("FBref Multi-Season Scraper (using soccerdata)")
    print("=" * 60)
    scrape_all_data()