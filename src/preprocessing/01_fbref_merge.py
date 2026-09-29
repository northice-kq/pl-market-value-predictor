import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # up 2 levels (so it runs back to the project root)
INPUT_FOLDER = os.path.join(project_root, 'data', 'raw', 'fbref')
OUTPUT_FOLDER = os.path.join(project_root, 'data', 'combined')

# Debug: list all files in INPUT_FOLDER
print(f"📁 Looking for files in: {INPUT_FOLDER}")
if os.path.exists(INPUT_FOLDER):
    all_files = os.listdir(INPUT_FOLDER)
    print(f"📄 Files found: {all_files}")
else:
    print(f"❌ Folder does not exist: {INPUT_FOLDER}")

SEASONS = ['2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026']

DROP_STANDARD = [
    'league',
    'born',
    'Playing Time_90s',
    'Performance_G+A',
    'Performance_CrdY',
    'Performance_CrdR',
    'Per 90 Minutes_G+A'
]

SHOOTING_COLS = [
    'Standard_Sh',
    'Standard_SoT',
    'Standard_SoT%',
    'Standard_Sh/90',
    'Standard_SoT/90',
    'Standard_G/Sh',
    'Standard_G/SoT'
]

MISC_COLS = [
    'Performance_CrdY',
    'Performance_CrdR',
    'Performance_2CrdY',
    'Performance_Fls',
    'Performance_Fld',
    'Performance_Off',
    'Performance_Crs',
    'Performance_Int',
    'Performance_TklW',
    'Performance_OG'
]

def integrate_season(season):
    """Merge standard, shooting, misc for one season and save integrated CSV."""
    print(f"\nProcessing {season}...")

    out_path = os.path.join(OUTPUT_FOLDER, f'fbref_{season}_integrated.csv')
    if os.path.exists(out_path):
        print(f"Combined file already exists: {out_path}, skipping")
        return

    std_path = os.path.join(INPUT_FOLDER, f'fbref_{season}_standard.csv')
    shoot_path = os.path.join(INPUT_FOLDER, f'fbref_{season}_shooting.csv')
    misc_path = os.path.join(INPUT_FOLDER, f'fbref_{season}_misc.csv')

    # Check if all files exist
    if not all(os.path.exists(p) for p in [std_path, shoot_path, misc_path]):
        print(f"Missing files for {season}, skipping.")
        return

    df_std = pd.read_csv(std_path)
    df_shoot = pd.read_csv(shoot_path)
    df_misc = pd.read_csv(misc_path)

    # Drop unwanted columns from standard
    cols_to_drop = [col for col in DROP_STANDARD if col in df_std.columns]
    df_std = df_std.drop(columns=cols_to_drop)

    # Select shooting columns (and also ensure they exist)
    shoot_cols_present = [col for col in SHOOTING_COLS if col in df_shoot.columns]
    if len(shoot_cols_present) != len(SHOOTING_COLS):
        missing = set(SHOOTING_COLS) - set(shoot_cols_present)
        print(f"Missing shooting columns: {missing}, will skip those.")
    df_shoot_sel = df_shoot[shoot_cols_present]

    # Select misc columns (als o ensure they exist)
    misc_cols_present = [col for col in MISC_COLS if col in df_misc.columns]
    if len(misc_cols_present) != len(MISC_COLS):
        missing = set(MISC_COLS) - set(misc_cols_present)
        print(f"Missing misc columns: {missing}, will skip those.")
    df_misc_sel = df_misc[misc_cols_present]

    df_integrated = pd.concat([df_std, df_shoot_sel, df_misc_sel], axis=1) # Concatenate horizontally


    os.makedirs(OUTPUT_FOLDER, exist_ok=True) # Ensure output directory exists
    df_integrated.to_csv(out_path, index=False)
    print(f"✅ Saved integrated file: {out_path} (columns: {len(df_integrated.columns)})")

def main():
    for season in SEASONS:
        integrate_season(season)
    print("\n🎉 All seasons integrated! Files are in 'data/combined/'.")

if __name__ == "__main__":
    main()