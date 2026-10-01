import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
INPUT_FOLDER = os.path.join(project_root, 'data', 'combined')
OUTPUT_FOLDER = os.path.join(project_root, 'data', 'combined')

SEASONS = ['2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026']

# to sum
SUM_COLS = [
    'Playing Time_MP', 'Playing Time_Starts', 'Playing Time_Min',
    'Performance_Gls', 'Performance_Ast', 'Performance_G-PK', 'Performance_PK', 'Performance_PKatt',
    'Standard_Sh', 'Standard_SoT',
    'Performance_CrdY', 'Performance_CrdR', 'Performance_2CrdY',
    'Performance_Fls', 'Performance_Fld', 'Performance_Off',
    'Performance_Crs', 'Performance_Int', 'Performance_TklW', 'Performance_OG'
]

# to average, weighted by minutes
WEIGHTED_AVG_COLS = ['KeyP', 'AvgP', 'PS%', 'LongB', 'ThrB', 'Rating']

# keep from row
CAT_COLS = ['season', 'team', 'player', 'nation', 'pos', 'age']

# compute after aggregation
DERIVED_COLS = {
    'Per 90 Minutes_Gls': ('Performance_Gls',),
    'Per 90 Minutes_Ast': ('Performance_Ast',),
    'Per 90 Minutes_G-PK': ('Performance_G-PK',),
    'Standard_Sh/90': ('Standard_Sh',),
    'Standard_SoT/90': ('Standard_SoT',),
    'Standard_G/Sh': ('Performance_Gls', 'Standard_Sh'),
    'Standard_G/SoT': ('Performance_Gls', 'Standard_SoT'),
    'Standard_SoT%': ('Standard_SoT', 'Standard_Sh')
}


def aggregate_season(input_file, output_file):
    df = pd.read_csv(input_file)
    # replace empty strings with NaN
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    # convert and fill NaN with 0
    numeric_cols = SUM_COLS + WEIGHTED_AVG_COLS + ['Playing Time_Min']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['Playing Time_Min'] = pd.to_numeric(df['Playing Time_Min'], errors='coerce')

    # group by player
    grouped = df.groupby('player', as_index=False)

    # sum cumulative stats
    agg_df = grouped[SUM_COLS].sum()

    first_occ = df.groupby('player').first()[['nation', 'pos', 'age']].reset_index()  # age, pos, nation are the same
    agg_df = agg_df.merge(first_occ, on='player', how='left')

    def get_teams(group):
        # sort by minutes descending
        sorted_group = group.sort_values('Playing Time_Min', ascending=False)
        primary_team = sorted_group.iloc[0]['team']
        secondary_team = sorted_group.iloc[1]['team'] if len(sorted_group) > 1 else ''
        return pd.Series({'primary_team': primary_team, 'secondary_team': secondary_team})

    team_info = grouped.apply(get_teams).reset_index()
    agg_df = agg_df.merge(team_info, on='player', how='left')
    agg_df['team'] = agg_df['primary_team']
    agg_df = agg_df.drop(columns=['primary_team'])
    agg_df['season'] = df['season'].iloc[0]

    # wighted average for per-game stats
    def weighted_avg(group, col, weight_col='Playing Time_Min'):
        total_weight = group[weight_col].sum()
        if total_weight == 0:
            return 0.0
        return (group[col] * group[weight_col]).sum() / total_weight

    weighted_avg_dict = {}
    for col in WEIGHTED_AVG_COLS:
        if col in df.columns:
            # weighted sum of value * minutes
            weighted_sum = (df[col] * df['Playing Time_Min']).groupby(df['player']).sum()
            total_minutes = df.groupby('player')['Playing Time_Min'].sum()
            weighted_avg_series = (weighted_sum / total_minutes).fillna(0)
            weighted_avg_dict[col] = weighted_avg_series
        else:
            print(f"Warning: {col} not found, skipping")

    # convert to dataframe and merge
    if weighted_avg_dict:
        weighted_avg_df = pd.DataFrame(weighted_avg_dict).reset_index().rename(columns={'index': 'player'})
        agg_df = agg_df.merge(weighted_avg_df, on='player', how='left')
    for col in WEIGHTED_AVG_COLS:
        if col in agg_df.columns:
            agg_df[col] = agg_df[col].round(2)

    # For any missing columns either not in df or not in dict, fill with 0, cuz this is how Fbref give(Fbref give 0, cuz 0/0)
    for col in WEIGHTED_AVG_COLS:
        if col not in agg_df.columns:
            agg_df[col] = 0.0

    #derived stats
    for new_col, numerator_tuple in DERIVED_COLS.items():
        numerator = numerator_tuple[0]  # first element is always the primary numerator
        if numerator not in agg_df.columns:
            print(f"Numerator {numerator} missing for {new_col}")
            agg_df[new_col] = 0.0
            continue

        if new_col.endswith('/90') or new_col.startswith('Per 90 Minutes'):
            #numerator * 90 / total minutes
            agg_df[new_col] = ((agg_df[numerator] * 90 / agg_df['Playing Time_Min']).fillna(0).replace([float('inf'), float('-inf')], 0).round(4))   # replace nan and empty str with 0 before rounding
        else:
            if len(numerator_tuple) > 1:
                denom = numerator_tuple[1]
                if denom in agg_df.columns:
                    agg_df[new_col] = (agg_df[numerator] / agg_df[denom]).fillna(0).replace([float('inf'), float('-inf')], 0).round(4)
                else:
                    print(f"Denominator {denom} missing for {new_col}")
                    agg_df[new_col] = 0.0
            else:
                print(f"No denominator specified for {new_col}")
                agg_df[new_col] = 0.0

    # special case: Per 90 Minutes_G+A-PK = (G-PK + Ast) * 90 / minutes
    agg_df['Per 90 Minutes_G+A-PK'] = ((agg_df['Performance_G-PK'] + agg_df['Performance_Ast']) * 90
            / agg_df['Playing Time_Min']).fillna(0).replace([float('inf'), float('-inf')], 0).round(4)


    # For ratios, replace inf with 0
    for col in DERIVED_COLS.keys():
        agg_df[col] = agg_df[col].replace([float('inf'), float('-inf')], 0).fillna(0)

    # all columns
    final_columns = ['season', 'team', 'secondary_team', 'player', 'nation', 'pos', 'age', 'Playing Time_MP', 'Playing Time_Starts', 'Playing Time_Min', 'Performance_Gls', 'Performance_Ast', 'Performance_G-PK', 'Performance_PK', 'Performance_PKatt', 'Per 90 Minutes_Gls', 'Per 90 Minutes_Ast', 'Per 90 Minutes_G-PK', 'Per 90 Minutes_G+A-PK', 'Standard_Sh', 'Standard_SoT', 'Standard_SoT%', 'Standard_Sh/90', 'Standard_SoT/90', 'Standard_G/Sh', 'Standard_G/SoT', 'Performance_CrdY', 'Performance_CrdR', 'Performance_2CrdY', 'Performance_Fls', 'Performance_Fld', 'Performance_Off', 'Performance_Crs', 'Performance_Int', 'Performance_TklW', 'Performance_OG', 'KeyP', 'AvgP', 'PS%', 'LongB', 'ThrB', 'Rating']

    # keep only existing columns
    final_columns = [col for col in final_columns if col in agg_df.columns]
    agg_df = agg_df[final_columns]

    # filter out low min non-GK players
    if 'pos' in agg_df.columns and 'Playing Time_MP' in agg_df.columns and 'Playing Time_Min' in agg_df.columns:
        mask = (agg_df['pos'] == 'GK') | ((agg_df['Playing Time_MP'] >= 3) & (agg_df['Playing Time_Min'] >= 200))
        original_count = len(agg_df)
        agg_df = agg_df[mask]
        removed = original_count - len(agg_df)
        if removed > 0:
            print(f"   Removed {removed} non‑GK players with <3 apps or <200 mins.")

    # Save
    agg_df.to_csv(output_file, index=False)
    print(f"Aggregated {len(agg_df)} players for {input_file} -> {output_file}")




def main():
    print("Missing WhoScored values will be filled with 0.")
    for season in SEASONS:
        input_file = os.path.join(INPUT_FOLDER, f'players_stats_combined_{season}.csv')
        output_file = os.path.join(OUTPUT_FOLDER, f'players_stats_aggregated_{season}.csv')
        if os.path.exists(input_file):
            aggregate_season(input_file, output_file)
        else:
            print(f"Input file not found: {input_file}")


if __name__ == "__main__":
    main()