# Premier League Market Value Predictor

> **Read the full report →** [`report.md`](report.md)
> *Deep-dive into the methodology, findings, and limitations. The README below is a quick summary.*

Predicting end-of-season Transfermarkt market values for Premier League outfield players from their season-wise performance statistics and value momentum, trained on five seasons *(2021–22 to 2025–26)*.

## Contents
- [Overview](#overview)
- [Key Findings](#key-findings)
- [Visualizations](#visualizations)
- [Data Sources](#data-sources)
- [Quick Reproducibility](#quick-reproducibility)
- [Limitations](#limitations)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Disclaimer](#disclaimer) 
- [Appendix](#appendix)


## Overview
Transfermarkt market values are community-driven estimates rather than fixed formulas. While the platform outlines key factors like reputation, marketing appeal, injury history, and situational context, it does not publicly quantify or weight them *(more on [Transfermarkt](https://www.transfermarkt.com/navigation/mwdefinition))*. 
This project asks how much of that variation can be explained by **observable data alone**, namely, in-season performance statistics and prior value momentum.

Trained on 2,044 player-seasons across five seasons, a Ridge regression model achieves **R² = 0.942** on the held-out 2025–26 season, with a 22.8% RMSE improvement over the strongest naive baseline. 
The results also expose structural biases: the model underpredicts the top of the market and is less reliable for very young players, which align with the reputation and narrative factors it cannot see.

## Key Findings

### 1. Prior market value is the strongest predictor

The momentum features carry more weight than the other features combined. The signal is almost entirely `log1p_mid_mv` *(+0.824)*, while `log1p_pre_mv` collapsed to *+0.014*, as the mid-season checkpoint already reflects everything the pre-season value knew.

Market value is highly autocorrelated. The best predictor of a player's June value is therefore their December value.

### 2. Age effect is subtler than it appears

Both `age` (−0.071) and `age²` (−0.172) are negative, which seems to contradict the inverted-U relationship that peaks at 22–27. 
Since current value already encodes the market's age premium, the age terms capture only the **residual** effect. 
The quadratic now dominates. The relationship is sharper than a simple inverted-U, i.e.,
given two players of equal current value, the older one depreciates more.

### 3. The market penalises attempt volume over completed actions

Four of the largest negative coefficients belong to **attempt** actions, recorded regardless of outcome: `Standard_Sh` *(−0.068)*, `Performance_Int` *(−0.032)*, `Performance_Crs` *(−0.021)*, and `LongB` *(−0.011)*. 
Stats recorded only on success are positive: `AvgP` *(+0.057)*, `KeyP` *(+0.009)*, `Performance_TklW` *(+0.008)*.

The market prices outcomes. This aligns with Vecer (2018), who found open crosses convert at roughly half the rate of final-third entries.

### 4. Average passes is the top raw performance statistic

Among raw pitch-based performance features, excluding composite metrics like WhoScored `Rating` and context features like playing time, 
`AvgP` (average passes per game, *+0.057*) carries the largest coefficient, followed by `Standard_Sh/90` *(+0.046)* and `PS%` *(+0.024)*.

Once prior value is controlled for, involvement in possession and build-up play is the most valuable single performance signal the market prices in.

*Full analysis in [report.md #Key Findings](report.md#key-findings).*

## Visualizations

**What drives predictions** - `log1p_mid_mv` dominates every other feature by a wide margin. `Rating` is the largest non-momentum predictor, and `age_2` (age²) is the largest negative.

![Ridge Coefficients](reports/figures/03_coefficients.png)

**Predicted vs actual** - tight cluster along the diagonal (perfect prediction), scatter widens at both extremes. Cheap young prospects sit above the diagonal, while premium players sit below it.

![Predicted vs Actual](reports/figures/01_pred_vs_actual.png)

*More figures in [report.md #Visualizations](report.md#visualizations).*

## Data Sources

Three sources contribute to the final dataset: 

| Source                                                                                | Contribution | Collection                                     |
|---------------------------------------------------------------------------------------|--------------|------------------------------------------------|
| [FBref](https://fbref.com/en/)                           | Standard, shooting, misc, and keeper statistics | Scraped per season via `fbref_scraper_all.py` |
| [WhoScored](https://www.whoscored.com/)                                               | Passing statistics and player ratings | Manually collected / `whoscored_scraper.py`                        |
| [Transfermarkt via Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) | Market value history | Kaggle dataset (Cariboo, 2024)                 |

The dataset contains five seasons of data, 2,044 outfield player-seasons after filtering. 

Excluded from the dataset:
- Goalkeepers 
- Players with under 200 minutes or under 3 appearances
- Players missing market value history for at least one of the three required checkpoints (pre-season, mid-season, or previous December)

For the full methodology, see [Data and Methodology](report.md#data-and-methodology)

## Quick Reproducibility

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Run the preprocessing pipeline (raw data → final dataset)
```bash
py src/raw_processing/process_mv.py
py src/scraping/fbref_scraper_all.py
py src/scraping/whoscored_scraper.py
py src/preprocessing/01_fbref_merge.py
py src/preprocessing/02_merge_whoscored_to_fbref.py
py src/preprocessing/03_aggregate_player_season.py
py src/preprocessing/04_add_players_id.py
py src/preprocessing/05_merge_mv.py
py src/preprocessing/FINAL_combine.py 
```

3. Run the ML pipeline (dataset → trained model)
```bash
py src/ml/01_prepare.py
py src/ml/02_split.py
py src/ml/03_ridge.py

```

4. To predict a player, fill `data/input/players_template.csv` with the required fields (age, position, pre-season value, mid-season value), and any optional stats, then run
```bash
py src/ml/07_PREDICT.py
```
Predictions are saved to `reports/model_output/predictions.csv`.

*Note I: Scraping WhoScored scripts may be rate-limited depending on the time of day. 
Pre-scraped raw files are included in `data/raw/`.
You can skip the scraping steps and start from `01_fbref_merge.py`.*

*Note II: Player name matching required a small number of manual corrections, stored in `player_name_mappings.txt`.*

*Note III: The rest of the files in `src/ml` are optional analyses documented in the report*

> **<span style="color:red">HIGHLY RECOMMENDED:</span>** Read the detailed version of **[Reproducibility](report.md#reproducibility)** in the full report. It covers input validation, dependency chains, and known limitations in more depth.

## Limitations

- **Temporal drift.** The model under-predicts the top of the market because 2025–26 values grew faster than the historical trend it learned. All 15 of the worst errors are under-predictions.
- **Youth over-prediction.** The 15 largest upward gaps are all aged 25 or under, with a median age of 22. The age curve rewards youth too aggressively relative to the market.
- **Sparse tails.** Under 5% of training rows fall outside age 20–32 or the €2M–€70M value band. Extreme inputs involve extrapolation.
- **Reputation and context are invisible.** Garnacho's attitude problems, Tottenham's 17th-place finish, Szoboszlai's elite-tier jump, all trace to factors the model cannot see.
- **Ridge assumptions.** Linear effects, no automatic interactions, global coefficients across the whole market.

*Full analysis in [`report.md` #Limitations](report.md#limitations).*

## Repository Structure

```
pl_market_predictor/
├── src/
│ ├── scraping/         # FBref and WhoScored scrapers
│ ├── raw_processing/   # Market value wide-format processing
│ ├── preprocessing/    # Merge, aggregate, align player data
│ └── ml/               # Feature engineering, training, prediction
├── data/
│ ├── raw/              # Raw scraped files (committed)
│ ├── input/            # Prediction template and outputs
│ └── ...               # Derived data (not committed)
├── reports/
│ ├── figures/          # Four visualizations
│ └── model_output/     # Predictions, valuations, baselines
├── report.md           # Full writeup
├── requirements.txt
└── README.md
```

## Tech Stack

Python · pandas · numpy · scikit-learn · Selenium · BeautifulSoup · matplotlib · joblib

## Disclaimer

Scraping code is provided for educational purposes. 
Users should respect the terms of service of each site. 
Raw scraped files are committed for reproducibility; redistribution is not intended for commercial use.

## Appendix

See the [full 43-feature list](report.md#a-full-feature-list) in the report's Appendix