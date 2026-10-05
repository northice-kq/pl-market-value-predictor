# Changelog

## v1.1 / 2026-10-03

**Fixed**
- WhoScored merge was zero-filling unmatched players across six passing and rating columns (`KeyP`, `AvgP`, `PS%`, `LongB`, `ThrB`, `Rating`), corrupting those features.
- Leftout report in `02_merge_whoscored_to_fbref.py` was never written when unmatched players existed.

**Impact**
- Test R² improved from *0.920* to *0.942*.
- `Rating` coefficient changed from *−0.003* to *+0.116*. WhoScored's composite score is now a real predictor.
- Finding reversed: the market penalises attempt volume (not just crosses)
- Data file renamed to `final_data_v1.1.csv`.

## v1.0 /

Initial release.