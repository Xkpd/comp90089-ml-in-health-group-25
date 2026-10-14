# COMP90089 Group 25

Predicting 48-hour extubation failure using MIMIC-IV v3.1.

- `data/data.ipynb`: cohort and feature extraction.
- `models/Model Training.ipynb`: logistic regression, XGBoost and evaluation. Package versions and settings are listed inside.

Open the modelling notebook and run it from top to bottom:
1. Use `mode="trial"` to test the real-data workflow with reduced tuning.
2. Use `mode="main"` for 5 outer / 3 inner folds.
3. Set `run_final_test=True` only after all analysis choices are fixed.

The main analysis uses `data/average_6h_filtered_dataset.csv`. Sensitivity analyses use `average_6h_full_dataset.csv` and the last available hourly summaries in `clean_full_final.csv`. `average_6h_coverage.csv` validates missingness and cohort membership; it is not a model input.

**Temporal metadata are still needed:** add `extubation_time`, `anchor_year` and `anchor_year_group` to both averaged CSVs, or supply a CSV keyed by `subject_id` through `metadata_csv`. Include `stay_id` to verify the index event. The current exports omit these fields.

Outputs go to `models/outputs/<mode>/`, including metrics, confidence intervals, calibration and XGBoost SHAP importance plots. Small test-event counts produce a warning; they do not remove patients or skip evaluation.

For submission, combine extraction and modelling in `ML4Health_report_25.ipynb` and complete the generative AI declaration.
