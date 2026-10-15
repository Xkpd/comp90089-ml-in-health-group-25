# COMP90089 Group 25

Predicting 48-hour extubation failure using MIMIC-IV v3.1.

- `data/data.ipynb`: cohort and feature extraction.
- `models/Model Training.ipynb`: modelling and evaluation.

Run the modelling notebook top to bottom once (5 outer / 3 inner folds). All analysis choices are fixed in advance, so the models are built, frozen and then scored on the temporal test set in the same run. A rerun with unchanged settings reloads the saved results; a rerun with changed settings or data is rejected after the test has been scored. Package versions are printed and saved.

The primary analysis uses `average_6h_filtered_dataset.csv`. Prespecified sensitivities use the full cohort and last-hour summaries on the same main-cohort patients. Coverage is validation-only. The averaged CSVs include the temporal metadata needed for splitting.

Development data are used only to build the models and for variable importance. All results tables and figures come from the temporal test set. Report outputs are in `models/outputs/`:

- `study_timeline.png`, `input_checks.csv`, `model_selection.csv` (Methods).
- `table1_cohort.csv`, `FINAL_analysis_comparison.csv` (Table 2), `FINAL_auroc_forest.png`, `FINAL_calibration_decision_curve.png`.
- `FINAL_rsbi_comparison.csv`, `FINAL_sensitivity_analyses.csv`, `FINAL_decision_curve.csv`.
- `filtered_six_hour/interpretation_lr_shap.png` (variable importance, development patients).
- `filtered_six_hour/FINAL_*.csv`: paired XGBoost − LR difference, deaths vs reintubations, error summary, subgroups.

Files ending in `_predictions.csv` are patient-level and must not be shared.

For submission, combine extraction and modelling in `ML4Health_report_25.ipynb` and complete the generative AI declaration.
