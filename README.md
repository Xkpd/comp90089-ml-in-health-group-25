# COMP90089 Group 25

Predicting 48-hour extubation failure using MIMIC-IV v3.1.

- `data/data.ipynb`: cohort and feature extraction.
- `models/Model Training.ipynb`: modelling and evaluation.

Run the modelling notebook top to bottom (5 outer / 3 inner folds). Run first with `run_final_test=False` and check the development results; then set `run_final_test=True` and rerun to score the temporal test with the saved models. Each run records package versions.

The primary analysis uses `average_6h_filtered_dataset.csv`. Prespecified sensitivities use the full cohort and last-hour summaries on the same main-cohort patients. Coverage is validation-only. The averaged CSVs include the temporal metadata needed for splitting.

Each section prints its key tables and figures and saves them in `models/outputs/` (combined tables and figures) and `models/outputs/<analysis>/` (per-analysis CSVs):

- Study timeline and Table 1 (development patients by outcome and missingness-filter membership).
- Development performance with 95% CIs, AUROC forest plot, primary calibration and decision curve.
- Paired XGBoost–LR and six-hour–last-hour differences, missingness-filter robustness, subgroups.
- LR coefficient and XGBoost SHAP figure (primary development patients).
- After the test is opened: development-versus-test Table 1, test performance and forest plot, test calibration and decision curve, RSBI comparison, exploratory subgroups and errors.

For submission, combine extraction and modelling in `ML4Health_report_25.ipynb` and complete the generative AI declaration.
