# COMP90089 Group 25

Predicting 48-hour extubation failure using MIMIC-IV v3.1.

- `data/report(4).ipynb`: cohort and feature extraction.
- `models/Model Training.ipynb`: modelling and evaluation.
- `models/requirements.txt`: pinned modelling dependencies, tested with Python 3.12.

Install the dependencies, then run the modelling notebook from top to bottom. Edit only `CONFIG`.

- `check`: tests the complete pipeline on synthetic data.
- `trial` (default): reduced tuning on real development patients; test sets remain closed.
- `main`: full 5×3 nested CV. Enable `run_final_test` after all choices are fixed.

The notebook runs the filtered six-hour main analysis, full-cohort sensitivity analysis and an optional filtered last-hour ablation. Each has a separate folder under `models/outputs/<mode>/`, with combined comparison tables in the mode folder.

Real runs use `data/clean_final_dataset.csv` and `data/clean_full_final.csv` before median imputation or splitting. Both need one row per patient and temporal metadata (`extubation_time`, `anchor_year`, `anchor_year_group`). The last-hour ablation requires matching hourly-bin predictors. The missingness filter and other eligibility rules belong to the extraction stage.

For submission, combine extraction and modelling into the required `ML4Health_report_25.ipynb`. Declare AI assistance as required by the course.
