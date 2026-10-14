# COMP90089 Group 25

Predicting extubation outcomes using MIMIC-IV data.

- `data/report(4).ipynb`: B's extraction and preparation.
- `models/Model Training.ipynb`: all Section C code, settings, software checks, training and evaluation. No local Python helper files or export step are needed.

Open the modelling notebook and run cells from top to bottom. Dependencies are pinned in its setup cell. `CONFIG` is the only editable analysis configuration:

- `execution.mode="check"`: test the whole pipeline, including final evaluation, with synthetic data. These results are not research findings.
- `execution.mode="trial"`: reduced tuning on real development data; the real final holdout stays closed.
- `execution.mode="main"`: full development. Enable `execution.run_final_test` only after fixing the main analysis.

Outputs are saved separately under `models/outputs/`. Real-data runs require the complete CSV before median imputation and splitting, plus temporal metadata (`extubation_time`, `anchor_year`, `anchor_year_group`). Missing temporal metadata stops the run; there is no automatic random-split fallback.

For the group's submission, combine B's extraction with this modelling notebook and use the required `ML4Health_report_25.ipynb` filename. Declare AI assistance as required by the course.
