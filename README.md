# COMP90089 — Group 25

Group project code for Machine Learning for Health: predicting extubation outcomes using MIMIC-IV data.

## Contents

- `data/report(4).ipynb`: cohort extraction and data preparation notebook using Google BigQuery.
- `section_c/Section_C_Model_Training.ipynb`: model training notebook.
- `section_c/section_c_pipeline.py`: reusable model training pipeline.
- `section_c/test_section_c.py`: software tests using synthetic data.
- `section_c/README.txt`: detailed input requirements, workflow, limitations, and AI-use disclosure.
- Project proposal and presentation PDFs in the repository root.

## Run the modelling code

Use Python 3.12 and a dedicated environment. From the repository folder:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r section_c/requirements.txt
```

Open `section_c/Section_C_Model_Training.ipynb` in your notebook editor, select this environment, and run it with `section_c/` as the working directory. Read `section_c/README.txt` before running the analysis.

Place the approved cohort export at `data/clean_final_dataset.csv`. The temporal holdout also requires the metadata described in `section_c/README.txt`, either in that CSV or `data/cohort_metadata.csv`. These data files are not included in GitHub.

The extraction notebook has separate BigQuery dependencies. Configure your own authorized Google Cloud project and authentication, and ensure you have the required MIMIC-IV access before using it.

To run the existing synthetic-data tests:

```sh
cd section_c
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q test_section_c.py
```

## Work with the group

1. Accept the repository invitation, then clone it in GitHub Desktop.
2. Fetch and pull the latest changes before starting work.
3. Create a branch for your changes, commit with a descriptive message, and push it.
4. Open a pull request for group review before merging into `main`.

## Data and notebook outputs

Share code through this repository. Keep patient datasets, credentials, generated predictions, and trained models outside Git. The `.gitignore` excludes common data files and output folders. Clear notebook outputs before committing so patient-level tables are not embedded in `.ipynb` files.

Saved outputs were cleared from the extraction notebook for the initial upload. A local backup is stored in the ignored `.local-backups/` folder on the original computer.
