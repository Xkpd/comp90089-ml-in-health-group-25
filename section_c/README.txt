SECTION C — MODEL TRAINING
Group 25 | COMP90089 | generated 13 October 2026

START HERE

Open Section_C_Model_Training.ipynb. It contains the complete modelling code;
section_c_pipeline.py is a matching reusable/testable copy, not a required import.
Run the notebook from this section_c folder using a Python 3.12 environment with
the versions in requirements.txt. The code uses CPU XGBoost; no GPU is required.

Edit the configuration cell to point to B's CSV. The default is:
  ../data/clean_final_dataset.csv
Use B's COMPLETE cohort export BEFORE median imputation and BEFORE train/test
splitting. clean_full_final.csv is another possible input, but choosing it changes
the missingness-based population; document that decision. Do not use train_set*
or test_set* exports. Within-patient forward fill performed by B is not undone.
The notebook does not apply another patient-missingness filter.

The dataset is NOT included, and no real MIMIC models/results have been produced.
Software tests used explicitly synthetic patients with a reduced search.

INPUT CONTRACT

Required:
  subject_id: one unique patient per row, with one approved index extubation.
  combined_outcome: 0/1; 1 = death OR reintubation within 48 hours.
  Physiological predictors from the preceding six-hour window; retain NaNs.

Recommended audit fields, never used as predictors:
  death_within_48_hours, reintubation_within_48_hours
  followup_complete: 1 for fully observed negatives under the team's chosen policy.
If components are supplied, the combined label is checked against their OR.
If a negative has followup_complete != 1, the code stops. Absence of this flag
produces an explicit warning: an aggregated CSV cannot prove outcome observation.

For the proposal's temporal holdout, also provide:
  subject_id, extubation_time, anchor_year, anchor_year_group
These can be in the main CSV or ../data/cohort_metadata.csv, with one row per
subject_id. C merges by subject_id, never by row position. B's latest notebook
drops extubation_time and does not export anchor_year_group, so this metadata is
an additional necessary export unless already added to the final CSV.

The code converts each index extubation to a possible real-year interval using
the documented offset from anchor_year. Earlier non-overlapping intervals are
development; later intervals are test; intervals crossing the cutoff are listed
as ambiguous_time and excluded. This is a conservative temporal design, not
exact calendar-date reconstruction. The default cutoff is the start of the latest
observed anchor_year_group, chosen without looking at model performance. Inspect
partition COUNTS and decide a cutoff before fitting. There is no silent fallback
to a random split. split_mode='stratified' is an explicit protocol deviation.

Predictor formats recognised automatically:
  heart_rate_bin_0 ... heart_rate_bin_5, other B physiological variables;
  the same variables without a suffix, or with _mean, _std, _sd, _min, _max,
  _last, _slope, _count, _missing, optional _6h, or simply _6h;
  age/admission_age, gender/sex, first_careunit.
See feature_audit.csv after running. Unrecognised fields are NOT silently used.
For different names, set feature_columns to an explicit list and categorical_columns
appropriately. Categorical encodings, medians and scaling are learned in folds.
IDs, outcome components, discharge/death times, length of stay and anchor-year
metadata are blocked. Naming checks supplement, not replace, a data dictionary.

RSBI auto-detection: RSBI_bin_5, rsbi_bin_5, RSBI, rsbi, in that order.
If B uses another name, set rsbi_column. Do not silently substitute an all-window
average for the last-hour score. B's RR/VT ratio made from separate hourly averages
is labelled a proxy; it is not verified paired SBT measurement. No RSBI imputation
is used for the clinical comparison. The historical threshold 105 is contextual,
not a validated threshold for this proxy or the project's composite endpoint.

WHAT THE CODE DOES

1. Validates patient uniqueness, labels, metadata and selected feature names.
2. Creates ONE patient split manifest shared by every model.
3. Runs five outer folds and three inner folds on development patients only.
4. Tunes balanced logistic regression and XGBoost by inner AUROC. XGBoost's class
   ratio is calculated within each fit. Both receive the same predictors and folds.
5. Uses training-fold median imputation and missing indicators for both; logistic
   regression additionally standardises numerical inputs. Categorical fields are
   one-hot encoded. Completely missing training features remain represented.
6. Applies training-only cross-validated sigmoid calibration to both models. This
   is a declared addition to the proposal, motivated by class weighting and its
   stated calibration objective. It is assessed on untouched outer folds/test.
7. Saves fold metrics, OOF predictions, search results, curves and model settings.
8. Selects the primary model by mean outer-fold AUROC. Derives each model's final
   threshold from development OOF using the SAME rule: maximum specificity at
   sensitivity >= 0.80 (an exploratory default, not a clinical standard).
9. Refits on development patients, saves frozen models/thresholds, and exports LR
   coefficients and native XGBoost TreeSHAP summaries from development patients.
10. Leaves final evaluation OFF until the decisions are fixed with D.

Model-specific preprocessing is legitimate. Fair comparison requires the same
cohort, outcome, partitions, feature information, metrics and threshold rule;
it does not require applying logistic-regression scaling to a tree model.

EVALUATION AND HANDOFF TO D

The common evaluator provides AUROC, average precision (AP), prevalence,
sensitivity, specificity, PPV, NPV, confusion counts, Brier score and log loss.
AP is our chosen PR summary; it is not trapezoidal PR-AUC. Raw RSBI and random
uniform scores are not calibrated probabilities and receive no Brier/log-loss
interpretation. A training-prevalence probability baseline is also included.

Outer-fold metrics use a fixed 0.5 threshold. Metrics at OOF-selected thresholds
are labelled DESCRIPTIVE_NOT_VALIDATION because the same OOF data chose them.
Final test metrics evaluate the frozen thresholds. Report fold mean/SD as well as
pooled OOF plots; variability across folds is not a confidence interval.

For D: leave RUN_FINAL_TEST=False during model development. Once decisions are
frozen, change it to True to export the common held-out predictions, comparison
table, ROC/PR/calibration plots, paired patient-bootstrap intervals, paired XGB-LR
differences, and basic error/subgroup summaries. Every model's RSBI comparison
uses exactly the same RSBI-available test patients. No threshold is reselected on
the test set. Default bootstrap size is 1,000; valid replicate counts are reported.
These intervals condition on the fitted models/test cohort, not a repeat of the
entire training procedure. Subgroup results are exploratory, not a fairness proof.

An unchanged rerun reuses the frozen analysis/results after test access. An
interrupted final evaluation can resume with identical saved models and settings.
Changed settings/data are blocked in that run once test evaluation has started.
For an independent reproduction, use a fresh output folder with the SAME analysis
settings; do not use repeated test access to guide model/feature choices.
When reloading trained_models.joblib, execute the notebook's definition cells first
(the custom estimator must be defined). Load only trusted local model files.

Key outputs under outputs/main_analysis/:
  patient_split_manifest.csv                 shared patient assignments
  feature_audit.csv                          exact included/excluded columns
  development_feature_missingness.csv        development-only descriptive audit
  outer_fold_summary_mean_sd.csv             primary validation summary
  development_oof_predictions.csv            aligned OOF predictions for D
  frozen_analysis.json                       settings, hashes, thresholds, choice
  trained_models.joblib                      fitted development models
  logistic_base_coefficients.csv             associations, not causal effects
  xgboost_base_treeshap.csv/.png              base log-odds explanations
  FINAL_*                                   created only by final evaluation

PROPOSAL / RUBRIC LIMITS THAT THIS CODE CANNOT FIX

The proposal specifies six hourly averages plus variability, first invasive episode
and first extubation, unplanned-extubation and DNR/DNI exclusions, and a >50%
missingness exclusion. B's later extraction and the proposed whole-six-hour
aggregation differ in places. This code accepts the TEAM-APPROVED CSV and records
the choices; it does not claim those changes are identical to the proposal.

Six-hour means alone do not capture temporal trends. The code does not fabricate
within-hour variability, ventilation-episode linkage, or missing follow-up from
an aggregated table. Existing _std/_slope columns can be included if B computed
them correctly from data available before the decision. All-window means and
hourly means are accepted as different input designs, not mixed automatically.

B must still supply extraction SQL, event/episode and inclusion definitions,
availability-time rules for labs, outcome-follow-up policy, exclusion flow/counts,
and retained/excluded missingness summaries. The final master notebook must
integrate those extraction steps to satisfy end-to-end reproducibility. This
standalone C notebook starts at a CSV and cannot by itself satisfy that criterion.

The study is retrospective and single-centre. An AUROC near 0.80 is an aspiration,
not a code test or a grade guarantee. Neither high discrimination nor SHAP
establishes clinical benefit, safe extubation decisions, or causality.

LITERATURE CHECK

Zhao et al. (2021) used four-hour summary measurements, a random development/test
split, and CatBoost as their principal model. Their outcome also included NIV.
We retain the proposal's six-hour window, LR/XGBoost, nested CV and temporal test;
therefore this is an adaptation, not a replication or direct performance contest.
https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2021.676343/full

Grinsztajn et al. (2022) supports considering boosted trees for tabular data;
it does not guarantee XGBoost will outperform LR in this extubation cohort.
https://papers.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract.html

Trivedi et al. (2022) evaluates RSBI for extubation SUCCESS. Sensitivity/specificity
depend on which outcome is positive, threshold, population and measurement setting.
Do not copy the proposal's 'catches 58% of failures' as this study's observed result.
https://journal.chestnet.org/article/S0012-3692%2821%2901257-5/fulltext

MIMIC year intervals:
https://mimic.mit.edu/docs/iv/modules/hosp/patients.html
Nested CV:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
Calibration and threshold selection:
https://scikit-learn.org/stable/modules/calibration.html
https://scikit-learn.org/stable/modules/classification_threshold.html

AI USE

This modelling code and accompanying explanations were generated with OpenAI
Codex and require the team's review and understanding. Do not reuse the proposal's
old declaration stating that AI did not generate code. The assignment PDF supplied
here restricts AI code generation, while you said the teaching team permits AI with
disclosure. Resolve that discrepancy with the current course guidance before
submitting. This note records provenance; it does not itself establish permission.

SOFTWARE VERIFICATION

Tests are in test_section_c.py and create synthetic CSVs in temporary directories.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q test_section_c.py
The environment variable avoids unrelated system pytest plugins; it does not
change the model tests. It is not needed to run the notebook itself.
The notebook is also checked cell by cell on synthetic inputs before delivery.
