"""Software checks with SYNTHETIC data; no MIMIC records or research results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import section_c_pipeline as c


def synthetic_inputs(folder, n=360):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    severity = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(1.4 - .8 * severity))).astype(int)
    death = (y & (rng.random(n) < .3)).astype(int)
    df = pd.DataFrame({"subject_id": np.arange(10000000, 10000000 + n).astype(str),
        "stay_id": np.arange(30000000, 30000000 + n), "combined_outcome": y,
        "death_within_48_hours": death, "reintubation_within_48_hours": y,
        "age": rng.integers(20, 90, n), "gender": rng.choice(["F", "M"], n),
        "followup_complete": 1, "hospital_expire_flag": death, "los_icu": rng.uniform(1, 20, n)})
    for b in range(6):
        df[f"heart_rate_bin_{b}"] = 80 + 8 * severity + rng.normal(0, 6, n)
        df[f"resp_rate_bin_{b}"] = 18 + 2 * severity + rng.normal(0, 2, n)
        df.loc[rng.random(n) < .15, f"heart_rate_bin_{b}"] = np.nan
    df["lactate_bin_0"] = np.nan  # entirely missing feature: pipeline must still work
    df["RSBI_bin_5"] = np.clip(70 + 15 * severity + rng.normal(0, 10, n), 1, None)
    df.loc[rng.random(n) < .3, "RSBI_bin_5"] = np.nan
    groups = np.repeat("2011 - 2013", n).astype(object)
    groups[-n // 4:] = "2017 - 2019"
    groups[:12] = "2014 - 2016"  # plus 1-year offset straddles cutoff: excluded
    meta = pd.DataFrame({"subject_id": df.subject_id, "anchor_year": 2140,
        "anchor_year_group": groups, "extubation_time": "2140-06-15 14:00:00"})
    meta.loc[:11, "extubation_time"] = "2141-06-15 14:00:00"
    df.to_csv(folder / "synthetic_cohort.csv", index=False)
    meta.to_csv(folder / "synthetic_metadata.csv", index=False)
    return {**c.DEFAULT_CONFIG, "data_csv": str(folder / "synthetic_cohort.csv"),
        "metadata_csv": str(folder / "synthetic_metadata.csv"), "output_dir": str(folder / "synthetic_outputs"),
        "outer_folds": 2, "inner_folds": 2, "quick_test": True, "synthetic_data": True,
        "xgb_threads": 1, "bootstrap_replicates": 20}


def test_input_and_temporal_safeguards(tmp_path):
    cfg = synthetic_inputs(tmp_path)
    df, nums, cats, rsbi, audit, _ = c.load_cohort(cfg)
    assert not set(c.OUTCOME_COMPONENTS + ["hospital_expire_flag", "los_icu", "stay_id", "subject_id", "combined_outcome"]) & set(nums + cats)
    manifest, dev, test, note = c.make_split(df, cfg)
    assert note["cutoff"] == 2017
    assert len(manifest[manifest.partition.eq("ambiguous_time")]) == 12
    assert not set(df.iloc[dev].subject_id) & set(df.iloc[test].subject_id)
    assert manifest.loc[dev, "possible_year_upper"].max() < manifest.loc[test, "possible_year_lower"].min()
    with pytest.raises(ValueError, match="Temporal holdout needs"):
        c.make_split(df.drop(columns="anchor_year"), cfg)
    with pytest.raises(ValueError, match="Forbidden"):
        c.load_cohort({**cfg, "feature_columns": ["age", "death_within_48_hours"]})
    broken = pd.read_csv(cfg["data_csv"])
    broken.loc[0, "combined_outcome"] = 1 - broken.loc[0, "combined_outcome"]
    broken.to_csv(cfg["data_csv"], index=False)
    with pytest.raises(ValueError, match="disagrees"):
        c.load_cohort(cfg)


def test_preprocessing_fits_on_training_only():
    train = pd.DataFrame({"age": [20., 30., np.nan, 50.]})
    validation = pd.DataFrame({"age": [10000., np.nan]})
    p = c.model_pipeline("logistic_regression", ["age"], [], c.DEFAULT_CONFIG)
    p.fit(train, [0, 1, 0, 1])
    imputer = p.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[0] == 30.
    p.predict_proba(validation)
    assert imputer.statistics_[0] == 30.
    xgb = c.FoldBalancedXGB(n_estimators=3, n_jobs=1).fit(np.arange(12).reshape(-1, 1), [0] * 9 + [1] * 3)
    assert xgb.scale_pos_weight_ == 3
    xgb.fit(np.arange(8).reshape(-1, 1), [0] * 4 + [1] * 4)
    assert xgb.scale_pos_weight_ == 1


def test_unknown_and_duplicate_patients_fail(tmp_path):
    cfg = synthetic_inputs(tmp_path)
    data = pd.read_csv(cfg["data_csv"])
    data.loc[data.combined_outcome.eq(0).idxmax(), "followup_complete"] = 0
    data.to_csv(cfg["data_csv"], index=False)
    with pytest.raises(ValueError, match="incomplete/unknown"):
        c.load_cohort(cfg)
    data.followup_complete = 1
    pd.concat([data, data.iloc[:1]]).to_csv(cfg["data_csv"], index=False)
    with pytest.raises(ValueError, match="ONE row"):
        c.load_cohort(cfg)


def test_end_to_end_training_and_evaluation(tmp_path):
    cfg = synthetic_inputs(tmp_path)
    run = c.run_development(cfg)
    out = run["output_dir"]
    assert c.final_evaluation(run, enabled=False) is None
    assert not (out / "FINAL_test_predictions_for_D.csv").exists()
    assert not (out / "final_test_started.json").exists()
    assert all(run["oof"][n].between(0, 1).all() for n in c.PROBABILITY_MODELS)
    frozen = run["bundle"]["frozen"]
    assert frozen["test_outcomes_used_for_training_or_selection"] is False
    assert frozen["selected_primary_model"] in c.MODEL_NAMES
    # Every development patient is predicted exactly once, every model shares that row.
    assert run["oof"].subject_id.is_unique
    assert len(run["oof"]) == frozen["development_n"]
    coef, shap = c.export_interpretation(run, max_patients=30)
    assert shap.mean_abs_shap_log_odds.ge(0).all()
    result = c.final_evaluation(run, enabled=True)
    matched = result[result.scope.eq("rsbi_available_same_patients")]
    assert matched.n.nunique() == 1
    assert matched.loc[matched.model.eq("rsbi"), "brier"].isna().all()
    assert (out / "FINAL_paired_difference_XGB_minus_LR_CI.csv").exists()
    # Identical reruns reuse frozen models and can reproduce the handoff safely.
    second = c.run_development(cfg)
    assert np.allclose(c.final_evaluation(second, enabled=True).auroc, result.auroc)
    with pytest.raises(RuntimeError, match="settings/data changed"):
        c.run_development({**cfg, "sensitivity_target": .9})
