"""Train and evaluate logistic regression and XGBoost for extubation outcomes."""

# %% Imports and defaults
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
    confusion_matrix, log_loss, precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import DMatrix, XGBClassifier

DEFAULT_CONFIG = {
    "data_csv": "../data/clean_final_dataset.csv",
    "metadata_csv": "../data/cohort_metadata.csv",  # optional if included in data_csv
    "output_dir": "outputs/main_analysis",
    "target": "combined_outcome", "seed": 42,
    "split_mode": "temporal",  # temporal or stratified
    "temporal_cutoff": None,  # None: start of latest observed anchor_year_group
    "test_fraction": 0.20,  # stratified split only
    "outer_folds": 5, "inner_folds": 3,
    "search_iterations": 8, "xgb_threads": 4,
    "calibrate": True, "sensitivity_target": 0.80,
    "bootstrap_replicates": 1000,
    "feature_columns": None,  # None selects recognised predictor names
    "categorical_columns": ["gender", "sex", "first_careunit"],
    "rsbi_column": None,  # auto: RSBI_bin_5, rsbi_bin_5, RSBI, rsbi (in this order)
    "synthetic_data": False,
    "quick_test": False,  # reduced search for software tests
}

PHYSIOLOGY = {
    "heart_rate", "mbp", "sbp", "dbp", "resp_rate", "temperature", "spo2",
    "respiratory_rate_total", "respiratory_rate_spontaneous",
    "tidal_volume_observed", "tidal_volume_spontaneous", "fio2", "peep",
    "plateau_pressure", "minute_volume", "minute_ventilation", "po2", "pco2",
    "ph", "fio2_bg", "fio2_chartevents", "pao2fio2ratio", "bicarbonate",
    "baseexcess", "lactate", "rsbi",
}
BASELINE = {"age", "admission_age", "gender", "sex", "first_careunit"}
OUTCOME_COMPONENTS = ["death_within_48_hours", "reintubation_within_48_hours"]
MODEL_NAMES = ["logistic_regression", "xgboost"]
PROBABILITY_MODELS = MODEL_NAMES + ["prevalence_baseline"]


def write_json(path, value):
    """Save settings, converting NumPy scalars and paths to JSON values."""
    def encode(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(f"Cannot encode {type(x)}")
    Path(path).write_text(json.dumps(value, indent=2, default=encode), encoding="utf-8")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binary_series(s, name):
    """Require observed binary labels."""
    out = pd.to_numeric(s, errors="raise")
    if out.isna().any() or not out.isin([0, 1]).all():
        raise ValueError(f"{name} must contain only observed 0/1 labels.")
    return out.astype(int)


def forbidden_feature(name, target):
    n = name.lower()
    return (n == target.lower() or n in {"subject_id", "hadm_id", "stay_id", "dod", "label", "y"}
            or bool(re.search(r"outcome|death|deathtime|reintubat|discharg|dischtime|outtime|expire|los_|length_of_stay|followup|follow_up|anchor_year|extubation_time|admittime|icu_intime", n)))


def recognised_feature(name):
    n = name.lower()
    if n in BASELINE:
        return True
    # Recognise hourly bins and whole-window summaries.
    base = re.sub(r"(_bin_[0-5]|_(mean|std|sd|min|max|last|slope|count|missing)(?:_6h)?|_6h)$", "", n)
    return base in PHYSIOLOGY


# %% Load and validate inputs
def load_cohort(cfg):
    path = Path(cfg["data_csv"])
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}. Set data_csv to the full cohort before median imputation and splitting.")
    if path.stem.startswith(("train_set", "test_set", "X_train", "X_test")):
        raise ValueError("Use the full cohort before median imputation and splitting.")
    # Check duplicate headers before pandas renames them.
    with path.open(newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names; fix the export.")
    df = pd.read_csv(path, dtype={"subject_id": "string", "stay_id": "string", "hadm_id": "string"})
    if df.columns.duplicated().any():
        raise ValueError("Duplicate CSV column names; fix the export.")
    if "subject_id" not in df or cfg["target"] not in df:
        raise ValueError("Required columns: subject_id and " + cfg["target"])
    if df.subject_id.isna().any() or df.subject_id.duplicated().any():
        raise ValueError("Expected ONE row per subject_id, with a single index extubation.")
    # Keep splits reproducible if input rows are reordered.
    df = df.sort_values("subject_id").reset_index(drop=True)
    df[cfg["target"]] = binary_series(df[cfg["target"]], cfg["target"])
    components_present = [c in df for c in OUTCOME_COMPONENTS]
    if all(components_present):
        for c in OUTCOME_COMPONENTS:
            df[c] = binary_series(df[c], c)
        if not np.array_equal(df[cfg["target"]], df[OUTCOME_COMPONENTS].max(axis=1)):
            raise ValueError("combined_outcome disagrees with death OR reintubation.")
    else:
        warnings.warn("Outcome components are missing; cannot verify the combined label.")
    metadata_path = Path(cfg["metadata_csv"]) if cfg.get("metadata_csv") else None
    if metadata_path and metadata_path.exists():
        meta = pd.read_csv(metadata_path, dtype={"subject_id": "string"})
        if "subject_id" not in meta or meta.subject_id.isna().any() or meta.subject_id.duplicated().any():
            raise ValueError("Metadata must have one non-missing row per subject_id.")
        # Overlapping columns must agree before merging metadata.
        both = sorted((set(meta.columns) & set(df.columns)) - {"subject_id"})
        joined = df[["subject_id"] + both].merge(meta, on="subject_id", how="left", validate="one_to_one", suffixes=("_main", "_meta"))
        for c in both:
            a, b = joined[c + "_main"], joined[c + "_meta"]
            if not (a.fillna("<NA>").astype(str).eq(b.fillna("<NA>").astype(str))).all():
                raise ValueError(f"Main CSV and metadata disagree in {c}; resolve before modelling.")
        extra = [c for c in meta if c not in df]
        df = df.merge(meta[["subject_id"] + extra], on="subject_id", how="left", validate="one_to_one")
    if "followup_complete" in df:
        known = pd.to_numeric(df.followup_complete, errors="coerce")
        if ((df[cfg["target"]] == 0) & (known != 1)).any():
            raise ValueError("Negative labels have incomplete/unknown follow-up; resolve them before training.")
    else:
        warnings.warn("No followup_complete flag; cannot verify 48-hour observation for negative labels.")
    requested = cfg.get("feature_columns")
    selected = list(requested) if requested is not None else [c for c in df if recognised_feature(c)]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("No recognised predictors or duplicate features. Set feature_columns explicitly.")
    if not set(selected).issubset(df.columns):
        raise ValueError("Feature columns missing: " + str(sorted(set(selected) - set(df.columns))))
    leaks = [c for c in selected if forbidden_feature(c, cfg["target"])]
    if leaks:
        raise ValueError("Forbidden identifier/outcome/future predictors: " + str(leaks))
    cats = [c for c in selected if c in cfg["categorical_columns"]]
    nums = [c for c in selected if c not in cats]
    if not nums:
        raise ValueError("No numeric physiological predictors selected.")
    for c in nums:
        df[c] = pd.to_numeric(df[c], errors="raise").replace([np.inf, -np.inf], np.nan)
    for c in cats:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    rsbi_candidates = [cfg["rsbi_column"]] if cfg.get("rsbi_column") else ["RSBI_bin_5", "rsbi_bin_5", "RSBI", "rsbi"]
    rsbi_col = next((c for c in rsbi_candidates if c in df), None)
    if cfg.get("rsbi_column") and rsbi_col is None:
        raise ValueError("Configured rsbi_column is absent.")
    if rsbi_col:
        df[rsbi_col] = pd.to_numeric(df[rsbi_col], errors="raise").replace([np.inf, -np.inf], np.nan)
        # Non-positive RSBI values are invalid measurements.
        df.loc[df[rsbi_col] <= 0, rsbi_col] = np.nan
    else:
        warnings.warn("No RSBI column; skipping the RSBI comparison.")
    audit = pd.DataFrame({"column": df.columns})
    audit["used_as_predictor"] = audit.column.isin(selected)
    audit["reason"] = ["predictor" if c in selected else "blocked ID/outcome/time/follow-up" if forbidden_feature(c, cfg["target"]) else "metadata or unrecognised: not selected" for c in audit.column]
    hashes = {"data_csv_sha256": file_hash(path)}
    if metadata_path and metadata_path.exists():
        hashes["metadata_csv_sha256"] = file_hash(metadata_path)
    return df, nums, cats, rsbi_col, audit, hashes


# %% Patient splits
def make_split(df, cfg):
    manifest = df[["subject_id"]].copy()
    manifest["partition"] = "ambiguous_time"
    if cfg["split_mode"] == "temporal":
        required = ["extubation_time", "anchor_year", "anchor_year_group"]
        missing = [c for c in required if c not in df]
        if missing:
            raise ValueError("Temporal holdout needs " + str(missing) + ". Include them in data_csv or metadata_csv, keyed by subject_id.")
        groups = df.anchor_year_group.astype(str).str.extract(r"^\s*(\d{4})\s*[-–]\s*(\d{4})\s*$").astype(float)
        anchor = pd.to_numeric(df.anchor_year, errors="coerce")
        year = pd.to_datetime(df.extubation_time, errors="coerce").dt.year
        if groups.isna().any().any() or anchor.isna().any() or year.isna().any():
            raise ValueError("Temporal metadata contain missing or invalid values.")
        if (groups[1] < groups[0]).any() or (anchor % 1 != 0).any():
            raise ValueError("Invalid anchor year or year-group interval.")
        offset = year - anchor
        # Map shifted extubation years to possible calendar-year intervals.
        manifest["possible_year_lower"] = (groups[0] + offset).astype(int)
        manifest["possible_year_upper"] = (groups[1] + offset).astype(int)
        cutoff = cfg["temporal_cutoff"] if cfg["temporal_cutoff"] is not None else int(groups[0].max())
        manifest.loc[manifest.possible_year_upper < cutoff, "partition"] = "development"
        manifest.loc[manifest.possible_year_lower >= cutoff, "partition"] = "test"
        split_note = {"mode": "temporal", "cutoff": int(cutoff), "rule": "development: upper possible year < cutoff; test: lower possible year >= cutoff; overlapping intervals excluded", "cutoff_choice": "explicit" if cfg["temporal_cutoff"] is not None else "start of latest observed anchor_year_group; chosen without model performance"}
    elif cfg["split_mode"] == "stratified":
        dev, test = train_test_split(np.arange(len(df)), test_size=cfg["test_fraction"], stratify=df[cfg["target"]], random_state=cfg["seed"])
        manifest.loc[dev, "partition"] = "development"
        manifest.loc[test, "partition"] = "test"
        split_note = {"mode": "stratified", "deviation": "Random patient holdout, NOT temporal validation", "test_fraction": cfg["test_fraction"]}
    else:
        raise ValueError("split_mode must be temporal or stratified.")
    dev = np.flatnonzero(manifest.partition.eq("development"))
    test = np.flatnonzero(manifest.partition.eq("test"))
    if not len(dev) or not len(test):
        raise ValueError("Empty development/test set. Check year intervals and temporal_cutoff.")
    assert not set(df.iloc[dev].subject_id) & set(df.iloc[test].subject_id)
    # Validate fold sizes using development labels only.
    checked_cv(df.iloc[dev][cfg["target"]].to_numpy(), cfg["outer_folds"], cfg["seed"])
    return manifest, dev, test, split_note


def checked_cv(y, folds, seed):
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    if folds < 2 or min(counts) < folds:
        raise ValueError(f"Need at least {folds} patients in each class for cross-validation; found {counts.tolist()}.")
    return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


# %% Preprocessing, tuning and calibration
class FoldBalancedXGB(ClassifierMixin, BaseEstimator):
    """Calculate class weights from the current training fold."""
    def __init__(self, n_estimators=250, max_depth=3, learning_rate=0.05,
                 min_child_weight=1, reg_lambda=1, subsample=0.8,
                 colsample_bytree=0.8, random_state=42, n_jobs=4):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
        if min(counts) == 0:
            raise ValueError("XGBoost fit requires both classes.")
        self.scale_pos_weight_ = float(counts[0] / counts[1])
        self.model_ = XGBClassifier(**self.get_params(), scale_pos_weight=self.scale_pos_weight_,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cpu")
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def model_pipeline(name, nums, cats, cfg):
    numeric = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if name == "logistic_regression":
        numeric.append(("scale", StandardScaler()))
    parts = [("numeric", Pipeline(numeric), nums)]
    if cats:
        categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="unknown", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        parts.append(("categorical", categorical, cats))
    prep = ColumnTransformer(parts, remainder="drop", sparse_threshold=0)
    if name == "logistic_regression":
        model = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=3000, random_state=cfg["seed"])
    elif name == "xgboost":
        model = FoldBalancedXGB(random_state=cfg["seed"], n_jobs=cfg["xgb_threads"])
    else:
        raise ValueError(name)
    return Pipeline([("preprocess", prep), ("model", model)])


def fit_tuned(name, X, y, nums, cats, cfg, cv):
    pipe = model_pipeline(name, nums, cats, cfg)
    common = dict(scoring={"auroc": "roc_auc", "average_precision": "average_precision"},
        refit="auroc", cv=cv, n_jobs=1, error_score="raise", return_train_score=False)
    if name == "logistic_regression":
        grid = [0.1, 1] if cfg["quick_test"] else [0.001, 0.01, 0.1, 1, 10, 100]
        search = GridSearchCV(pipe, {"model__C": grid}, **common)
    else:
        params = {"model__n_estimators": [30] if cfg["quick_test"] else [200, 400],
            "model__max_depth": [2, 3, 4], "model__learning_rate": [0.03, 0.1],
            "model__min_child_weight": [1, 5], "model__reg_lambda": [1, 10]}
        search = RandomizedSearchCV(pipe, params, n_iter=1 if cfg["quick_test"] else cfg["search_iterations"], random_state=cfg["seed"], **common)
    search.fit(X, y)
    fitted = search.best_estimator_
    if cfg["calibrate"]:
        # Calibrate within training folds, then refit on this training partition.
        fitted = CalibratedClassifierCV(clone(fitted), method="sigmoid", cv=cv, ensemble=False, n_jobs=1)
        fitted.fit(X, y)
    cols = [c for c in search.cv_results_ if c == "params" or c.startswith(("mean_test_", "std_test_", "rank_test_", "mean_fit_"))]
    return fitted, search.best_params_, pd.DataFrame(search.cv_results_)[cols]


# %% Metrics, thresholds and plots
def safe_ratio(a, b):
    return float(a / b) if b else np.nan


def metrics(y, score, threshold=0.5, probability=True):
    """Compute classification metrics and, for probabilities, calibration losses."""
    y, score = np.asarray(y, int), np.asarray(score, float)
    keep = np.isfinite(score)
    y, score = y[keep], score[keep]
    if not len(y):
        return {"n": 0}
    two = len(np.unique(y)) == 2
    tn, fp, fn, tp = confusion_matrix(y, score >= threshold, labels=[0, 1]).ravel()
    result = {"n": len(y), "events": int(y.sum()), "prevalence": float(y.mean()),
        "auroc": roc_auc_score(y, score) if two else np.nan,
        "average_precision": average_precision_score(y, score) if two else np.nan,
        "threshold": float(threshold), "sensitivity": safe_ratio(tp, tp + fn),
        "specificity": safe_ratio(tn, tn + fp), "ppv": safe_ratio(tp, tp + fp),
        "npv": safe_ratio(tn, tn + fn), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    result["brier"] = brier_score_loss(y, score) if probability else np.nan
    result["log_loss"] = log_loss(y, np.clip(score, 1e-7, 1 - 1e-7), labels=[0, 1]) if probability else np.nan
    return result


def choose_threshold(y, score, sensitivity_target):
    """Maximise specificity at the target sensitivity using development OOF scores."""
    if not 0 < sensitivity_target <= 1:
        raise ValueError("sensitivity_target must be in (0,1].")
    fpr, tpr, thresholds = roc_curve(y, score, drop_intermediate=False)
    eligible = np.flatnonzero((tpr >= sensitivity_target) & np.isfinite(thresholds))
    # roc_curve thresholds descend: among tied FPR choose the highest threshold.
    best = eligible[np.argmin(fpr[eligible])]
    return float(thresholds[best])


def prediction_table(df, cfg, scores, rsbi_col, fold=None):
    cols = [c for c in ["subject_id", "hadm_id", "stay_id", cfg["target"], *OUTCOME_COMPONENTS, "age", "admission_age", "gender", "sex"] if c in df]
    out = df[cols].copy().reset_index(drop=True)
    out = out.rename(columns={cfg["target"]: "y_true"})
    for name, values in scores.items():
        out[name] = values
    if rsbi_col:
        out["rsbi"] = df[rsbi_col].to_numpy()
    if fold is not None:
        out["outer_fold"] = fold
    return out


def comparison_tables(predictions, thresholds):
    rows = []
    scopes = {"all_patients": np.ones(len(predictions), dtype=bool)}
    if "rsbi" in predictions:
        scopes["rsbi_available_same_patients"] = np.isfinite(predictions.rsbi)
    for scope, mask in scopes.items():
        p = predictions.loc[mask]
        for name in PROBABILITY_MODELS + ["random_guess"]:
            if name in p:
                # Calibration losses apply to probability models only.
                rows.append({"scope": scope, "model": name, **metrics(p.y_true, p[name], thresholds.get(name, 0.5), name in PROBABILITY_MODELS)})
        if scope == "rsbi_available_same_patients":
            rows.append({"scope": scope, "model": "rsbi", **metrics(p.y_true, p.rsbi, 105, False)})
    return pd.DataFrame(rows)


def plot_comparison(p, folder, prefix, include_rsbi=False):
    folder = Path(folder)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    if p.y_true.nunique() == 2:
        names = PROBABILITY_MODELS + ["random_guess"] + (["rsbi"] if include_rsbi else [])
        for name in names:
            fpr, tpr, _ = roc_curve(p.y_true, p[name])
            precision, recall, _ = precision_recall_curve(p.y_true, p[name])
            axes[0].plot(fpr, tpr, label=name)
            # Step interpolation matches average precision.
            axes[1].step(recall, precision, where="post", label=name)
        axes[0].plot([0, 1], [0, 1], "k--", alpha=.5)
        axes[1].axhline(p.y_true.mean(), color="k", linestyle="--", label="prevalence")
        for name in PROBABILITY_MODELS:
            observed, predicted = calibration_curve(p.y_true, p[name], n_bins=8, strategy="quantile")
            axes[2].plot(predicted, observed, "o-", label=name)
    axes[2].plot([0, 1], [0, 1], "k--", alpha=.5)
    for ax, xlabel, ylabel in zip(axes, ["False positive rate", "Recall", "Mean predicted probability"], ["Sensitivity", "Precision", "Observed event rate"]):
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7)
    fig.suptitle(prefix + ": retrospective predictions")
    fig.tight_layout()
    fig.savefig(folder / f"{prefix}_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# %% Model development
def run_development(config):
    cfg = {**DEFAULT_CONFIG, **config}
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df, nums, cats, rsbi_col, audit, hashes = load_cohort(cfg)
    # Reuse frozen models after the test set has been opened.
    if (out / "final_test_started.json").exists():
        frozen = json.loads((out / "frozen_analysis.json").read_text())
        settings_changed = any(cfg[key] != frozen["config"].get(key) for key in DEFAULT_CONFIG)
        if settings_changed or hashes != frozen["source_hashes"]:
            raise RuntimeError("Final test already opened and settings/data changed. Restore the frozen settings and data.")
        bundle = joblib.load(out / "trained_models.joblib")
        dev = df.set_index("subject_id", drop=False).loc[bundle["development_ids"]].reset_index(drop=True)
        print("Reusing saved models and results; final evaluation has already started.")
        return {"bundle": bundle, "data": df, "development": dev,
            "oof": pd.read_csv(out / "development_oof_predictions.csv", dtype={"subject_id": "string"}),
            "manifest": pd.read_csv(out / "patient_split_manifest.csv", dtype={"subject_id": "string"}), "output_dir": out}
    manifest, dev_idx, test_idx, split_note = make_split(df, cfg)
    dev = df.iloc[dev_idx].reset_index(drop=True)
    X, y = dev[nums + cats], dev[cfg["target"]].to_numpy()
    if min(np.bincount(y, minlength=2)) < 50:
        warnings.warn("Fewer than 50 development patients in one class; estimates may be unstable.")
    outer = checked_cv(y, cfg["outer_folds"], cfg["seed"])
    scores = {n: np.full(len(dev), np.nan) for n in PROBABILITY_MODELS}
    # Fixed random scores provide a reproducible chance baseline.
    scores["random_guess"] = np.random.default_rng(cfg["seed"]).uniform(size=len(dev))
    folds = np.full(len(dev), -1, dtype=int)
    fold_results, best_params = [], []
    for fold, (train, valid) in enumerate(outer):
        inner = checked_cv(y[train], cfg["inner_folds"], cfg["seed"] + fold + 1)
        folds[valid] = fold
        scores["prevalence_baseline"][valid] = y[train].mean()
        for name in MODEL_NAMES:
            print(f"Outer fold {fold + 1}/{len(outer)}: {name}", flush=True)
            fitted, params, search_table = fit_tuned(name, X.iloc[train], y[train], nums, cats, cfg, inner)
            scores[name][valid] = fitted.predict_proba(X.iloc[valid])[:, 1]
            search_table.to_csv(out / f"search_outer_{fold}_{name}.csv", index=False)
            best_params.append({"outer_fold": fold, "model": name, "parameters": params})
        for name, values in scores.items():
            fold_results.append({"outer_fold": fold, "model": name, **metrics(y[valid], values[valid], 0.5, name in PROBABILITY_MODELS)})
    assert all(np.isfinite(s).all() for s in scores.values()) and (folds >= 0).all()
    oof = prediction_table(dev, cfg, scores, rsbi_col, folds)
    oof.to_csv(out / "development_oof_predictions.csv", index=False)
    manifest["outer_fold"] = -1
    manifest.loc[dev_idx, "outer_fold"] = folds
    manifest.to_csv(out / "patient_split_manifest.csv", index=False)
    pd.DataFrame(fold_results).to_csv(out / "outer_fold_metrics_fixed_0p5.csv", index=False)
    comparison_tables(oof, {}).to_csv(out / "development_metrics_fixed_0p5.csv", index=False)
    fold_frame = pd.DataFrame(fold_results)
    fold_frame.groupby("model")[["auroc", "average_precision", "brier", "sensitivity", "specificity"]].agg(["mean", "std"]).to_csv(out / "outer_fold_summary_mean_sd.csv")
    ranked = fold_frame[fold_frame.model.isin(MODEL_NAMES)].groupby("model").auroc.mean().sort_values(ascending=False)
    winner = str(ranked.index[0])
    thresholds = {name: choose_threshold(y, scores[name], cfg["sensitivity_target"]) for name in MODEL_NAMES}
    thresholds.update({"prevalence_baseline": 0.5, "random_guess": 0.5})
    # Threshold-selection scores are descriptive, not independent validation.
    comparison_tables(oof, thresholds).to_csv(out / "threshold_derivation_DESCRIPTIVE_NOT_VALIDATION.csv", index=False)
    plot_comparison(oof, out, "development_oof")
    if "rsbi" in oof:
        plot_comparison(oof.loc[np.isfinite(oof.rsbi)], out, "development_rsbi_matched_subset", include_rsbi=True)
    final_models = {}
    final_cv = checked_cv(y, cfg["inner_folds"], cfg["seed"] + 100)
    final_params = {}
    for name in MODEL_NAMES:
        print(f"Refitting {name} on development patients only", flush=True)
        fitted, params, search_table = fit_tuned(name, X, y, nums, cats, cfg, final_cv)
        final_models[name] = fitted
        final_params[name] = params
        search_table.to_csv(out / f"search_final_development_{name}.csv", index=False)
    # Describe missingness on development patients only.
    dev[nums + cats].isna().mean().rename("missing_fraction_development").to_csv(out / "development_feature_missingness.csv")
    audit.to_csv(out / "feature_audit.csv", index=False)
    write_json(out / "outer_best_parameters.json", best_params)
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "scipy", "matplotlib", "joblib"]}
    versions["python"] = platform.python_version()
    write_json(out / "environment_versions.json", versions)
    frozen = {"config": cfg, "source_hashes": hashes, "split": split_note,
        "selected_features": nums + cats, "rsbi_column": rsbi_col,
        "partition_counts": manifest.partition.value_counts().to_dict(),
        "thresholds": thresholds, "threshold_rule": "maximum specificity at development OOF sensitivity >= configured target; exploratory, not clinically validated",
        "selected_primary_model": winner, "selection_rule": "highest mean outer-fold AUROC, before final test",
        "mean_outer_fold_auroc": ranked.to_dict(), "final_parameters": final_params,
        "development_n": len(dev), "development_events": int(y.sum()),
        "test_outcomes_used_for_training_or_selection": False}
    write_json(out / "frozen_analysis.json", frozen)
    bundle = {"models": final_models, "frozen": frozen, "development_prevalence": float(y.mean()),
        "development_ids": dev.subject_id.tolist(), "test_ids": df.iloc[test_idx].subject_id.tolist()}
    joblib.dump(bundle, out / "trained_models.joblib")
    print(f"Development complete. Selected model: {winner}. Final test not evaluated.")
    return {"bundle": bundle, "data": df, "development": dev, "oof": oof, "manifest": manifest, "output_dir": out}


# %% Model interpretation
def base_pipeline(model):
    return model.calibrated_classifiers_[0].estimator if isinstance(model, CalibratedClassifierCV) else model


def export_interpretation(run, max_patients=300):
    bundle, dev, out = run["bundle"], run["development"], run["output_dir"]
    cfg = bundle["frozen"]["config"]
    features = bundle["frozen"]["selected_features"]
    lr = base_pipeline(bundle["models"]["logistic_regression"])
    names = lr.named_steps["preprocess"].get_feature_names_out()
    coefficients = pd.DataFrame({"feature": names, "coefficient": lr.named_steps["model"].coef_[0]})
    coefficients["abs_coefficient"] = coefficients.coefficient.abs()
    coefficients.sort_values("abs_coefficient", ascending=False).to_csv(out / "logistic_base_coefficients.csv", index=False)
    # Native TreeSHAP contributions sum to the base model's log odds.
    xgb = base_pipeline(bundle["models"]["xgboost"])
    sample = dev[features].sample(min(max_patients, len(dev)), random_state=cfg["seed"])
    transformed = xgb.named_steps["preprocess"].transform(sample)
    booster = xgb.named_steps["model"].model_.get_booster()
    contrib = booster.predict(DMatrix(transformed), pred_contribs=True)
    margin = booster.predict(DMatrix(transformed), output_margin=True)
    np.testing.assert_allclose(contrib.sum(axis=1), margin, atol=1e-4, rtol=1e-4)
    shap_table = pd.DataFrame({"feature": xgb.named_steps["preprocess"].get_feature_names_out(),
        "mean_abs_shap_log_odds": np.abs(contrib[:, :-1]).mean(axis=0)}).sort_values("mean_abs_shap_log_odds", ascending=False)
    shap_table.to_csv(out / "xgboost_base_treeshap.csv", index=False)
    top = shap_table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.feature, top.mean_abs_shap_log_odds)
    ax.set(xlabel="Mean absolute TreeSHAP contribution (base-model log odds)", title="Development sample: predictive associations, not causal effects")
    fig.tight_layout()
    fig.savefig(out / "xgboost_base_treeshap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return coefficients, shap_table


# %% Final evaluation
def bootstrap_intervals(p, thresholds, cfg):
    """Use the same resampled patients for every model in each bootstrap replicate."""
    rng = np.random.default_rng(cfg["seed"] + 200)
    samples, differences = [], []
    for rep in range(cfg["bootstrap_replicates"]):
        sample = p.iloc[rng.integers(0, len(p), len(p))]
        if sample.y_true.nunique() < 2:
            continue
        table = comparison_tables(sample, thresholds)
        for _, row in table.iterrows():
            for metric in ["auroc", "average_precision", "sensitivity", "specificity", "ppv", "npv", "brier"]:
                samples.append({"replicate": rep, "scope": row.scope, "model": row.model, "metric": metric, "value": row.get(metric, np.nan)})
        paired = table[table.scope.eq("all_patients")].set_index("model")
        for metric in ["auroc", "average_precision", "brier"]:
            differences.append({"replicate": rep, "metric": metric, "value": paired.loc["xgboost", metric] - paired.loc["logistic_regression", metric]})
    if not samples:
        raise ValueError("No usable bootstrap replicates; insufficient test events.")
    def summarise(frame, groups):
        return frame.groupby(groups).value.agg(valid_replicates="count", lower_95=lambda s: s.quantile(.025), upper_95=lambda s: s.quantile(.975)).reset_index()
    return summarise(pd.DataFrame(samples), ["scope", "model", "metric"]), summarise(pd.DataFrame(differences), ["metric"])


def final_evaluation(run, enabled=False):
    if not enabled:
        print("Final test disabled. Enable it after fixing models, features and thresholds.")
        return None
    bundle, df, out = run["bundle"], run["data"], run["output_dir"]
    frozen = bundle["frozen"]
    cfg = frozen["config"]
    marker = out / "final_test_started.json"
    signature = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    model_hash = file_hash(out / "trained_models.joblib")
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior.get("analysis_signature") != signature or prior.get("model_sha256") != model_hash:
            raise RuntimeError("Models or settings changed after final evaluation started.")
        if prior["status"] == "completed":
            print("Loading saved final-test results.")
            return pd.read_csv(out / "FINAL_test_metrics.csv")
        print("Resuming final evaluation with the saved models.")
    # Save the marker before evaluation so interrupted runs can resume.
    marker_info = {"analysis_signature": signature, "model_sha256": model_hash,
        "selected_primary_model": frozen["selected_primary_model"], "synthetic_data": cfg["synthetic_data"]}
    write_json(marker, {**marker_info, "status": "started"})
    test = df.set_index("subject_id", drop=False).loc[bundle["test_ids"]].reset_index(drop=True)
    X = test[frozen["selected_features"]]
    scores = {n: m.predict_proba(X)[:, 1] for n, m in bundle["models"].items()}
    scores["prevalence_baseline"] = np.full(len(test), bundle["development_prevalence"])
    scores["random_guess"] = np.random.default_rng(cfg["seed"] + 300).uniform(size=len(test))
    p = prediction_table(test, cfg, scores, frozen["rsbi_column"])
    p.to_csv(out / "FINAL_test_predictions.csv", index=False)
    table = comparison_tables(p, frozen["thresholds"])
    table.to_csv(out / "FINAL_test_metrics.csv", index=False)
    plot_comparison(p, out, "FINAL_test")
    if "rsbi" in p:
        plot_comparison(p.loc[np.isfinite(p.rsbi)], out, "FINAL_test_rsbi_matched_subset", include_rsbi=True)
    if p.y_true.nunique() == 2 and cfg["bootstrap_replicates"] > 0:
        ci, differences = bootstrap_intervals(p, frozen["thresholds"], cfg)
        ci.to_csv(out / "FINAL_test_patient_bootstrap_CI.csv", index=False)
        differences.to_csv(out / "FINAL_paired_difference_XGB_minus_LR_CI.csv", index=False)
    else:
        warnings.warn("Test set has one outcome class or bootstrap is disabled; AUROC or intervals may be unavailable.")
    # Apply the same frozen thresholds to error and subgroup summaries.
    for name in MODEL_NAMES:
        p[name + "_predicted_failure"] = (p[name] >= frozen["thresholds"][name]).astype(int)
        p[name + "_error"] = np.select([(p.y_true == 1) & (p[name + "_predicted_failure"] == 0),
            (p.y_true == 0) & (p[name + "_predicted_failure"] == 1)], ["false_negative", "false_positive"], default="correct")
    p.to_csv(out / "FINAL_test_error_analysis.csv", index=False)
    subgroup_rows = []
    for col in ["gender", "sex"]:
        if col in p:
            for value, group in p.groupby(col, dropna=False):
                for name in MODEL_NAMES:
                    subgroup_rows.append({"variable": col, "group": str(value), "model": name,
                        "interpretation": "exploratory; small subgroups may be unstable", **metrics(group.y_true, group[name], frozen["thresholds"][name])})
    if subgroup_rows:
        pd.DataFrame(subgroup_rows).to_csv(out / "FINAL_exploratory_subgroup_metrics.csv", index=False)
    write_json(marker, {**marker_info, "status": "completed",
        "CI_interpretation": "Conditional on fitted models and this test cohort; does not include uncertainty from repeating model development."})
    return table
