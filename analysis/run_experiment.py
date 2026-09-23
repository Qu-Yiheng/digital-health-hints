from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "tmp" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import t as student_t
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier


RANDOM_STATE = 20260915
JK_COEF = 0.98
REPLICATES = 50

DATA_ROOT = PROJECT / "data_extracted"
PROCESSED_DIR = PROJECT / "data_processed"
OUTPUT_DIR = PROJECT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

for directory in [PROCESSED_DIR, TABLE_DIR, FIGURE_DIR, MODEL_DIR, PROJECT / "tmp" / "matplotlib"]:
    directory.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=UnicodeWarning)
warnings.filterwarnings("ignore", message=".*Perfect separation.*")


PLOT_COLORS = {
    "navy": "#1F4E79",
    "orange": "#D97706",
    "green": "#2E7D32",
    "purple": "#7B2CBF",
    "gray": "#5B6573",
}


def find_source_data(filename: str) -> Path:
    """Locate a HINTS Stata file and provide an actionable error if absent."""
    matches = sorted(DATA_ROOT.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {DATA_ROOT}. "
            "Download the HINTS Stata files and extract them anywhere inside "
            "the data_extracted directory; see README.md for instructions."
        )
    return matches[0]


def set_publication_plot_style() -> None:
    """Apply a consistent, journal-ready visual style to every figure."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.titleweight": "semibold",
            "axes.labelsize": 16,
            "axes.labelweight": "medium",
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 13.5,
            "legend.fontsize": 13,
            "legend.title_fontsize": 14,
            "figure.titlesize": 21,
            "figure.titleweight": "semibold",
            "axes.edgecolor": "#6B7280",
            "axes.linewidth": 1.0,
            "grid.color": "#D7DCE2",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.72,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_publication_figure(fig: plt.Figure, filename: str) -> None:
    """Save a high-resolution raster copy and a vector PDF copy."""
    fig.savefig(FIGURE_DIR / filename, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / Path(filename).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def valid_binary(series: pd.Series, yes: tuple[int, ...] = (1,), no: tuple[int, ...] = (2,)) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[series.isin(yes)] = 1.0
    out.loc[series.isin(no)] = 0.0
    return out


def valid_range(series: pd.Series, low: int, high: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").where(series.between(low, high))


def harmonize(path: Path, cycle: int) -> pd.DataFrame:
    raw = pd.read_stata(path, convert_categoricals=False)
    out = pd.DataFrame(index=raw.index)
    out["cycle"] = cycle
    out["cycle_2024"] = float(cycle == 2024)
    out["HHID"] = raw["HHID"].astype(str)

    # Survey weights are preserved exactly; scaling is added after cycles are stacked.
    for r in range(REPLICATES + 1):
        out[f"PERSON_FINWT{r}"] = pd.to_numeric(raw[f"PERSON_FINWT{r}"], errors="coerce")

    out["hcp_encourage"] = valid_binary(raw["HCPEncourageOnlineRec2"])
    out["telehealth_user"] = valid_binary(raw["ReceiveTelehealthCare"], yes=(1, 2, 3), no=(4,))
    out["hcp_x_telehealth"] = out["hcp_encourage"] * out["telehealth_user"]

    access_name = "AccessOnlineRecord2" if cycle == 2022 else "AccessOnlineRecord3"
    access_raw = pd.to_numeric(raw[access_name], errors="coerce")
    access_freq = access_raw.where(access_raw.isin([0, 1, 2, 3, 4]))
    if cycle == 2022:
        access_freq = access_freq.mask(access_raw.eq(5), 0)
    out["access_frequency_code"] = access_freq
    out["portal_use"] = np.where(access_freq.notna(), access_freq.between(1, 4).astype(float), np.nan)
    out["routine_use"] = np.where(
        out["portal_use"].eq(1), access_freq.between(2, 4).astype(float), np.nan
    )
    out["high_frequency_use"] = np.where(
        out["portal_use"].eq(1), access_freq.between(3, 4).astype(float), np.nan
    )

    results_name = "RecordsOnline_ViewResults" if cycle == 2022 else "RecordsOnline2_ViewResults"
    notes_name = "RecordsOnline_ViewNotes" if cycle == 2022 else "RecordsOnline2_ViewNotes"
    viewed_results = valid_binary(raw[results_name])
    viewed_notes = valid_binary(raw[notes_name])
    out["viewed_results"] = np.where(out["portal_use"].eq(1), viewed_results, np.nan)
    out["viewed_notes"] = np.where(out["portal_use"].eq(1), viewed_notes, np.nan)
    both_valid = viewed_results.notna() & viewed_notes.notna() & out["portal_use"].eq(1)
    out["deep_functional_use"] = np.where(
        both_valid, (viewed_results.eq(1) & viewed_notes.eq(1)).astype(float), np.nan
    )
    out["any_functional_use"] = np.where(
        both_valid, (viewed_results.eq(1) | viewed_notes.eq(1)).astype(float), np.nan
    )

    organizer = valid_binary(raw["UsedPortalOrganizerApp"])
    out["organizer_use"] = np.where(out["portal_use"].eq(1), organizer, np.nan)

    source_columns = [
        "OnlinePortal_PCP",
        "OnlinePortal_OthHCP",
        "OnlinePortal_Insurer",
        "OnlinePortal_Lab",
        "OnlinePortal_Pharmacy",
    ]
    source_selected = pd.concat([valid_binary(raw[c]) for c in source_columns], axis=1)
    source_selected.columns = source_columns
    out["portal_source_count_common"] = source_selected.sum(axis=1, min_count=3)
    out["multiple_portal_sources"] = np.where(
        out["portal_source_count_common"].notna(),
        out["portal_source_count_common"].ge(2).astype(float),
        np.nan,
    )
    if cycle == 2022:
        multiple_portals = valid_binary(raw["MultipleOnlinePortals"], yes=(2,), no=(1,))
        out["integration_eligible"] = (
            out["portal_use"].eq(1) & multiple_portals.eq(1) & out["organizer_use"].notna()
        )
    else:
        out["integration_eligible"] = (
            out["portal_use"].eq(1)
            & out["multiple_portal_sources"].eq(1)
            & out["organizer_use"].notna()
        )

    out["journey_stage"] = np.nan
    out.loc[out["portal_use"].eq(0), "journey_stage"] = 0
    out.loc[out["portal_use"].eq(1) & access_freq.eq(1), "journey_stage"] = 1
    out.loc[out["portal_use"].eq(1) & access_freq.between(2, 4), "journey_stage"] = 2
    out.loc[out["portal_use"].eq(1) & out["organizer_use"].eq(1), "journey_stage"] = 3

    offered_hcp = valid_binary(raw["OfferedAccessHCP3"], yes=(1,), no=(2, 3))
    offered_insurer = valid_binary(raw["OfferedAccessInsurer3"], yes=(1,), no=(2, 3))
    offered_valid = offered_hcp.notna() & offered_insurer.notna()
    out["offered_any_portal"] = np.where(
        offered_valid,
        (offered_hcp.eq(1) | offered_insurer.eq(1)).astype(float),
        np.nan,
    )

    out["age_group"] = valid_range(raw["AgeGrpB"], 1, 5)
    out["age_65_plus"] = np.where(
        out["age_group"].notna(), out["age_group"].ge(4).astype(float), np.nan
    )
    if cycle == 2022:
        out["female"] = valid_binary(raw["BirthGender"], yes=(2,), no=(1,))
    else:
        out["female"] = valid_binary(raw["BirthSex"], yes=(1,), no=(2,))
    out["education"] = valid_range(raw["EducA"], 1, 4)
    out["low_education"] = np.where(
        out["education"].notna(), out["education"].le(2).astype(float), np.nan
    )
    out["race_ethnicity"] = valid_range(raw["RaceEthn5"], 1, 5)
    out["income_group"] = valid_range(raw["IncomeRanges_IMP"], 1, 9)
    out["low_income"] = np.where(
        out["income_group"].notna(), out["income_group"].le(5).astype(float), np.nan
    )
    out["insured"] = valid_binary(raw["HealthInsurance2"])
    out["provider_visits"] = valid_range(raw["FreqGoProvider"], 0, 6)
    ruc = valid_range(raw["RUC2013"], 1, 9)
    out["rural"] = np.where(ruc.notna(), ruc.ge(4).astype(float), np.nan)
    out["general_health"] = valid_range(raw["GeneralHealth"], 1, 5)

    if cycle == 2022:
        out["internet_user"] = valid_binary(raw["UseInternet"])
        out["smartphone_user"] = valid_binary(raw["HaveDevice_SmartPh"])
    else:
        out["internet_user"] = valid_binary(raw["FreqUseInternet"], yes=(1, 2, 3, 4, 5), no=(6,))
        out["smartphone_user"] = valid_binary(raw["UseDevice_SmPhone"])

    out["health_app_user"] = valid_binary(raw["UsedHealthWellnessApps2"], yes=(1,), no=(2, 3, 4))

    condition_columns = [
        "MedConditions_Diabetes",
        "MedConditions_HighBP",
        "MedConditions_HeartCondition",
        "MedConditions_LungDisease",
        "MedConditions_Depression",
        "EverHadCancer",
    ]
    conditions = pd.concat([valid_binary(raw[c]) for c in condition_columns], axis=1)
    out["comorbidity_count"] = conditions.sum(axis=1, min_count=4)
    out["comorbidity_missing"] = conditions.isna().sum(axis=1).gt(0).astype(float)
    return out


def scale_survey_weights(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    # The HINTS 7 guide's HINTS 6+7 merge example stacks the final weights.
    # A single global rescaling improves numerical conditioning without changing estimates.
    scale = len(data) / data["PERSON_FINWT0"].sum()
    data["survey_weight"] = data["PERSON_FINWT0"] * scale
    for r in range(1, REPLICATES + 1):
        data[f"rep_weight_{r}"] = data[f"PERSON_FINWT{r}"] * scale
    return data


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid]))


def cycle_jk_proportion(data: pd.DataFrame, indicator: pd.Series) -> dict[str, float]:
    valid = indicator.notna() & data["survey_weight"].notna()
    y = indicator.loc[valid].to_numpy(float)
    point = weighted_mean(y, data.loc[valid, "survey_weight"].to_numpy(float))
    reps = []
    for r in range(1, REPLICATES + 1):
        reps.append(weighted_mean(y, data.loc[valid, f"rep_weight_{r}"].to_numpy(float)))
    se = math.sqrt(JK_COEF * np.sum((np.asarray(reps) - point) ** 2))
    critical = student_t.ppf(0.975, REPLICATES - 1)
    return {
        "unweighted_n": int(valid.sum()),
        "estimate": point,
        "standard_error": se,
        "ci_lower": max(0.0, point - critical * se),
        "ci_upper": min(1.0, point + critical * se),
    }


def build_design_matrix(data: pd.DataFrame, include_cycle: bool = True) -> pd.DataFrame:
    binary = [
        "hcp_encourage",
        "telehealth_user",
        "hcp_x_telehealth",
        "female",
        "insured",
        "rural",
        "internet_user",
        "smartphone_user",
    ]
    continuous = ["provider_visits", "general_health", "comorbidity_count"]
    if include_cycle:
        binary.append("cycle_2024")

    blocks = []
    for col in binary + continuous:
        values = pd.to_numeric(data[col], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        blocks.append(values.fillna(median).rename(col))
        if values.isna().any():
            blocks.append(values.isna().astype(float).rename(f"{col}_missing"))

    categorical = data[["age_group", "education", "race_ethnicity", "income_group"]].copy()
    for col in categorical.columns:
        categorical[col] = categorical[col].map(lambda x: f"{int(x)}" if pd.notna(x) else "Missing")
    dummies = pd.get_dummies(categorical, prefix=categorical.columns, drop_first=True, dtype=float)
    x = pd.concat(blocks + [dummies], axis=1).astype(float)
    x = x.loc[:, ~x.columns.duplicated()]
    varying = x.nunique(dropna=False).gt(1)
    x = x.loc[:, varying]
    return sm.add_constant(x, has_constant="add")


def fit_weighted_logit(y: pd.Series, x: pd.DataFrame, weights: pd.Series):
    model = sm.GLM(
        y.to_numpy(float),
        x.to_numpy(float),
        family=sm.families.Binomial(),
        freq_weights=weights.to_numpy(float),
    )
    result = model.fit(maxiter=120, disp=0)
    return pd.Series(result.params, index=x.columns)


def probability_contrasts(x: pd.DataFrame, beta: pd.Series, weights: pd.Series) -> dict[str, float]:
    columns = list(x.columns)
    b = beta.reindex(columns).to_numpy(float)
    matrix = x.to_numpy(float)
    h_idx = columns.index("hcp_encourage")
    t_idx = columns.index("telehealth_user")
    i_idx = columns.index("hcp_x_telehealth")
    eta_base = matrix @ b
    eta_base = (
        eta_base
        - matrix[:, h_idx] * b[h_idx]
        - matrix[:, t_idx] * b[t_idx]
        - matrix[:, i_idx] * b[i_idx]
    )

    def p(h: int, t: int) -> np.ndarray:
        return expit(eta_base + h * b[h_idx] + t * b[t_idx] + h * t * b[i_idx])

    p00, p10, p01, p11 = p(0, 0), p(1, 0), p(0, 1), p(1, 1)
    h_observed = matrix[:, h_idx]
    t_observed = matrix[:, t_idx]
    p_h1 = np.where(t_observed == 1, p11, p10)
    p_h0 = np.where(t_observed == 1, p01, p00)
    p_t1 = np.where(h_observed == 1, p11, p01)
    p_t0 = np.where(h_observed == 1, p10, p00)
    w = weights.to_numpy(float)
    return {
        "AME_HCP_encouragement": weighted_mean(p_h1 - p_h0, w),
        "AME_telehealth_use": weighted_mean(p_t1 - p_t0, w),
        "additive_interaction": weighted_mean(p11 - p10 - p01 + p00, w),
        "scenario_neither": weighted_mean(p00, w),
        "scenario_HCP_only": weighted_mean(p10, w),
        "scenario_telehealth_only": weighted_mean(p01, w),
        "scenario_both": weighted_mean(p11, w),
    }


def survey_logit_with_replicates(
    data: pd.DataFrame,
    outcome: str,
    model_label: str,
    eligibility: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    mask = (
        eligibility
        & data[outcome].notna()
        & data["hcp_encourage"].notna()
        & data["telehealth_user"].notna()
        & data["survey_weight"].notna()
    )
    sample = data.loc[mask].copy()
    x = build_design_matrix(sample, include_cycle=True)
    y = sample[outcome].astype(float)
    beta = fit_weighted_logit(y, x, sample["survey_weight"])
    point_contrasts = probability_contrasts(x, beta, sample["survey_weight"])

    replicate_betas = []
    replicate_contrasts = []
    failed = 0
    for cycle in sorted(sample["cycle"].unique()):
        cycle_mask = sample["cycle"].eq(cycle)
        for r in range(1, REPLICATES + 1):
            w = sample["survey_weight"].copy()
            w.loc[cycle_mask] = sample.loc[cycle_mask, f"rep_weight_{r}"]
            try:
                b_r = fit_weighted_logit(y, x, w)
                replicate_betas.append(b_r.reindex(beta.index).to_numpy(float))
                replicate_contrasts.append(probability_contrasts(x, b_r, w))
            except Exception:
                failed += 1

    rep_beta = np.asarray(replicate_betas)
    df = max(1, len(replicate_betas) - 2)
    critical = student_t.ppf(0.975, df)
    beta_se = np.sqrt(JK_COEF * np.sum((rep_beta - beta.to_numpy(float)) ** 2, axis=0))

    coef_rows = []
    for term in ["hcp_encourage", "telehealth_user", "hcp_x_telehealth"]:
        index = beta.index.get_loc(term)
        estimate = float(beta[term])
        se = float(beta_se[index])
        p_value = float(2 * student_t.sf(abs(estimate / se), df)) if se > 0 else np.nan
        coef_rows.append(
            {
                "model": model_label,
                "outcome": outcome,
                "term": term,
                "log_odds": estimate,
                "jk_standard_error": se,
                "odds_ratio": math.exp(estimate),
                "or_ci_lower": math.exp(estimate - critical * se),
                "or_ci_upper": math.exp(estimate + critical * se),
                "p_value": p_value,
                "degrees_of_freedom": df,
                "unweighted_n": len(sample),
                "events": int(y.sum()),
            }
        )

    contrast_rows = []
    for contrast, estimate in point_contrasts.items():
        values = np.asarray([row[contrast] for row in replicate_contrasts], dtype=float)
        se = math.sqrt(JK_COEF * np.sum((values - estimate) ** 2))
        contrast_rows.append(
            {
                "model": model_label,
                "outcome": outcome,
                "contrast": contrast,
                "estimate": estimate,
                "estimate_percentage_points": estimate * 100,
                "jk_standard_error": se,
                "ci_lower": estimate - critical * se,
                "ci_upper": estimate + critical * se,
                "p_value": float(2 * student_t.sf(abs(estimate / se), df)) if se > 0 else np.nan,
                "degrees_of_freedom": df,
                "unweighted_n": len(sample),
                "events": int(y.sum()),
            }
        )
    flow = {
        "model": model_label,
        "eligible_and_complete_n": len(sample),
        "events": int(y.sum()),
        "non_events": int((1 - y).sum()),
        "failed_replicate_fits": failed,
        "successful_replicate_fits": len(replicate_betas),
    }
    return pd.DataFrame(coef_rows), pd.DataFrame(contrast_rows), flow


def build_ml_matrix(data: pd.DataFrame) -> pd.DataFrame:
    x = build_design_matrix(data, include_cycle=False).drop(columns="const")
    # Health-app adoption is included only in prediction, not the adjusted association models.
    app = pd.to_numeric(data["health_app_user"], errors="coerce")
    x["health_app_user"] = app
    x["health_app_user_missing"] = app.isna().astype(float)
    return x.astype(float)


def fit_temporal_models(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_all = build_ml_matrix(data)
    specs = [
        ("Activation", "portal_use", data["portal_use"].notna()),
        ("Routinization", "routine_use", data["portal_use"].eq(1) & data["routine_use"].notna()),
        (
            "Functional depth",
            "deep_functional_use",
            data["portal_use"].eq(1) & data["deep_functional_use"].notna(),
        ),
    ]
    metric_rows = []
    shap_rows = []
    scenario_rows = []

    for label, outcome, eligible in specs:
        complete_key = data["hcp_encourage"].notna() & data["telehealth_user"].notna()
        train_mask = eligible & complete_key & data["cycle"].eq(2022)
        test_mask = eligible & complete_key & data["cycle"].eq(2024)
        x_train = x_all.loc[train_mask].copy()
        x_test = x_all.loc[test_mask].copy()
        y_train = data.loc[train_mask, outcome].astype(int)
        y_test = data.loc[test_mask, outcome].astype(int)
        w_train = data.loc[train_mask, "survey_weight"].to_numpy(float)
        w_test = data.loc[test_mask, "survey_weight"].to_numpy(float)
        w_train = w_train / np.mean(w_train)
        w_test = w_test / np.mean(w_test)

        imputer = SimpleImputer(strategy="median", add_indicator=False)
        x_train_imp = imputer.fit_transform(x_train)
        x_test_imp = imputer.transform(x_test)
        logistic = LogisticRegression(max_iter=2500, random_state=RANDOM_STATE)
        logistic.fit(x_train_imp, y_train, sample_weight=w_train)
        p_logit = logistic.predict_proba(x_test_imp)[:, 1]

        xgb = XGBClassifier(
            objective="binary:logistic",
            n_estimators=350,
            learning_rate=0.035,
            max_depth=3,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=2.0,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
        xgb.fit(x_train, y_train, sample_weight=w_train, verbose=False)
        p_xgb = xgb.predict_proba(x_test)[:, 1]
        xgb.save_model(MODEL_DIR / f"{outcome}_xgboost.json")

        for model_name, probability in [("Weighted logistic", p_logit), ("Weighted XGBoost", p_xgb)]:
            metric_rows.append(
                {
                    "stage_model": label,
                    "outcome": outcome,
                    "algorithm": model_name,
                    "training_cycle": 2022,
                    "test_cycle": 2024,
                    "training_n": int(train_mask.sum()),
                    "test_n": int(test_mask.sum()),
                    "test_weighted_prevalence": weighted_mean(y_test.to_numpy(float), w_test),
                    "weighted_roc_auc": roc_auc_score(y_test, probability, sample_weight=w_test),
                    "weighted_pr_auc": average_precision_score(y_test, probability, sample_weight=w_test),
                    "weighted_brier_score": brier_score_loss(y_test, probability, sample_weight=w_test),
                    "weighted_balanced_accuracy_0_5": balanced_accuracy_score(
                        y_test, probability >= 0.5, sample_weight=w_test
                    ),
                }
            )

        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(x_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap_values = np.asarray(shap_values)
        importance = np.average(np.abs(shap_values), axis=0, weights=w_test)
        signed = np.average(shap_values, axis=0, weights=w_test)
        order = np.argsort(importance)[::-1]
        for rank, idx in enumerate(order, start=1):
            shap_rows.append(
                {
                    "stage_model": label,
                    "outcome": outcome,
                    "feature": x_test.columns[idx],
                    "rank": rank,
                    "weighted_mean_absolute_shap": float(importance[idx]),
                    "weighted_mean_signed_shap": float(signed[idx]),
                }
            )

        for hcp, telehealth, scenario in [
            (0, 0, "Neither"),
            (1, 0, "HCP encouragement only"),
            (0, 1, "Telehealth only"),
            (1, 1, "Both"),
        ]:
            x_cf = x_test.copy()
            x_cf["hcp_encourage"] = hcp
            x_cf["telehealth_user"] = telehealth
            x_cf["hcp_x_telehealth"] = hcp * telehealth
            probability = xgb.predict_proba(x_cf)[:, 1]
            scenario_rows.append(
                {
                    "stage_model": label,
                    "outcome": outcome,
                    "scenario": scenario,
                    "predicted_probability": weighted_mean(probability, w_test),
                    "test_cycle": 2024,
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(shap_rows), pd.DataFrame(scenario_rows)


def create_figures(
    prevalence: pd.DataFrame,
    contrasts: pd.DataFrame,
    shap_importance: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> None:
    set_publication_plot_style()
    colors = [PLOT_COLORS["navy"], PLOT_COLORS["orange"], PLOT_COLORS["green"], PLOT_COLORS["purple"]]

    stage = prevalence.loc[prevalence["measure_type"].eq("journey_stage")].copy()
    stage["estimate_percent"] = stage["estimate"] * 100
    stage["lower_percent"] = stage["ci_lower"] * 100
    stage["upper_percent"] = stage["ci_upper"] * 100
    fig, ax = plt.subplots(figsize=(12.6, 7.2))
    labels = list(stage["measure"].drop_duplicates())
    display_labels = [
        "No recent\nportal use",
        "Episodic use\n(1--2)",
        "Routine use\n(3+)",
        "Organizer-app\nintegration",
    ]
    x = np.arange(len(labels))
    width = 0.35
    for j, cycle in enumerate([2022, 2024]):
        subset = stage.loc[stage["cycle"].eq(cycle)].set_index("measure").reindex(labels)
        position = x + (j - 0.5) * width
        yerr = np.vstack(
            [
                subset["estimate_percent"] - subset["lower_percent"],
                subset["upper_percent"] - subset["estimate_percent"],
            ]
        )
        bars = ax.bar(
            position,
            subset["estimate_percent"],
            width,
            color=colors[j],
            edgecolor="white",
            linewidth=0.8,
            label=str(cycle),
            zorder=3,
        )
        ax.errorbar(
            position,
            subset["estimate_percent"],
            yerr=yerr,
            fmt="none",
            ecolor="#374151",
            elinewidth=1.8,
            capsize=5,
            capthick=1.8,
            zorder=4,
        )
        for bar, value in zip(bars, subset["estimate_percent"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.6,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12.5,
                fontweight="semibold",
                color="#263238",
            )
    ax.set_xticks(x, display_labels)
    ax.set_ylabel("Survey-weighted percentage")
    ax.set_title("Digital Health Adoption Journey by HINTS Cycle", pad=15)
    ax.set_ylim(0, stage["upper_percent"].max() + 8)
    ax.legend(title="Cycle", frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.text(
        0.06,
        0.015,
        "Note: Organizer-app eligibility differed between cycles; integration bars are descriptive and should not be interpreted as a temporal contrast.",
        fontsize=11.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0.03, 0.075, 1, 1])
    save_publication_figure(fig, "figure1_adoption_journey.png")
    plt.close(fig)

    forest = contrasts.loc[
        contrasts["contrast"].isin(["AME_HCP_encouragement", "AME_telehealth_use", "additive_interaction"])
    ].copy()
    forest["estimate_pp"] = forest["estimate"] * 100
    forest["lower_pp"] = forest["ci_lower"] * 100
    forest["upper_pp"] = forest["ci_upper"] * 100
    models = ["Activation", "Routinization", "Functional depth", "Strict integration"]
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.7), sharey=True)
    titles = {
        "AME_HCP_encouragement": "Clinician encouragement",
        "AME_telehealth_use": "Telehealth use",
        "additive_interaction": "Additive interaction",
    }
    panel_colors = [PLOT_COLORS["navy"], PLOT_COLORS["orange"], PLOT_COLORS["purple"]]
    for ax, contrast, panel_color in zip(axes, titles, panel_colors):
        subset = forest.loc[forest["contrast"].eq(contrast)].set_index("model").reindex(models)
        y = np.arange(len(models))
        ax.errorbar(
            subset["estimate_pp"],
            y,
            xerr=np.vstack(
                [subset["estimate_pp"] - subset["lower_pp"], subset["upper_pp"] - subset["estimate_pp"]]
            ),
            fmt="o",
            markersize=8.5,
            color=panel_color,
            ecolor="#4B5563",
            elinewidth=2.1,
            capsize=5,
            capthick=1.8,
        )
        ax.axvline(0, color="#6B7280", linewidth=1.3, linestyle="--")
        ax.set_title(titles[contrast])
        ax.set_xlabel("Average marginal effect (pp)")
        ax.set_yticks(y, models)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Survey-Weighted Sequential Adoption Models", y=1.015)
    fig.tight_layout(w_pad=2.2)
    save_publication_figure(fig, "figure2_sequential_marginal_effects.png")
    plt.close(fig)

    feature_labels = {
        "hcp_encourage": "Clinician encouragement",
        "telehealth_user": "Telehealth use",
        "hcp_x_telehealth": "Encouragement × telehealth",
        "health_app_user": "Health-app use",
        "provider_visits": "Provider visits",
        "internet_user": "Internet use",
        "smartphone_user": "Smartphone use",
        "female": "Female",
        "comorbidity_count": "Chronic-condition count",
        "general_health": "General health",
        "education_2": "High school graduate",
        "education_4": "College graduate or more",
        "age_group_2": "Age 35--49",
        "age_group_3": "Age 50--64",
        "income_group_6": "Income category 6",
        "income_group_8": "Income category 8",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18.4, 7.3))
    for ax, label in zip(axes, ["Activation", "Routinization", "Functional depth"]):
        subset = shap_importance.loc[shap_importance["stage_model"].eq(label)].nsmallest(
            10, "rank"
        )
        subset = subset.sort_values("weighted_mean_absolute_shap")
        readable = [feature_labels.get(value, value.replace("_", " ").title()) for value in subset["feature"]]
        bars = ax.barh(
            readable,
            subset["weighted_mean_absolute_shap"],
            color=PLOT_COLORS["navy"],
            alpha=0.92,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_title(label)
        ax.set_xlabel("Weighted mean |SHAP value|")
        ax.tick_params(axis="y", labelsize=12.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", visible=False)
        for bar, value in zip(bars, subset["weighted_mean_absolute_shap"]):
            ax.text(
                value + subset["weighted_mean_absolute_shap"].max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                fontsize=11.5,
                color="#374151",
            )
        ax.set_xlim(0, subset["weighted_mean_absolute_shap"].max() * 1.20)
    fig.suptitle("Main XGBoost Predictors in the HINTS 7 Temporal Test Set", y=1.015)
    fig.tight_layout(w_pad=2.7)
    save_publication_figure(fig, "figure3_shap_importance.png")
    plt.close(fig)

    scenario_order = ["Neither", "HCP encouragement only", "Telehealth only", "Both"]
    plot_data = scenarios.copy()
    plot_data["predicted_percent"] = plot_data["predicted_probability"] * 100
    fig, ax = plt.subplots(figsize=(12.6, 7.0))
    annotation_y_offsets = [12, 10, -20]
    annotation_x_offsets = [-7, 0, 7]
    for i, label in enumerate(["Activation", "Routinization", "Functional depth"]):
        subset = plot_data.loc[plot_data["stage_model"].eq(label)].set_index("scenario").reindex(scenario_order)
        ax.plot(
            np.arange(len(scenario_order)),
            subset["predicted_percent"],
            marker="o",
            markersize=8.5,
            linewidth=2.7,
            label=label,
            color=colors[i],
        )
        for xpos, value in enumerate(subset["predicted_percent"]):
            ax.annotate(
                f"{value:.1f}%",
                (xpos, value),
                xytext=(annotation_x_offsets[i], annotation_y_offsets[i]),
                textcoords="offset points",
                ha="right" if annotation_x_offsets[i] < 0 else ("left" if annotation_x_offsets[i] > 0 else "center"),
                va="bottom" if annotation_y_offsets[i] > 0 else "top",
                fontsize=11.5,
                color=colors[i],
            )
    ax.set_xticks(
        np.arange(len(scenario_order)),
        ["Neither", "Clinician encouragement\nonly", "Telehealth only", "Both"],
    )
    ax.set_ylabel("Survey-weighted predicted probability (%)")
    ax.set_title("XGBoost Prediction Scenarios in HINTS 7", pad=15)
    ax.legend(title="Adoption stage", frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.13)
    fig.tight_layout()
    save_publication_figure(fig, "figure4_hcp_telehealth_scenarios.png")
    plt.close(fig)


def main() -> None:
    paths = {
        2022: find_source_data("hints6_public.dta"),
        2024: find_source_data("hints7_public.dta"),
    }
    data = pd.concat([harmonize(path, cycle) for cycle, path in paths.items()], ignore_index=True)
    data = scale_survey_weights(data)

    processed_columns = [
        "cycle",
        "HHID",
        "journey_stage",
        "portal_use",
        "routine_use",
        "high_frequency_use",
        "deep_functional_use",
        "any_functional_use",
        "viewed_results",
        "viewed_notes",
        "organizer_use",
        "integration_eligible",
        "access_frequency_code",
        "offered_any_portal",
        "portal_source_count_common",
        "multiple_portal_sources",
        "hcp_encourage",
        "telehealth_user",
        "hcp_x_telehealth",
        "age_group",
        "age_65_plus",
        "female",
        "education",
        "low_education",
        "race_ethnicity",
        "income_group",
        "low_income",
        "insured",
        "provider_visits",
        "rural",
        "general_health",
        "internet_user",
        "smartphone_user",
        "health_app_user",
        "comorbidity_count",
        "survey_weight",
    ] + [f"rep_weight_{r}" for r in range(1, REPLICATES + 1)]
    data[processed_columns].to_csv(PROCESSED_DIR / "hints6_7_harmonized_analysis.csv", index=False)

    prevalence_rows = []
    stage_labels = {
        0: "0 No recent portal use",
        1: "1 Episodic use (1-2)",
        2: "2 Routine use (3+)",
        3: "3 Organizer-app integration",
    }
    measures = {
        "portal_use": "Any portal use",
        "routine_use": "Routine use among users",
        "deep_functional_use": "Results and notes among users",
    }
    for cycle in [2022, 2024]:
        cycle_data = data.loc[data["cycle"].eq(cycle)].copy()
        for code, label in stage_labels.items():
            indicator = np.where(
                cycle_data["journey_stage"].notna(), cycle_data["journey_stage"].eq(code).astype(float), np.nan
            )
            result = cycle_jk_proportion(cycle_data, pd.Series(indicator, index=cycle_data.index))
            prevalence_rows.append(
                {"cycle": cycle, "measure_type": "journey_stage", "measure": label, **result}
            )
        for variable, label in measures.items():
            indicator = cycle_data[variable]
            if variable in ["routine_use", "deep_functional_use"]:
                indicator = indicator.where(cycle_data["portal_use"].eq(1))
            result = cycle_jk_proportion(cycle_data, indicator)
            prevalence_rows.append(
                {"cycle": cycle, "measure_type": "adoption_gate", "measure": label, **result}
            )
        integration_indicator = cycle_data["organizer_use"].where(cycle_data["integration_eligible"])
        result = cycle_jk_proportion(cycle_data, integration_indicator)
        prevalence_rows.append(
            {
                "cycle": cycle,
                "measure_type": "adoption_gate",
                "measure": "Organizer app among integration-eligible users",
                **result,
            }
        )
    prevalence = pd.DataFrame(prevalence_rows)
    prevalence.to_csv(TABLE_DIR / "table1_weighted_prevalence.csv", index=False)

    # Independent-cycle differences are restricted to identically worded/common gates.
    comparable = prevalence.loc[
        prevalence["measure_type"].eq("adoption_gate")
        & prevalence["measure"].isin(
            ["Any portal use", "Routine use among users", "Results and notes among users"]
        )
    ].copy()
    wide = comparable.pivot(index="measure", columns="cycle")
    cycle_difference_rows = []
    critical_difference = student_t.ppf(0.975, 98)
    for measure in wide.index:
        estimate = float(wide.loc[measure, ("estimate", 2024)] - wide.loc[measure, ("estimate", 2022)])
        se = math.sqrt(
            float(wide.loc[measure, ("standard_error", 2024)]) ** 2
            + float(wide.loc[measure, ("standard_error", 2022)]) ** 2
        )
        cycle_difference_rows.append(
            {
                "measure": measure,
                "estimate_2022": float(wide.loc[measure, ("estimate", 2022)]),
                "estimate_2024": float(wide.loc[measure, ("estimate", 2024)]),
                "difference_2024_minus_2022": estimate,
                "difference_percentage_points": estimate * 100,
                "standard_error": se,
                "ci_lower": estimate - critical_difference * se,
                "ci_upper": estimate + critical_difference * se,
                "p_value": float(2 * student_t.sf(abs(estimate / se), 98)),
                "degrees_of_freedom": 98,
            }
        )
    pd.DataFrame(cycle_difference_rows).to_csv(
        TABLE_DIR / "table1b_cycle_differences.csv", index=False
    )

    model_specs = [
        ("Activation", "portal_use", data["portal_use"].notna()),
        ("Routinization", "routine_use", data["portal_use"].eq(1)),
        ("Functional depth", "deep_functional_use", data["portal_use"].eq(1)),
        ("Strict integration", "organizer_use", data["integration_eligible"]),
    ]
    coefficient_tables = []
    contrast_tables = []
    flow_rows = []
    for label, outcome, eligibility in model_specs:
        coefficients, contrasts, flow = survey_logit_with_replicates(data, outcome, label, eligibility)
        coefficient_tables.append(coefficients)
        contrast_tables.append(contrasts)
        flow_rows.append(flow)
    coefficients = pd.concat(coefficient_tables, ignore_index=True)
    contrasts = pd.concat(contrast_tables, ignore_index=True)
    flow = pd.DataFrame(flow_rows)
    coefficients.to_csv(TABLE_DIR / "table2_survey_logit_key_effects.csv", index=False)
    contrasts.to_csv(TABLE_DIR / "table3_average_marginal_effects.csv", index=False)
    flow.to_csv(TABLE_DIR / "table4_analysis_sample_flow.csv", index=False)

    metrics, shap_importance, scenarios = fit_temporal_models(data)
    metrics.to_csv(TABLE_DIR / "table5_temporal_validation_metrics.csv", index=False)
    shap_importance.to_csv(TABLE_DIR / "table6_shap_importance.csv", index=False)
    scenarios.to_csv(TABLE_DIR / "table7_ml_scenario_probabilities.csv", index=False)

    create_figures(prevalence, contrasts, shap_importance, scenarios)

    summary = {
        "raw_rows": {str(c): int(data["cycle"].eq(c).sum()) for c in [2022, 2024]},
        "total_rows": len(data),
        "models": flow.to_dict(orient="records"),
        "output_tables": sorted(p.name for p in TABLE_DIR.glob("*.csv")),
        "output_figures": sorted(p.name for p in FIGURE_DIR.glob("*.png")),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
