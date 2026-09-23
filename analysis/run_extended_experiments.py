from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "tmp" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.special import expit, logit
from scipy.stats import t as student_t
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import run_experiment as base


TABLE_DIR = PROJECT / "outputs" / "tables"
FIGURE_DIR = PROJECT / "outputs" / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


ADJUSTMENT_COLUMNS = [
    "age_group",
    "female",
    "education",
    "race_ethnicity",
    "income_group",
    "insured",
    "provider_visits",
    "rural",
    "general_health",
    "comorbidity_count",
    "internet_user",
    "smartphone_user",
]


def load_data() -> pd.DataFrame:
    paths = {
        2022: base.find_source_data("hints6_public.dta"),
        2024: base.find_source_data("hints7_public.dta"),
    }
    data = pd.concat(
        [base.harmonize(path, cycle) for cycle, path in paths.items()],
        ignore_index=True,
    )
    return base.scale_survey_weights(data)


def replicate_weight_vectors(sample: pd.DataFrame):
    for cycle in sorted(sample["cycle"].unique()):
        cycle_mask = sample["cycle"].eq(cycle)
        for replicate in range(1, base.REPLICATES + 1):
            weights = sample["survey_weight"].copy()
            weights.loc[cycle_mask] = sample.loc[cycle_mask, f"rep_weight_{replicate}"]
            yield int(cycle), replicate, weights


def jk_summary(point: float, replicates: list[float], df: int) -> dict[str, float]:
    values = np.asarray(replicates, dtype=float)
    se = math.sqrt(base.JK_COEF * np.sum((values - point) ** 2))
    critical = student_t.ppf(0.975, df)
    return {
        "estimate": point,
        "jk_standard_error": se,
        "ci_lower": point - critical * se,
        "ci_upper": point + critical * se,
        "p_value": float(2 * student_t.sf(abs(point / se), df)) if se > 0 else np.nan,
        "degrees_of_freedom": df,
    }


def fit_replicated_model(
    data: pd.DataFrame,
    outcome: str,
    eligibility: pd.Series,
    design_builder,
    contrast_function=None,
) -> dict:
    mask = (
        eligibility
        & data[outcome].notna()
        & data["hcp_encourage"].notna()
        & data["telehealth_user"].notna()
        & data["survey_weight"].notna()
    )
    sample = data.loc[mask].copy()
    x = design_builder(sample).astype(float)
    y = sample[outcome].astype(float)
    beta = base.fit_weighted_logit(y, x, sample["survey_weight"])
    point_contrasts = (
        contrast_function(sample, x, beta, sample["survey_weight"])
        if contrast_function is not None
        else {}
    )

    replicate_betas = []
    replicate_contrasts = []
    failed = []
    for cycle, replicate, weights in replicate_weight_vectors(sample):
        try:
            beta_r = base.fit_weighted_logit(y, x, weights)
            replicate_betas.append(beta_r.reindex(beta.index).to_numpy(float))
            if contrast_function is not None:
                replicate_contrasts.append(
                    contrast_function(sample, x, beta_r, weights)
                )
        except Exception as exc:
            failed.append({"cycle": cycle, "replicate": replicate, "error": str(exc)})

    expected = base.REPLICATES * sample["cycle"].nunique()
    if len(replicate_betas) != expected:
        raise RuntimeError(
            f"{outcome}: {len(replicate_betas)}/{expected} replicate fits succeeded; "
            f"failures={failed[:3]}"
        )
    df = len(replicate_betas) - sample["cycle"].nunique()
    rep_beta = np.asarray(replicate_betas, dtype=float)
    beta_se = np.sqrt(
        base.JK_COEF * np.sum((rep_beta - beta.to_numpy(float)) ** 2, axis=0)
    )
    return {
        "sample": sample,
        "x": x,
        "y": y,
        "beta": beta,
        "beta_se": pd.Series(beta_se, index=beta.index),
        "point_contrasts": point_contrasts,
        "replicate_contrasts": replicate_contrasts,
        "df": int(df),
        "replicate_count": len(replicate_betas),
        "failed": failed,
    }


def coefficient_rows(fit: dict, model: str, outcome: str, terms=None) -> list[dict]:
    beta = fit["beta"]
    se = fit["beta_se"]
    df = fit["df"]
    critical = student_t.ppf(0.975, df)
    rows = []
    selected = list(beta.index) if terms is None else [term for term in terms if term in beta.index]
    for term in selected:
        estimate = float(beta[term])
        standard_error = float(se[term])
        rows.append(
            {
                "model": model,
                "outcome": outcome,
                "term": term,
                "log_odds": estimate,
                "jk_standard_error": standard_error,
                "odds_ratio": math.exp(estimate),
                "or_ci_lower": math.exp(estimate - critical * standard_error),
                "or_ci_upper": math.exp(estimate + critical * standard_error),
                "p_value": float(
                    2 * student_t.sf(abs(estimate / standard_error), df)
                )
                if standard_error > 0
                else np.nan,
                "degrees_of_freedom": df,
                "unweighted_n": len(fit["sample"]),
                "events": int(fit["y"].sum()),
                "replicate_fits": fit["replicate_count"],
            }
        )
    return rows


def contrast_rows(fit: dict, model: str, outcome: str) -> list[dict]:
    rows = []
    for name, point in fit["point_contrasts"].items():
        values = [row[name] for row in fit["replicate_contrasts"]]
        stats = jk_summary(float(point), values, fit["df"])
        rows.append(
            {
                "model": model,
                "outcome": outcome,
                "contrast": name,
                **stats,
                "estimate_percentage_points": stats["estimate"] * 100,
                "ci_lower_percentage_points": stats["ci_lower"] * 100,
                "ci_upper_percentage_points": stats["ci_upper"] * 100,
                "unweighted_n": len(fit["sample"]),
                "events": int(fit["y"].sum()),
                "replicate_fits": fit["replicate_count"],
            }
        )
    return rows


def main_design(sample: pd.DataFrame, include_cycle: bool = True) -> pd.DataFrame:
    return base.build_design_matrix(sample, include_cycle=include_cycle)


def main_contrasts(
    sample: pd.DataFrame,
    x: pd.DataFrame,
    beta: pd.Series,
    weights: pd.Series,
) -> dict[str, float]:
    return base.probability_contrasts(x, beta, weights)


def run_full_main_models(data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    specs = [
        ("Activation", "portal_use", data["portal_use"].notna()),
        ("Routinization", "routine_use", data["portal_use"].eq(1)),
        ("Functional depth", "deep_functional_use", data["portal_use"].eq(1)),
        ("Strict integration", "organizer_use", data["integration_eligible"]),
    ]
    rows = []
    diagnostics = []
    for model, outcome, eligibility in specs:
        fit = fit_replicated_model(
            data,
            outcome,
            eligibility,
            lambda sample: main_design(sample, include_cycle=True),
            main_contrasts,
        )
        rows.extend(coefficient_rows(fit, model, outcome, terms=None))
        diagnostics.append(
            {
                "experiment": "full_main_model",
                "model": model,
                "n": len(fit["sample"]),
                "events": int(fit["y"].sum()),
                "replicate_fits": fit["replicate_count"],
                "failed": len(fit["failed"]),
            }
        )
    return pd.DataFrame(rows), diagnostics


def cycle_interaction_design(sample: pd.DataFrame) -> pd.DataFrame:
    x = main_design(sample, include_cycle=True)
    x["hcp_x_cycle_2024"] = sample["hcp_encourage"].to_numpy(float) * sample[
        "cycle_2024"
    ].to_numpy(float)
    x["telehealth_x_cycle_2024"] = sample["telehealth_user"].to_numpy(float) * sample[
        "cycle_2024"
    ].to_numpy(float)
    return x


def predict_from_modified(x: pd.DataFrame, beta: pd.Series, updates: dict[str, np.ndarray | float]):
    modified = x.copy()
    for column, values in updates.items():
        if column in modified.columns:
            modified[column] = values
    return expit(modified.to_numpy(float) @ beta.reindex(modified.columns).to_numpy(float))


def cycle_interaction_contrasts(sample, x, beta, weights) -> dict[str, float]:
    output = {}
    hcp = sample["hcp_encourage"].to_numpy(float)
    tele = sample["telehealth_user"].to_numpy(float)
    cycle = sample["cycle_2024"].to_numpy(float)
    for year, cycle_value in [(2022, 0.0), (2024, 1.0)]:
        subset = sample["cycle"].eq(year).to_numpy()
        x_year = x.loc[subset].copy()
        w_year = weights.loc[sample.index[subset]].to_numpy(float)
        tele_year = tele[subset]
        hcp_year = hcp[subset]
        cycle_year = np.full(subset.sum(), cycle_value)

        p_h1 = predict_from_modified(
            x_year,
            beta,
            {
                "cycle_2024": cycle_year,
                "hcp_encourage": 1.0,
                "hcp_x_telehealth": tele_year,
                "hcp_x_cycle_2024": cycle_year,
                "telehealth_x_cycle_2024": tele_year * cycle_year,
            },
        )
        p_h0 = predict_from_modified(
            x_year,
            beta,
            {
                "cycle_2024": cycle_year,
                "hcp_encourage": 0.0,
                "hcp_x_telehealth": 0.0,
                "hcp_x_cycle_2024": 0.0,
                "telehealth_x_cycle_2024": tele_year * cycle_year,
            },
        )
        p_t1 = predict_from_modified(
            x_year,
            beta,
            {
                "cycle_2024": cycle_year,
                "telehealth_user": 1.0,
                "hcp_x_telehealth": hcp_year,
                "hcp_x_cycle_2024": hcp_year * cycle_year,
                "telehealth_x_cycle_2024": cycle_year,
            },
        )
        p_t0 = predict_from_modified(
            x_year,
            beta,
            {
                "cycle_2024": cycle_year,
                "telehealth_user": 0.0,
                "hcp_x_telehealth": 0.0,
                "hcp_x_cycle_2024": hcp_year * cycle_year,
                "telehealth_x_cycle_2024": 0.0,
            },
        )
        output[f"HCP_AME_{year}"] = base.weighted_mean(p_h1 - p_h0, w_year)
        output[f"telehealth_AME_{year}"] = base.weighted_mean(p_t1 - p_t0, w_year)
    output["HCP_AME_change_2024_minus_2022"] = output["HCP_AME_2024"] - output[
        "HCP_AME_2022"
    ]
    output["telehealth_AME_change_2024_minus_2022"] = output[
        "telehealth_AME_2024"
    ] - output["telehealth_AME_2022"]
    return output


def run_cycle_interactions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    specs = [
        ("Activation", "portal_use", data["portal_use"].notna()),
        ("Routinization", "routine_use", data["portal_use"].eq(1)),
        ("Functional depth", "deep_functional_use", data["portal_use"].eq(1)),
    ]
    coefficient_output = []
    effect_output = []
    diagnostics = []
    for model, outcome, eligibility in specs:
        fit = fit_replicated_model(
            data,
            outcome,
            eligibility,
            cycle_interaction_design,
            cycle_interaction_contrasts,
        )
        coefficient_output.extend(
            coefficient_rows(
                fit,
                model,
                outcome,
                terms=["hcp_x_cycle_2024", "telehealth_x_cycle_2024"],
            )
        )
        effect_output.extend(contrast_rows(fit, model, outcome))
        diagnostics.append(
            {
                "experiment": "cycle_interaction",
                "model": model,
                "n": len(fit["sample"]),
                "events": int(fit["y"].sum()),
                "replicate_fits": fit["replicate_count"],
                "failed": len(fit["failed"]),
            }
        )
    return pd.DataFrame(coefficient_output), pd.DataFrame(effect_output), diagnostics


def heterogeneity_design(sample: pd.DataFrame, modifier: str) -> pd.DataFrame:
    x = main_design(sample, include_cycle=True)
    x[f"hcp_x_{modifier}"] = (
        sample["hcp_encourage"].to_numpy(float) * sample[modifier].to_numpy(float)
    )
    return x


def heterogeneity_contrast_function(modifier: str):
    interaction = f"hcp_x_{modifier}"

    def calculate(sample, x, beta, weights):
        tele = sample["telehealth_user"].to_numpy(float)
        modifier_values = sample[modifier].to_numpy(float)
        p1 = predict_from_modified(
            x,
            beta,
            {
                "hcp_encourage": 1.0,
                "hcp_x_telehealth": tele,
                interaction: modifier_values,
            },
        )
        p0 = predict_from_modified(
            x,
            beta,
            {
                "hcp_encourage": 0.0,
                "hcp_x_telehealth": 0.0,
                interaction: 0.0,
            },
        )
        output = {}
        for level in [0, 1]:
            subset = modifier_values == level
            output[f"HCP_AME_modifier_{level}"] = base.weighted_mean(
                (p1 - p0)[subset], weights.to_numpy(float)[subset]
            )
        output["HCP_AME_difference_1_minus_0"] = (
            output["HCP_AME_modifier_1"] - output["HCP_AME_modifier_0"]
        )
        return output

    return calculate


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return pd.Series(output, index=p_values.index)


def run_heterogeneity(data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    specs = [
        ("Activation", "portal_use", data["portal_use"].notna()),
        ("Routinization", "routine_use", data["portal_use"].eq(1)),
        ("Functional depth", "deep_functional_use", data["portal_use"].eq(1)),
    ]
    modifiers = {
        "age_65_plus": ("Age 65+", "Age <65", "Age 65+"),
        "low_education": ("High school or less", "Some college or more", "High school or less"),
        "low_income": ("Household income <$50,000", "Household income >=$50,000", "Household income <$50,000"),
        "rural": ("Nonmetropolitan residence", "Metropolitan", "Nonmetropolitan"),
    }
    rows = []
    diagnostics = []
    for model, outcome, base_eligibility in specs:
        for modifier, (label, level0, level1) in modifiers.items():
            eligibility = base_eligibility & data[modifier].notna()
            fit = fit_replicated_model(
                data,
                outcome,
                eligibility,
                lambda sample, modifier=modifier: heterogeneity_design(sample, modifier),
                heterogeneity_contrast_function(modifier),
            )
            coefficient = coefficient_rows(
                fit, model, outcome, terms=[f"hcp_x_{modifier}"]
            )[0]
            contrast_lookup = {
                row["contrast"]: row for row in contrast_rows(fit, model, outcome)
            }
            for contrast, level_label in [
                ("HCP_AME_modifier_0", level0),
                ("HCP_AME_modifier_1", level1),
                ("HCP_AME_difference_1_minus_0", f"Difference: {level1} minus {level0}"),
            ]:
                row = contrast_lookup[contrast]
                rows.append(
                    {
                        "stage_model": model,
                        "outcome": outcome,
                        "modifier": modifier,
                        "modifier_label": label,
                        "effect": contrast,
                        "level_label": level_label,
                        "estimate": row["estimate"],
                        "estimate_percentage_points": row["estimate_percentage_points"],
                        "jk_standard_error": row["jk_standard_error"],
                        "ci_lower": row["ci_lower"],
                        "ci_upper": row["ci_upper"],
                        "p_value": row["p_value"],
                        "multiplicative_interaction_or": coefficient["odds_ratio"],
                        "multiplicative_interaction_p": coefficient["p_value"],
                        "unweighted_n": row["unweighted_n"],
                        "events": row["events"],
                    }
                )
            diagnostics.append(
                {
                    "experiment": "heterogeneity",
                    "model": model,
                    "modifier": modifier,
                    "n": len(fit["sample"]),
                    "events": int(fit["y"].sum()),
                    "replicate_fits": fit["replicate_count"],
                    "failed": len(fit["failed"]),
                }
            )
    result = pd.DataFrame(rows)
    interaction_rows = result["effect"].eq("HCP_AME_difference_1_minus_0")
    unique_tests = result.loc[interaction_rows].copy()
    unique_tests["interaction_p_fdr"] = benjamini_hochberg(
        unique_tests["multiplicative_interaction_p"]
    )
    fdr_map = unique_tests.set_index(["stage_model", "modifier"])["interaction_p_fdr"]
    result["multiplicative_interaction_p_fdr"] = [
        fdr_map.loc[(stage, modifier)]
        for stage, modifier in zip(result["stage_model"], result["modifier"])
    ]
    return result, diagnostics


def run_sensitivity(data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    complete_covariates = data[ADJUSTMENT_COLUMNS].notna().all(axis=1)
    specs = [
        (
            "Activation among respondents offered a portal",
            "portal_use",
            data["portal_use"].notna() & data["offered_any_portal"].eq(1),
            True,
        ),
        (
            "High-frequency use (6+ accesses) among portal users",
            "high_frequency_use",
            data["portal_use"].eq(1),
            True,
        ),
        (
            "Any functional use (results or notes) among portal users",
            "any_functional_use",
            data["portal_use"].eq(1),
            True,
        ),
        (
            "Activation complete-case covariates",
            "portal_use",
            data["portal_use"].notna() & complete_covariates,
            True,
        ),
        (
            "Routinization complete-case covariates",
            "routine_use",
            data["portal_use"].eq(1) & complete_covariates,
            True,
        ),
        (
            "Functional depth complete-case covariates",
            "deep_functional_use",
            data["portal_use"].eq(1) & complete_covariates,
            True,
        ),
        (
            "HINTS 7 organizer use among all portal users",
            "organizer_use",
            data["cycle"].eq(2024) & data["portal_use"].eq(1),
            False,
        ),
    ]
    rows = []
    diagnostics = []
    selected_contrasts = [
        "AME_HCP_encouragement",
        "AME_telehealth_use",
        "additive_interaction",
    ]
    for model, outcome, eligibility, include_cycle in specs:
        fit = fit_replicated_model(
            data,
            outcome,
            eligibility,
            lambda sample, include_cycle=include_cycle: main_design(
                sample, include_cycle=include_cycle
            ),
            main_contrasts,
        )
        for row in contrast_rows(fit, model, outcome):
            if row["contrast"] in selected_contrasts:
                rows.append(row)
        diagnostics.append(
            {
                "experiment": "sensitivity",
                "model": model,
                "n": len(fit["sample"]),
                "events": int(fit["y"].sum()),
                "replicate_fits": fit["replicate_count"],
                "failed": len(fit["failed"]),
            }
        )
    return pd.DataFrame(rows), diagnostics


def run_cycle_specific_integration(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    coefficient_output = []
    effect_output = []
    diagnostics = []
    for cycle in [2022, 2024]:
        eligibility = data["cycle"].eq(cycle) & data["integration_eligible"]
        label = f"Strict integration, HINTS {6 if cycle == 2022 else 7}"
        fit = fit_replicated_model(
            data,
            "organizer_use",
            eligibility,
            lambda sample: main_design(sample, include_cycle=False),
            main_contrasts,
        )
        coefficient_output.extend(
            coefficient_rows(
                fit,
                label,
                "organizer_use",
                terms=["hcp_encourage", "telehealth_user", "hcp_x_telehealth"],
            )
        )
        effect_output.extend(contrast_rows(fit, label, "organizer_use"))
        diagnostics.append(
            {
                "experiment": "cycle_specific_integration",
                "cycle": cycle,
                "n": len(fit["sample"]),
                "events": int(fit["y"].sum()),
                "replicate_fits": fit["replicate_count"],
                "failed": len(fit["failed"]),
            }
        )
    return pd.DataFrame(coefficient_output), pd.DataFrame(effect_output), diagnostics


def cycle_jk_mean(data: pd.DataFrame, values: pd.Series) -> dict[str, float]:
    valid = values.notna() & data["survey_weight"].notna()
    y = values.loc[valid].to_numpy(float)
    point = base.weighted_mean(y, data.loc[valid, "survey_weight"].to_numpy(float))
    replicates = [
        base.weighted_mean(y, data.loc[valid, f"rep_weight_{r}"].to_numpy(float))
        for r in range(1, base.REPLICATES + 1)
    ]
    stats = jk_summary(point, replicates, base.REPLICATES - 1)
    return {"unweighted_n": int(valid.sum()), **stats}


def characteristic_definitions(data: pd.DataFrame):
    definitions = []
    binary = [
        ("Digital access", "HCP encouraged portal use", "hcp_encourage"),
        ("Digital access", "Used telehealth in past 12 months", "telehealth_user"),
        ("Demographic", "Female", "female"),
        ("Healthcare", "Insured", "insured"),
        ("Geography", "Nonmetropolitan county", "rural"),
        ("Digital access", "Internet user", "internet_user"),
        ("Digital access", "Smartphone user", "smartphone_user"),
        ("Digital access", "Health or wellness app user", "health_app_user"),
    ]
    for section, label, column in binary:
        definitions.append((section, label, "Proportion", lambda d, c=column: d[c]))

    categories = {
        "age_group": {
            1: "Age 18-34",
            2: "Age 35-49",
            3: "Age 50-64",
            4: "Age 65-74",
            5: "Age 75+",
        },
        "education": {
            1: "Less than high school",
            2: "High school graduate",
            3: "Some college",
            4: "College graduate or more",
        },
        "race_ethnicity": {
            1: "Non-Hispanic White",
            2: "Non-Hispanic Black",
            3: "Hispanic",
            4: "Non-Hispanic Asian",
            5: "Non-Hispanic other race/ethnicity",
        },
    }
    sections = {
        "age_group": "Age",
        "education": "Education",
        "race_ethnicity": "Race and ethnicity",
    }
    for column, levels in categories.items():
        for code, label in levels.items():
            definitions.append(
                (
                    sections[column],
                    label,
                    "Proportion",
                    lambda d, c=column, value=code: np.where(
                        d[c].notna(), d[c].eq(value).astype(float), np.nan
                    ),
                )
            )
    definitions.extend(
        [
            (
                "Income",
                "Household income <$50,000",
                "Proportion",
                lambda d: np.where(
                    d["income_group"].notna(), d["income_group"].le(5).astype(float), np.nan
                ),
            ),
            (
                "Income",
                "Household income $50,000-$99,999",
                "Proportion",
                lambda d: np.where(
                    d["income_group"].notna(), d["income_group"].isin([6, 7]).astype(float), np.nan
                ),
            ),
            (
                "Income",
                "Household income >=$100,000",
                "Proportion",
                lambda d: np.where(
                    d["income_group"].notna(), d["income_group"].isin([8, 9]).astype(float), np.nan
                ),
            ),
            (
                "Healthcare",
                "Provider-visit frequency category (0-6), mean",
                "Mean",
                lambda d: d["provider_visits"],
            ),
            (
                "Health",
                "General health score (1=excellent, 5=poor), mean",
                "Mean",
                lambda d: d["general_health"],
            ),
            (
                "Health",
                "Chronic-condition count (0-6), mean",
                "Mean",
                lambda d: d["comorbidity_count"],
            ),
        ]
    )
    return definitions


def run_weighted_characteristics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    critical = student_t.ppf(0.975, 98)
    for section, label, measure_type, getter in characteristic_definitions(data):
        cycle_results = {}
        for cycle in [2022, 2024]:
            subset = data.loc[data["cycle"].eq(cycle)].copy()
            values = pd.Series(getter(subset), index=subset.index, dtype=float)
            cycle_results[cycle] = cycle_jk_mean(subset, values)
        difference = cycle_results[2024]["estimate"] - cycle_results[2022]["estimate"]
        se = math.sqrt(
            cycle_results[2022]["jk_standard_error"] ** 2
            + cycle_results[2024]["jk_standard_error"] ** 2
        )
        rows.append(
            {
                "section": section,
                "characteristic": label,
                "measure_type": measure_type,
                "unweighted_n_2022": cycle_results[2022]["unweighted_n"],
                "estimate_2022": cycle_results[2022]["estimate"],
                "standard_error_2022": cycle_results[2022]["jk_standard_error"],
                "unweighted_n_2024": cycle_results[2024]["unweighted_n"],
                "estimate_2024": cycle_results[2024]["estimate"],
                "standard_error_2024": cycle_results[2024]["jk_standard_error"],
                "difference_2024_minus_2022": difference,
                "difference_standard_error": se,
                "difference_ci_lower": difference - critical * se,
                "difference_ci_upper": difference + critical * se,
                "difference_p_value": float(2 * student_t.sf(abs(difference / se), 98))
                if se > 0
                else np.nan,
                "degrees_of_freedom": 98,
            }
        )
    return pd.DataFrame(rows)


def new_xgb() -> XGBClassifier:
    return XGBClassifier(
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
        random_state=base.RANDOM_STATE,
        n_jobs=4,
    )


def weighted_ece(y: np.ndarray, probability: np.ndarray, weights: np.ndarray, bins=10) -> float:
    frame = pd.DataFrame({"y": y, "p": probability, "w": weights})
    frame["bin"] = pd.qcut(frame["p"], q=bins, labels=False, duplicates="drop")
    total = frame["w"].sum()
    value = 0.0
    for _, group in frame.groupby("bin", observed=True):
        observed = np.average(group["y"], weights=group["w"])
        predicted = np.average(group["p"], weights=group["w"])
        value += group["w"].sum() / total * abs(observed - predicted)
    return float(value)


def calibration_parameters(y: np.ndarray, probability: np.ndarray, weights: np.ndarray):
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    x = sm.add_constant(logit(clipped), has_constant="add")
    result = sm.GLM(
        y,
        x,
        family=sm.families.Binomial(),
        freq_weights=weights,
    ).fit(maxiter=100, disp=0)
    return float(result.params[0]), float(result.params[1])


def metric_with_replicates(y, probability, point_weights, replicate_weights, metric):
    point = metric(y, probability, point_weights)
    replicates = [metric(y, probability, weights) for weights in replicate_weights]
    return jk_summary(float(point), [float(value) for value in replicates], base.REPLICATES - 1)


def run_temporal_validation_extended(data: pd.DataFrame):
    x_all = base.build_ml_matrix(data)
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
    comparison_rows = []
    calibration_rows = []
    cv_rows = []

    for stage, outcome, eligibility in specs:
        complete_key = data["hcp_encourage"].notna() & data["telehealth_user"].notna()
        train_mask = eligibility & complete_key & data["cycle"].eq(2022)
        test_mask = eligibility & complete_key & data["cycle"].eq(2024)
        x_train = x_all.loc[train_mask].copy()
        x_test = x_all.loc[test_mask].copy()
        y_train = data.loc[train_mask, outcome].astype(int).to_numpy()
        y_test = data.loc[test_mask, outcome].astype(int).to_numpy()
        train_weights = data.loc[train_mask, "survey_weight"].to_numpy(float).copy()
        test_weights = data.loc[test_mask, "survey_weight"].to_numpy(float).copy()
        train_weights /= np.mean(train_weights)
        test_weights /= np.mean(test_weights)
        test_rep_weights = [
            data.loc[test_mask, f"rep_weight_{r}"].to_numpy(float)
            for r in range(1, base.REPLICATES + 1)
        ]

        imputer = SimpleImputer(strategy="median", add_indicator=False)
        x_train_imp = imputer.fit_transform(x_train)
        x_test_imp = imputer.transform(x_test)
        logistic = LogisticRegression(max_iter=2500, random_state=base.RANDOM_STATE)
        logistic.fit(x_train_imp, y_train, sample_weight=train_weights)
        predictions = {
            "Weighted logistic": logistic.predict_proba(x_test_imp)[:, 1]
        }
        xgb = new_xgb()
        xgb.fit(x_train, y_train, sample_weight=train_weights, verbose=False)
        predictions["Weighted XGBoost"] = xgb.predict_proba(x_test)[:, 1]

        for algorithm, probability in predictions.items():
            auc = metric_with_replicates(
                y_test,
                probability,
                test_weights,
                test_rep_weights,
                lambda y, p, w: roc_auc_score(y, p, sample_weight=w),
            )
            pr_auc = metric_with_replicates(
                y_test,
                probability,
                test_weights,
                test_rep_weights,
                lambda y, p, w: average_precision_score(y, p, sample_weight=w),
            )
            brier = metric_with_replicates(
                y_test,
                probability,
                test_weights,
                test_rep_weights,
                lambda y, p, w: brier_score_loss(y, p, sample_weight=w),
            )
            intercept, slope = calibration_parameters(y_test, probability, test_weights)
            calibration_replicates = [
                calibration_parameters(y_test, probability, weights)
                for weights in test_rep_weights
            ]
            intercept_stats = jk_summary(
                intercept,
                [value[0] for value in calibration_replicates],
                base.REPLICATES - 1,
            )
            slope_stats = jk_summary(
                slope,
                [value[1] for value in calibration_replicates],
                base.REPLICATES - 1,
            )
            ece = metric_with_replicates(
                y_test,
                probability,
                test_weights,
                test_rep_weights,
                lambda y, p, w: weighted_ece(y, p, w),
            )
            metric_rows.append(
                {
                    "stage_model": stage,
                    "outcome": outcome,
                    "algorithm": algorithm,
                    "training_cycle": 2022,
                    "test_cycle": 2024,
                    "training_n": int(train_mask.sum()),
                    "test_n": int(test_mask.sum()),
                    "test_weighted_prevalence": base.weighted_mean(y_test.astype(float), test_weights),
                    "weighted_roc_auc": auc["estimate"],
                    "roc_auc_jk_standard_error": auc["jk_standard_error"],
                    "roc_auc_ci_lower": auc["ci_lower"],
                    "roc_auc_ci_upper": auc["ci_upper"],
                    "weighted_pr_auc": pr_auc["estimate"],
                    "pr_auc_ci_lower": pr_auc["ci_lower"],
                    "pr_auc_ci_upper": pr_auc["ci_upper"],
                    "weighted_brier_score": brier["estimate"],
                    "brier_ci_lower": brier["ci_lower"],
                    "brier_ci_upper": brier["ci_upper"],
                    "weighted_balanced_accuracy_0_5": balanced_accuracy_score(
                        y_test, probability >= 0.5, sample_weight=test_weights
                    ),
                    "calibration_intercept": intercept_stats["estimate"],
                    "calibration_intercept_se": intercept_stats["jk_standard_error"],
                    "calibration_intercept_ci_lower": intercept_stats["ci_lower"],
                    "calibration_intercept_ci_upper": intercept_stats["ci_upper"],
                    "calibration_slope": slope_stats["estimate"],
                    "calibration_slope_se": slope_stats["jk_standard_error"],
                    "calibration_slope_ci_lower": slope_stats["ci_lower"],
                    "calibration_slope_ci_upper": slope_stats["ci_upper"],
                    "weighted_ece_10_bins": ece["estimate"],
                    "ece_ci_lower": ece["ci_lower"],
                    "ece_ci_upper": ece["ci_upper"],
                    "replicate_weights": base.REPLICATES,
                    "degrees_of_freedom": base.REPLICATES - 1,
                }
            )

            calibration_frame = pd.DataFrame(
                {"observed": y_test, "predicted": probability, "weight": test_weights}
            )
            calibration_frame["decile"] = pd.qcut(
                calibration_frame["predicted"], q=10, labels=False, duplicates="drop"
            )
            for decile, group in calibration_frame.groupby("decile", observed=True):
                calibration_rows.append(
                    {
                        "stage_model": stage,
                        "outcome": outcome,
                        "algorithm": algorithm,
                        "decile": int(decile) + 1,
                        "unweighted_n": len(group),
                        "weighted_mean_predicted": np.average(
                            group["predicted"], weights=group["weight"]
                        ),
                        "weighted_observed_proportion": np.average(
                            group["observed"], weights=group["weight"]
                        ),
                    }
                )

        probability_logistic = predictions["Weighted logistic"]
        probability_xgb = predictions["Weighted XGBoost"]
        paired_metrics = {
            "ROC AUC": (
                lambda y, p, w: roc_auc_score(y, p, sample_weight=w),
                "higher_is_better",
            ),
            "PR AUC": (
                lambda y, p, w: average_precision_score(y, p, sample_weight=w),
                "higher_is_better",
            ),
            "Brier score": (
                lambda y, p, w: brier_score_loss(y, p, sample_weight=w),
                "lower_is_better",
            ),
        }
        for metric_name, (metric_function, direction) in paired_metrics.items():
            point_difference = metric_function(
                y_test, probability_xgb, test_weights
            ) - metric_function(y_test, probability_logistic, test_weights)
            replicate_differences = [
                metric_function(y_test, probability_xgb, weights)
                - metric_function(y_test, probability_logistic, weights)
                for weights in test_rep_weights
            ]
            stats = jk_summary(
                float(point_difference),
                [float(value) for value in replicate_differences],
                base.REPLICATES - 1,
            )
            comparison_rows.append(
                {
                    "stage_model": stage,
                    "outcome": outcome,
                    "metric": metric_name,
                    "comparison": "Weighted XGBoost minus weighted logistic",
                    "direction": direction,
                    **stats,
                    "replicate_weights": base.REPLICATES,
                }
            )

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=base.RANDOM_STATE)
        for fold, (fit_index, validation_index) in enumerate(
            folds.split(x_train, y_train), start=1
        ):
            x_fit = x_train.iloc[fit_index]
            x_validation = x_train.iloc[validation_index]
            y_fit = y_train[fit_index]
            y_validation = y_train[validation_index]
            w_fit = train_weights[fit_index]
            w_validation = train_weights[validation_index]

            fold_imputer = SimpleImputer(strategy="median", add_indicator=False)
            x_fit_imp = fold_imputer.fit_transform(x_fit)
            x_validation_imp = fold_imputer.transform(x_validation)
            fold_logistic = LogisticRegression(max_iter=2500, random_state=base.RANDOM_STATE)
            fold_logistic.fit(x_fit_imp, y_fit, sample_weight=w_fit)
            fold_predictions = {
                "Weighted logistic": fold_logistic.predict_proba(x_validation_imp)[:, 1]
            }
            fold_xgb = new_xgb()
            fold_xgb.fit(x_fit, y_fit, sample_weight=w_fit, verbose=False)
            fold_predictions["Weighted XGBoost"] = fold_xgb.predict_proba(x_validation)[:, 1]
            for algorithm, probability in fold_predictions.items():
                cv_rows.append(
                    {
                        "stage_model": stage,
                        "outcome": outcome,
                        "algorithm": algorithm,
                        "fold": fold,
                        "training_cycle": 2022,
                        "validation_n": len(validation_index),
                        "weighted_roc_auc": roc_auc_score(
                            y_validation, probability, sample_weight=w_validation
                        ),
                        "weighted_pr_auc": average_precision_score(
                            y_validation, probability, sample_weight=w_validation
                        ),
                        "weighted_brier_score": brier_score_loss(
                            y_validation, probability, sample_weight=w_validation
                        ),
                    }
                )

    cv = pd.DataFrame(cv_rows)
    summaries = (
        cv.groupby(["stage_model", "outcome", "algorithm"], as_index=False)
        .agg(
            validation_n=("validation_n", "sum"),
            weighted_roc_auc=("weighted_roc_auc", "mean"),
            roc_auc_fold_sd=("weighted_roc_auc", "std"),
            weighted_pr_auc=("weighted_pr_auc", "mean"),
            pr_auc_fold_sd=("weighted_pr_auc", "std"),
            weighted_brier_score=("weighted_brier_score", "mean"),
            brier_fold_sd=("weighted_brier_score", "std"),
        )
    )
    summaries["fold"] = "Mean (5 folds)"
    summaries["training_cycle"] = 2022
    cv["fold"] = cv["fold"].astype(str)
    cv = pd.concat([cv, summaries], ignore_index=True, sort=False)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(comparison_rows),
        pd.DataFrame(calibration_rows),
        cv,
    )


def create_extended_figures(
    cycle_effects: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    base.set_publication_plot_style()
    colors = {
        "Weighted logistic": base.PLOT_COLORS["navy"],
        "Weighted XGBoost": base.PLOT_COLORS["orange"],
    }

    plot = cycle_effects.loc[
        cycle_effects["contrast"].isin(
            ["HCP_AME_2022", "HCP_AME_2024", "telehealth_AME_2022", "telehealth_AME_2024"]
        )
    ].copy()
    plot["exposure"] = np.where(plot["contrast"].str.startswith("HCP"), "HCP encouragement", "Telehealth use")
    plot["cycle"] = plot["contrast"].str.extract(r"(2022|2024)").astype(int)
    plot["estimate_pp"] = plot["estimate"] * 100
    plot["lower_pp"] = plot["ci_lower"] * 100
    plot["upper_pp"] = plot["ci_upper"] * 100
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.6), sharey=True)
    for ax, exposure in zip(axes, ["HCP encouragement", "Telehealth use"]):
        subset = plot.loc[plot["exposure"].eq(exposure)]
        for cycle, color in [(2022, base.PLOT_COLORS["navy"]), (2024, base.PLOT_COLORS["orange"])]:
            values = subset.loc[subset["cycle"].eq(cycle)].set_index("model").reindex(
                ["Activation", "Routinization", "Functional depth"]
            )
            y = np.arange(3) + (-0.10 if cycle == 2022 else 0.10)
            ax.errorbar(
                values["estimate_pp"],
                y,
                xerr=np.vstack(
                    [values["estimate_pp"] - values["lower_pp"], values["upper_pp"] - values["estimate_pp"]]
                ),
                fmt="o",
                markersize=8.5,
                elinewidth=2.1,
                capsize=5,
                capthick=1.8,
                color=color,
                label=str(cycle),
            )
        ax.axvline(0, color="#6B7280", linewidth=1.3, linestyle="--")
        ax.set_title("Clinician encouragement" if exposure == "HCP encouragement" else exposure)
        ax.set_xlabel("Average marginal effect (pp)")
        ax.set_yticks(np.arange(3), ["Activation", "Routinization", "Functional depth"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Cycle",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Adjusted Associations by HINTS Cycle", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.89], w_pad=2.5)
    base.save_publication_figure(fig, "figure5_cycle_specific_marginal_effects.png")
    plt.close(fig)

    interaction = heterogeneity.loc[
        heterogeneity["effect"].eq("HCP_AME_difference_1_minus_0")
    ].copy()
    interaction["estimate_pp"] = interaction["estimate"] * 100
    interaction["lower_pp"] = interaction["ci_lower"] * 100
    interaction["upper_pp"] = interaction["ci_upper"] * 100
    modifier_order = ["age_65_plus", "low_education", "low_income", "rural"]
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.8), sharey=True)
    for ax, stage in zip(axes, ["Activation", "Routinization", "Functional depth"]):
        subset = interaction.loc[interaction["stage_model"].eq(stage)].set_index("modifier").reindex(modifier_order)
        y = np.arange(len(subset))
        for ypos, (_, row) in zip(y, subset.iterrows()):
            significant = float(row["multiplicative_interaction_p_fdr"]) <= 0.05
            point_color = base.PLOT_COLORS["orange"] if significant else base.PLOT_COLORS["green"]
            ax.errorbar(
                row["estimate_pp"],
                ypos,
                xerr=np.asarray(
                    [[row["estimate_pp"] - row["lower_pp"]], [row["upper_pp"] - row["estimate_pp"]]]
                ),
                fmt="o",
                markersize=8.5,
                elinewidth=2.0,
                capsize=5,
                capthick=1.7,
                color=point_color,
            )
        ax.axvline(0, color="#6B7280", linewidth=1.3, linestyle="--")
        ax.set_title(stage)
        ax.set_xlabel("Difference in encouragement AME (pp)")
        ax.set_yticks(y, subset["modifier_label"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", visible=False)
    axes[0].plot([], [], "o", color=base.PLOT_COLORS["orange"], label="FDR-significant")
    axes[0].plot([], [], "o", color=base.PLOT_COLORS["green"], label="Not FDR-significant")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Heterogeneity in the Association of Clinician Encouragement", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.89], w_pad=2.5)
    base.save_publication_figure(fig, "figure6_hcp_heterogeneity.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.2), sharex=True, sharey=True)
    for ax, stage in zip(axes, ["Activation", "Routinization", "Functional depth"]):
        subset = calibration.loc[calibration["stage_model"].eq(stage)]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#4B5563", linewidth=1.5, label="Ideal")
        for algorithm in ["Weighted logistic", "Weighted XGBoost"]:
            values = subset.loc[subset["algorithm"].eq(algorithm)].sort_values("weighted_mean_predicted")
            ax.plot(
                values["weighted_mean_predicted"],
                values["weighted_observed_proportion"],
                marker="o",
                markersize=6.8,
                linewidth=2.4,
                color=colors[algorithm],
                label=algorithm,
            )
        ax.set_title(stage)
        ax.set_xlabel("Weighted mean predicted probability")
        if ax is axes[0]:
            ax.set_ylabel("Weighted observed proportion")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_aspect("equal", adjustable="box")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Temporal Calibration in HINTS 7", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.88], w_pad=2.0)
    base.save_publication_figure(fig, "figure7_temporal_calibration.png")
    plt.close(fig)


def main() -> None:
    data = load_data()
    ml_only = "--ml-only" in sys.argv
    if ml_only:
        ml_metrics, ml_comparison, calibration, cross_validation = run_temporal_validation_extended(data)
        ml_metrics.to_csv(TABLE_DIR / "table14_temporal_validation_uncertainty_calibration.csv", index=False)
        ml_comparison.to_csv(TABLE_DIR / "table14b_paired_algorithm_comparisons.csv", index=False)
        cross_validation.to_csv(TABLE_DIR / "table15_hints6_five_fold_cross_validation.csv", index=False)
        calibration.to_csv(TABLE_DIR / "table16_calibration_curve_data.csv", index=False)
        cycle_effects = pd.read_csv(TABLE_DIR / "table9b_cycle_specific_marginal_effects.csv")
        heterogeneity = pd.read_csv(TABLE_DIR / "table11_heterogeneity_hcp_encouragement.csv")
        create_extended_figures(cycle_effects, heterogeneity, calibration)
        summary = {
            "mode": "ml_only",
            "total_rows": int(len(data)),
            "finite_auc": bool(np.isfinite(ml_metrics["weighted_roc_auc"]).all()),
            "finite_paired_differences": bool(np.isfinite(ml_comparison["estimate"]).all()),
            "tables": ["table14", "table14b", "table15", "table16"],
        }
        (PROJECT / "outputs" / "ml_extension_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return
    all_diagnostics = []

    full_coefficients, diagnostics = run_full_main_models(data)
    all_diagnostics.extend(diagnostics)
    full_coefficients.to_csv(TABLE_DIR / "table8_full_survey_logit_coefficients.csv", index=False)

    cycle_coefficients, cycle_effects, diagnostics = run_cycle_interactions(data)
    all_diagnostics.extend(diagnostics)
    cycle_coefficients.to_csv(TABLE_DIR / "table9a_cycle_interaction_coefficients.csv", index=False)
    cycle_effects.to_csv(TABLE_DIR / "table9b_cycle_specific_marginal_effects.csv", index=False)

    sensitivity, diagnostics = run_sensitivity(data)
    all_diagnostics.extend(diagnostics)
    sensitivity.to_csv(TABLE_DIR / "table10_sensitivity_analyses.csv", index=False)

    heterogeneity, diagnostics = run_heterogeneity(data)
    all_diagnostics.extend(diagnostics)
    heterogeneity.to_csv(TABLE_DIR / "table11_heterogeneity_hcp_encouragement.csv", index=False)

    integration_coefficients, integration_effects, diagnostics = run_cycle_specific_integration(data)
    all_diagnostics.extend(diagnostics)
    integration_coefficients.to_csv(
        TABLE_DIR / "table12a_integration_cycle_specific_coefficients.csv", index=False
    )
    integration_effects.to_csv(
        TABLE_DIR / "table12b_integration_cycle_specific_marginal_effects.csv", index=False
    )

    characteristics = run_weighted_characteristics(data)
    characteristics.to_csv(TABLE_DIR / "table13_weighted_sample_characteristics.csv", index=False)

    ml_metrics, ml_comparison, calibration, cross_validation = run_temporal_validation_extended(data)
    ml_metrics.to_csv(TABLE_DIR / "table14_temporal_validation_uncertainty_calibration.csv", index=False)
    ml_comparison.to_csv(TABLE_DIR / "table14b_paired_algorithm_comparisons.csv", index=False)
    cross_validation.to_csv(TABLE_DIR / "table15_hints6_five_fold_cross_validation.csv", index=False)
    calibration.to_csv(TABLE_DIR / "table16_calibration_curve_data.csv", index=False)

    create_extended_figures(cycle_effects, heterogeneity, calibration)

    qa = {
        "total_rows": int(len(data)),
        "rows_by_cycle": {
            str(cycle): int(data["cycle"].eq(cycle).sum()) for cycle in [2022, 2024]
        },
        "unique_cycle_hhid": bool(~data.duplicated(["cycle", "HHID"]).any()),
        "all_replicate_fits_successful": bool(
            all(item["failed"] == 0 for item in all_diagnostics)
        ),
        "diagnostics": all_diagnostics,
        "finite_checks": {
            "full_coefficients": bool(np.isfinite(full_coefficients["odds_ratio"]).all()),
            "cycle_effects": bool(np.isfinite(cycle_effects["estimate"]).all()),
            "sensitivity": bool(np.isfinite(sensitivity["estimate"]).all()),
            "heterogeneity": bool(np.isfinite(heterogeneity["estimate"]).all()),
            "integration_effects": bool(np.isfinite(integration_effects["estimate"]).all()),
            "ml_auc": bool(np.isfinite(ml_metrics["weighted_roc_auc"]).all()),
        },
        "created_tables": [
            f"table{number}" for number in range(8, 17)
        ] + [
            "table14b"
        ],
        "created_figures": [
            "figure5_cycle_specific_marginal_effects.png",
            "figure6_hcp_heterogeneity.png",
            "figure7_temporal_calibration.png",
        ],
    }
    (PROJECT / "outputs" / "extended_run_summary.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
