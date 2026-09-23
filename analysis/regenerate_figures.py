"""Regenerate all publication figures from saved result tables.

This script is intentionally model-free: it reads the validated CSV outputs and
rebuilds Figures 1--7 using the publication style defined in the analysis code.
"""

from __future__ import annotations

import pandas as pd

import run_experiment as base
import run_extended_experiments as extended


def main() -> None:
    table_dir = base.TABLE_DIR

    base.create_figures(
        pd.read_csv(table_dir / "table1_weighted_prevalence.csv"),
        pd.read_csv(table_dir / "table3_average_marginal_effects.csv"),
        pd.read_csv(table_dir / "table6_shap_importance.csv"),
        pd.read_csv(table_dir / "table7_ml_scenario_probabilities.csv"),
    )

    extended.create_extended_figures(
        pd.read_csv(table_dir / "table9b_cycle_specific_marginal_effects.csv"),
        pd.read_csv(table_dir / "table11_heterogeneity_hcp_encouragement.csv"),
        pd.read_csv(table_dir / "table16_calibration_curve_data.csv"),
    )

    print(f"Regenerated Figures 1--7 in {base.FIGURE_DIR}")


if __name__ == "__main__":
    main()
