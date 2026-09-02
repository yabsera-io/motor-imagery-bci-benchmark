"""Reproduce participant-level statistical tests for the BCI benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42
N_BOOTSTRAPS = 20_000


def main() -> None:
    benchmark = pd.read_csv(RESULTS_DIR / "participant_model_benchmark.csv")
    model_scores = benchmark.pivot(
        index="subject", columns="model", values="balanced_accuracy"
    )
    lda = model_scores["CSP + shrinkage LDA"]
    logistic = model_scores["CSP + logistic regression"]

    rng = np.random.default_rng(RANDOM_STATE)
    bootstrap_means = np.array(
        [
            rng.choice(lda.to_numpy(), size=len(lda), replace=True).mean()
            for _ in range(N_BOOTSTRAPS)
        ]
    )
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])

    channels = pd.read_csv(RESULTS_DIR / "channel_tradeoff_results.csv")
    channel_scores = channels.pivot(
        index="subject", columns="channel_set", values="balanced_accuracy"
    )

    calibration = pd.read_csv(RESULTS_DIR / "calibration_burden_results.csv")
    calibration_scores = (
        calibration.groupby(["subject", "training_fraction"])["balanced_accuracy"]
        .mean()
        .unstack()
    )

    statistics = {
        "participants": int(len(lda)),
        "lda_mean_balanced_accuracy": float(lda.mean()),
        "lda_bootstrap_95_ci": [float(ci_low), float(ci_high)],
        "lda_vs_chance": {
            "test": "one-sided Wilcoxon signed-rank",
            "chance_level": 0.5,
            "p_value": float(wilcoxon(lda - 0.5, alternative="greater").pvalue),
        },
        "lda_vs_logistic": {
            "test": "paired two-sided Wilcoxon signed-rank",
            "p_value": float(wilcoxon(lda, logistic).pvalue),
        },
        "channel_configuration": {
            "test": "Friedman repeated-measures test",
            "p_value": float(
                friedmanchisquare(
                    *[channel_scores[column] for column in channel_scores.columns]
                ).pvalue
            ),
        },
        "calibration_fraction": {
            "test": "Friedman repeated-measures test",
            "p_value": float(
                friedmanchisquare(
                    *[
                        calibration_scores[column]
                        for column in calibration_scores.columns
                    ]
                ).pvalue
            ),
        },
        "calibration_75_vs_100": {
            "test": "paired two-sided Wilcoxon signed-rank",
            "p_value": float(
                wilcoxon(calibration_scores[0.75], calibration_scores[1.0]).pvalue
            ),
        },
    }

    output_path = RESULTS_DIR / "statistical_tests.json"
    output_path.write_text(json.dumps(statistics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(statistics, indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
