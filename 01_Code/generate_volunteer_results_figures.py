from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import kruskal, mannwhitneyu, fisher_exact


CONDITION_ORDER = [
    "payoff_only",
    "reputation_only",
    "belonging_only",
    "full_structured_model",
    "full_llm_mediated_model",
]

CONDITION_LABELS = {
    "payoff_only": "Payoff only",
    "reputation_only": "Reputation only",
    "belonging_only": "Belonging only",
    "full_structured_model": "Full structured",
    "full_llm_mediated_model": "Full LLM-mediated",
}

COLORS = {
    "payoff_only": "#7F8C8D",
    "reputation_only": "#8E6C8A",
    "belonging_only": "#D28E2B",
    "full_structured_model": "#2A9D8F",
    "full_llm_mediated_model": "#C44E52",
}

ACTION_ORDER = ["help", "repair", "delay", "refuse", "withdraw"]
ACTION_LABELS = {
    "help": "Help",
    "repair": "Repair",
    "delay": "Delay",
    "refuse": "Refuse",
    "withdraw": "Withdraw",
}
ACTION_COLORS = {
    "help": "#2A9D8F",
    "repair": "#457B9D",
    "delay": "#E9C46A",
    "refuse": "#E76F51",
    "withdraw": "#6D597A",
}

def resolve_data_directory(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    input_path = input_path.expanduser().resolve()

    if input_path.is_dir():
        candidates = [input_path]
        candidates.extend(p for p in input_path.iterdir() if p.is_dir())
        for candidate in candidates:
            if (candidate / "volunteer_run_summary.csv").exists():
                return candidate, None
        raise FileNotFoundError(
            f"No volunteer_run_summary.csv was found inside {input_path}"
        )

    if input_path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="volunteer_results_")
        with zipfile.ZipFile(input_path) as archive:
            archive.extractall(tmp.name)
        root = Path(tmp.name)
        candidates = [root]
        candidates.extend(p for p in root.rglob("*") if p.is_dir())
        for candidate in candidates:
            if (candidate / "volunteer_run_summary.csv").exists():
                return candidate, tmp
        tmp.cleanup()
        raise FileNotFoundError(
            f"The ZIP {input_path} does not contain volunteer_run_summary.csv"
        )

    raise ValueError("--input must be a results directory or a .zip file")


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run = pd.read_csv(data_dir / "volunteer_run_summary.csv")
    weekly = pd.read_csv(data_dir / "volunteer_weekly_metrics.csv")
    log = pd.read_csv(data_dir / "volunteer_agent_log.csv", low_memory=False)

    required_run = {
        "run", "condition", "final_mean_belonging", "mean_helping_rate",
        "mean_reciprocal_support_rate", "mean_volunteer_retention",
        "mean_withdrawal_rate", "mean_collective_avoidance_rate",
        "community_breakdown",
    }
    required_weekly = {
        "run", "step", "condition", "mean_belonging", "helping_rate",
        "volunteer_retention",
    }
    required_log = {"condition", "action", "use_llm", "extracted_action"}

    for name, frame, required in [
        ("run summary", run, required_run),
        ("weekly metrics", weekly, required_weekly),
        ("agent log", log, required_log),
    ]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {name}: {sorted(missing)}")

    return run, weekly, log

def clean_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def condition_positions(conditions: list[str]) -> np.ndarray:
    return np.arange(len(conditions), dtype=float)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: positive values indicate x tends to exceed y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    comparisons = np.sign(x[:, None] - y[None, :])
    return float(comparisons.mean())


def plot_belonging_trajectory(weekly: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        weekly.groupby(["condition", "step"], as_index=False)["mean_belonging"]
        .agg(mean="mean", sem="sem")
    )
    summary["ci95"] = 1.96 * summary["sem"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    for condition in CONDITION_ORDER:
        data = summary[summary["condition"] == condition].sort_values("step")
        if data.empty:
            continue
        x = data["step"].to_numpy()
        y = data["mean"].to_numpy()
        ci = data["ci95"].to_numpy()
        ax.plot(
            x, y, linewidth=2.2, color=COLORS[condition],
            label=CONDITION_LABELS[condition]
        )
        ax.fill_between(x, y - ci, y + ci, color=COLORS[condition], alpha=0.14)

    ax.axhline(0.5, linestyle="--", linewidth=1.0, color="black", alpha=0.45)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Mean belonging")
    ax.set_ylim(-0.02, 1.03)
    ax.legend(frameon=False, ncol=2, loc="best")
    clean_axes(ax)
    save_figure(fig, output_dir / "fig1_belonging_trajectory.png")


def plot_helping_reciprocity_dumbbell(run: pd.DataFrame, output_dir: Path) -> None:
    means = (
        run.groupby("condition")[["mean_helping_rate", "mean_reciprocal_support_rate"]]
        .mean()
        .reindex(CONDITION_ORDER)
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    y = condition_positions(CONDITION_ORDER)

    for idx, condition in enumerate(CONDITION_ORDER):
        helping = means.loc[condition, "mean_helping_rate"]
        reciprocal = means.loc[condition, "mean_reciprocal_support_rate"]
        ax.plot([helping, reciprocal], [idx, idx], color="#B0B0B0", linewidth=2.0, zorder=1)
        ax.scatter(helping, idx, s=85, marker="o", color=COLORS[condition], zorder=3)
        ax.scatter(reciprocal, idx, s=95, marker="D", facecolor="white",
                   edgecolor=COLORS[condition], linewidth=2.0, zorder=3)
        ax.text(helping - 0.012, idx + 0.18, f"{helping:.3f}", ha="right", va="center", fontsize=9)
        ax.text(reciprocal + 0.012, idx + 0.18, f"{reciprocal:.3f}", ha="left", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_xlabel("Mean rate across runs")
    ax.set_xlim(0.10, 0.75)
    ax.invert_yaxis()
    legend = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#555555",
               markeredgecolor="#555555", markersize=8, label="Helping rate"),
        Line2D([0], [0], marker="D", linestyle="None", markerfacecolor="white",
               markeredgecolor="#555555", markeredgewidth=1.8, markersize=8,
               label="Reciprocal support rate"),
    ]
    ax.legend(handles=legend, frameon=False, loc="lower right")
    clean_axes(ax, grid_axis="x")
    save_figure(fig, output_dir / "fig2_helping_reciprocity_dumbbell.png")
def plot_retention_violin(run: pd.DataFrame, output_dir: Path) -> None:
    rng = np.random.default_rng(20260713)
    data = [
        run.loc[run["condition"] == condition, "mean_volunteer_retention"].dropna().to_numpy()
        for condition in CONDITION_ORDER
    ]
    positions = np.arange(1, len(CONDITION_ORDER) + 1)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    violin = ax.violinplot(
        data, positions=positions, widths=0.72,
        showmeans=False, showmedians=False, showextrema=False
    )
    for body, condition in zip(violin["bodies"], CONDITION_ORDER):
        body.set_facecolor(COLORS[condition])
        body.set_edgecolor(COLORS[condition])
        body.set_alpha(0.22)

    box = ax.boxplot(
        data, positions=positions, widths=0.22, patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.7},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    for patch, condition in zip(box["boxes"], CONDITION_ORDER):
        patch.set_facecolor("white")
        patch.set_edgecolor(COLORS[condition])
        patch.set_linewidth(1.6)

    for pos, values, condition in zip(positions, data, CONDITION_ORDER):
        jitter = rng.normal(0.0, 0.055, size=len(values))
        ax.scatter(
            np.full_like(values, pos, dtype=float) + jitter,
            values,
            s=23,
            color=COLORS[condition],
            alpha=0.70,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], rotation=17, ha="right")
    ax.set_ylabel("Mean proportion of active agents")
    ax.set_ylim(-0.03, 1.04)
    clean_axes(ax)
    save_figure(fig, output_dir / "fig3_retention_violin.png")
def plot_action_composition(log: pd.DataFrame, output_dir: Path) -> None:
    composition = (
        pd.crosstab(log["condition"], log["action"], normalize="index")
        .reindex(index=CONDITION_ORDER, columns=ACTION_ORDER, fill_value=0.0)
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    y = np.arange(len(CONDITION_ORDER))
    left = np.zeros(len(CONDITION_ORDER))

    for action in ACTION_ORDER:
        values = composition[action].to_numpy()
        ax.barh(
            y, values, left=left, height=0.63,
            label=ACTION_LABELS[action], color=ACTION_COLORS[action],
            edgecolor="white", linewidth=0.7
        )
        for idx, value in enumerate(values):
            if value >= 0.075:
                ax.text(
                    left[idx] + value / 2, idx, f"{100 * value:.1f}%",
                    ha="center", va="center", fontsize=8.4,
                    color="white" if action in {"help", "refuse", "withdraw"} else "black",
                    fontweight="bold"
                )
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_xlabel("Share of recorded actions")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{int(x * 100)}%" for x in np.linspace(0, 1, 6)])
    ax.invert_yaxis()
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    clean_axes(ax, grid_axis="x")
    save_figure(fig, output_dir / "fig4_action_composition.png")


def plot_belonging_retention_scatter(run: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 6.0))

    for condition in CONDITION_ORDER:
        data = run[run["condition"] == condition]
        normal = data[data["community_breakdown"] == 0]
        broken = data[data["community_breakdown"] == 1]

        ax.scatter(
            normal["final_mean_belonging"], normal["mean_volunteer_retention"],
            s=44, color=COLORS[condition], alpha=0.58,
            edgecolor="white", linewidth=0.45
        )
        ax.scatter(
            broken["final_mean_belonging"], broken["mean_volunteer_retention"],
            s=58, marker="X", color=COLORS[condition], alpha=0.80,
            edgecolor="white", linewidth=0.45
        )

        x_mean = data["final_mean_belonging"].mean()
        y_mean = data["mean_volunteer_retention"].mean()
        ax.scatter(
            [x_mean], [y_mean], s=165, marker="o",
            facecolor="white", edgecolor=COLORS[condition], linewidth=2.4,
            zorder=5
        )
        label_dx = 0.012
        label_dy = 0.016
        if condition == "payoff_only":
            label_dy = -0.045
        elif condition == "reputation_only":
            label_dx = -0.21
            label_dy = -0.038
        ax.text(
            x_mean + label_dx, y_mean + label_dy,
            CONDITION_LABELS[condition], fontsize=9.2,
            color=COLORS[condition], fontweight="bold"
        )

    ax.set_xlabel("Final mean belonging")
    ax.set_ylabel("Mean volunteer retention")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.43, 1.035)
    marker_legend = [
        Line2D([0], [0], marker="o", linestyle="None", color="#555555",
               markerfacecolor="#555555", markersize=7, label="Run without breakdown"),
        Line2D([0], [0], marker="X", linestyle="None", color="#555555",
               markerfacecolor="#555555", markersize=8, label="Run with breakdown"),
        Line2D([0], [0], marker="o", linestyle="None", color="#555555",
               markerfacecolor="white", markeredgewidth=2, markersize=10,
               label="Condition mean"),
    ]
    ax.legend(handles=marker_legend, frameon=False, loc="lower right")
    clean_axes(ax)
    save_figure(fig, output_dir / "fig5_belonging_retention_scatter.png")


def plot_llm_extraction_accuracy(log: pd.DataFrame, output_dir: Path) -> None:
    llm = log[log["use_llm"] == 1].copy()
    llm = llm.dropna(subset=["action", "extracted_action"])
    if llm.empty:
        return

    accuracy = (
        llm.assign(correct=llm["action"].eq(llm["extracted_action"]))
        .groupby("action")["correct"]
        .agg(["mean", "count"])
        .reindex(ACTION_ORDER)
        .dropna()
    )
    overall = llm["action"].eq(llm["extracted_action"]).mean()

    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    y = np.arange(len(accuracy))
    values = accuracy["mean"].to_numpy()

    ax.hlines(y, 0, values, color="#B7B7B7", linewidth=2.1)
    for idx, (action, row) in enumerate(accuracy.iterrows()):
        ax.scatter(row["mean"], idx, s=105, color=ACTION_COLORS[action], zorder=3)
        ax.text(
            min(row["mean"] + 0.018, 1.015), idx,
            f"{100 * row['mean']:.1f}%  (n={int(row['count']):,})",
            va="center", fontsize=9.3
        )

    ax.axvline(overall, linestyle="--", linewidth=1.5, color="black", alpha=0.60)
    ax.text(
        overall - 0.012, 0.97,
        f"Overall exact match: {100 * overall:.1f}%",
        transform=ax.get_xaxis_transform(),
        ha="right", va="top", fontsize=9.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0}
    )
    ax.set_yticks(y)
    ax.set_yticklabels([ACTION_LABELS[a] for a in accuracy.index])
    ax.set_xlabel("Exact action-extraction match rate")
    ax.set_xlim(0, 1.08)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    ax.invert_yaxis()
    clean_axes(ax, grid_axis="x")
    save_figure(fig, output_dir / "fig6_llm_extraction_accuracy.png")


def create_descriptive_table(run: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    metrics = [
        "final_mean_belonging",
        "mean_helping_rate",
        "mean_reciprocal_support_rate",
        "mean_volunteer_retention",
        "mean_withdrawal_rate",
        "mean_collective_avoidance_rate",
        "community_breakdown",
    ]
    table = run.groupby("condition")[metrics].agg(["mean", "std", "median"])
    table = table.reindex(CONDITION_ORDER)
    table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
    table = table.reset_index()
    table.insert(1, "condition_label", table["condition"].map(CONDITION_LABELS))
    table.to_csv(output_dir / "descriptive_results_by_condition.csv", index=False)
    return table


def create_statistical_tests(run: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    continuous_metrics = [
        "final_mean_belonging",
        "mean_helping_rate",
        "mean_reciprocal_support_rate",
        "mean_volunteer_retention",
        "mean_withdrawal_rate",
        "mean_collective_avoidance_rate",
    ]

    omnibus_rows = []
    pairwise_rows = []
    reference = "full_structured_model"

    for metric in continuous_metrics:
        groups = [
            run.loc[run["condition"] == condition, metric].dropna().to_numpy()
            for condition in CONDITION_ORDER
        ]
        statistic, p_value = kruskal(*groups)
        omnibus_rows.append({
            "metric": metric,
            "test": "Kruskal-Wallis",
            "statistic": statistic,
            "p_value": p_value,
        })

        raw_p = []
        metric_rows = []
        x = run.loc[run["condition"] == reference, metric].dropna().to_numpy()
        for comparison in [c for c in CONDITION_ORDER if c != reference]:
            y = run.loc[run["condition"] == comparison, metric].dropna().to_numpy()
            u_stat, p = mannwhitneyu(x, y, alternative="two-sided")
            row = {
                "metric": metric,
                "reference": reference,
                "comparison": comparison,
                "test": "Mann-Whitney U",
                "statistic": u_stat,
                "p_value_raw": p,
                "cliffs_delta_reference_minus_comparison": cliffs_delta(x, y),
                "reference_mean": float(np.mean(x)),
                "comparison_mean": float(np.mean(y)),
            }
            metric_rows.append(row)
            raw_p.append(p)

        adjusted = holm_adjust(raw_p)
        for row, p_adj in zip(metric_rows, adjusted):
            row["p_value_holm"] = p_adj
            pairwise_rows.append(row)

    raw_p = []
    breakdown_rows = []
    reference_values = run.loc[run["condition"] == reference, "community_breakdown"]
    ref_break = int(reference_values.sum())
    ref_ok = int(len(reference_values) - ref_break)

    for comparison in [c for c in CONDITION_ORDER if c != reference]:
        values = run.loc[run["condition"] == comparison, "community_breakdown"]
        comp_break = int(values.sum())
        comp_ok = int(len(values) - comp_break)
        odds_ratio, p = fisher_exact(
            [[ref_break, ref_ok], [comp_break, comp_ok]], alternative="two-sided"
        )
        breakdown_rows.append({
            "metric": "community_breakdown",
            "reference": reference,
            "comparison": comparison,
            "test": "Fisher exact",
            "odds_ratio": odds_ratio,
            "p_value_raw": p,
            "reference_probability": float(reference_values.mean()),
            "comparison_probability": float(values.mean()),
        })
        raw_p.append(p)

    adjusted = holm_adjust(raw_p)
    for row, p_adj in zip(breakdown_rows, adjusted):
        row["p_value_holm"] = p_adj

    omnibus = pd.DataFrame(omnibus_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    breakdown = pd.DataFrame(breakdown_rows)

    omnibus.to_csv(output_dir / "omnibus_tests.csv", index=False)
    pairwise.to_csv(output_dir / "planned_pairwise_tests_vs_full_structured.csv", index=False)
    breakdown.to_csv(output_dir / "breakdown_fisher_tests.csv", index=False)
    return omnibus, pairwise


def write_plain_text_summary(
    run: pd.DataFrame,
    weekly: pd.DataFrame,
    log: pd.DataFrame,
    output_dir: Path,
) -> None:
    mean_table = run.groupby("condition").mean(numeric_only=True).reindex(CONDITION_ORDER)
    endpoint = (
        weekly.sort_values("step")
        .groupby(["condition", "run"], as_index=False)
        .tail(1)
        .groupby("condition")
        .mean(numeric_only=True)
        .reindex(CONDITION_ORDER)
    )
    llm = log[log["use_llm"] == 1].dropna(subset=["action", "extracted_action"])
    exact_match = llm["action"].eq(llm["extracted_action"]).mean() if not llm.empty else np.nan

    lines = [
        "VOLUNTEER COMMUNITY EXPERIMENT: DESCRIPTIVE SUMMARY",
        "=" * 60,
        "",
    ]
    for condition in CONDITION_ORDER:
        row = mean_table.loc[condition]
        end = endpoint.loc[condition]
        lines.extend([
            CONDITION_LABELS[condition],
            f"  Final mean belonging: {row['final_mean_belonging']:.4f}",
            f"  Mean helping rate: {row['mean_helping_rate']:.4f}",
            f"  Mean reciprocal support rate: {row['mean_reciprocal_support_rate']:.4f}",
            f"  Mean retention across steps: {row['mean_volunteer_retention']:.4f}",
            f"  Final-step retention: {end['volunteer_retention']:.4f}",
            f"  Mean withdrawal rate: {row['mean_withdrawal_rate']:.4f}",
            f"  Collective avoidance rate: {row['mean_collective_avoidance_rate']:.4f}",
            f"  Breakdown probability: {row['community_breakdown']:.4f}",
            "",
        ])
    lines.append(f"LLM exact action-extraction match rate: {exact_match:.4f}")
    (output_dir / "results_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# Main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate varied publication-ready figures for the volunteer experiment."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the experiment results directory or ZIP file.",
    )
    parser.add_argument(
        "--output-dir",
        default="volunteer_result_figures",
        help="Directory where figures and statistical tables will be written.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir, temporary_directory = resolve_data_directory(input_path)
    try:
        run, weekly, log = load_data(data_dir)

        plot_belonging_trajectory(weekly, output_dir)
        plot_helping_reciprocity_dumbbell(run, output_dir)
        plot_retention_violin(run, output_dir)
        plot_action_composition(log, output_dir)
        plot_belonging_retention_scatter(run, output_dir)
        plot_llm_extraction_accuracy(log, output_dir)

        create_descriptive_table(run, output_dir)
        create_statistical_tests(run, output_dir)
        write_plain_text_summary(run, weekly, log, output_dir)

        print(f"Results written to: {output_dir}")
        for file_path in sorted(output_dir.iterdir()):
            print(f"  - {file_path.name}")
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


if __name__ == "__main__":
    main()