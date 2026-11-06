import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PRIMARY_CONDITION_ORDER = [
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

    # Controls
    "belonging_only_matched_initial_help": (
        "Belonging only\n(matched initial help)"
    ),
    "belonging_only_initial_help_matched": (
        "Belonging only\n(matched initial help)"
    ),
    "belonging_only_renormalized": (
        "Belonging only\n(renormalized)"
    ),
    "belonging_only_uniform_partner": (
        "Belonging only\n(uniform partner)"
    ),

    "full_constant_participation": (
        "Full structured\n(constant participation)"
    ),
    "full_structured_constant_participation": (
        "Full structured\n(constant participation)"
    ),

    "full_random_partner_selection": (
        "Full structured\n(random partner)"
    ),
    "full_structured_random_partner": (
        "Full structured\n(random partner)"
    ),

    "full_minus_belonging": (
        "Full minus belonging"
    ),
    "full_minus_private_reputation": (
        "Full minus private reputation"
    ),
    "full_minus_collective_reputation": (
        "Full minus collective reputation"
    ),
}

ACTION_ORDER = [
    "help",
    "repair",
    "delay",
    "refuse",
    "withdraw",
]

CUE_ORDER = [
    "acceptance",
    "reciprocity",
    "consistency",
    "utility",
]


def first_existing_file(base_dir, candidates, required=True):
    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p

    if required:
        raise FileNotFoundError(
            f"No se encontró ninguno de estos archivos en {base_dir}:\n"
            f"{candidates}"
        )

    return None


def find_col(df, candidates, required=True):
    cols_lower = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    if required:
        raise KeyError(
            f"No se encontró ninguna columna entre: {candidates}\n"
            f"Columnas disponibles:\n{list(df.columns)}"
        )

    return None


def normalize_condition_name(x):
    if pd.isna(x):
        return x
    return str(x).strip()


def standardize_action(x):
    if pd.isna(x):
        return x

    x = str(x).strip().lower()

    aliases = {
        "reject": "refuse",
        "refusal": "refuse",
        "withdrawal": "withdraw",
        "postpone": "delay",
        "fix": "repair",
    }

    return aliases.get(x, x)


def safe_group_order(values, preferred, include_extras=False):
    present = list(
        pd.Series(values)
        .dropna()
        .unique()
    )

    ordered = [
        x
        for x in preferred
        if x in present
    ]

    if include_extras:
        ordered += [
            x
            for x in present
            if x not in ordered
        ]

    return ordered


def ensure_output_dir(path):
    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def ci95(series):
    series = pd.Series(series).dropna()

    if len(series) <= 1:
        return 0.0

    return (
        1.96
        * series.std(ddof=1)
        / np.sqrt(len(series))
    )


def save_figure(fig, path):
    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)






def load_inputs(input_dir):
    input_dir = Path(input_dir)

    files = {
        "weekly": first_existing_file(
            input_dir,
            [
                "volunteer_weekly_metrics_revision.csv",
                "volunteer_weekly_metrics.csv",
            ],
            required=False,
        ),

        "condition_summary": first_existing_file(
            input_dir,
            [
                "volunteer_condition_summary_revision.csv",
                "volunteer_condition_summary.csv",
            ],
            required=False,
        ),

        "run_summary": first_existing_file(
            input_dir,
            [
                "volunteer_run_summary_revision.csv",
                "volunteer_run_summary.csv",
            ],
            required=False,
        ),

        "agent_log": first_existing_file(
            input_dir,
            [
                "volunteer_agent_log_revision.csv",
                "volunteer_agent_log.csv",
            ],
            required=False,
        ),

        "llm_diag": first_existing_file(
            input_dir,
            [
                "llm_cue_diagnostics_by_action.csv",
            ],
            required=False,
        ),

        "llm_mech": first_existing_file(
            input_dir,
            [
                "llm_mechanistic_effect_by_action.csv",
                "llm_mechanistic_effects_by_action.csv",
                "llm_mechanistic_diagnostics.csv",
            ],
            required=False,
        ),
    }

    data = {}

    for key, path in files.items():
        if path is not None:
            data[key] = pd.read_csv(
                path,
                low_memory=False,
            )
        else:
            data[key] = None

    return data, files





def print_belonging_diagnostic(data):
    print("\n======================================")
    print("BELONGING DIAGNOSTIC")
    print("======================================")

    if data["condition_summary"] is None:
        return

    df = data["condition_summary"].copy()

    condition_col = find_col(
        df,
        [
            "condition",
            "condition_name",
        ],
        required=False,
    )

    belonging_col = find_col(
        df,
        [
            "final_mean_belonging_mean",
            "mean_final_belonging",
            "final_belonging_mean",
            "final_belonging",
        ],
        required=False,
    )

    if condition_col is None or belonging_col is None:
        return

    diagnostic = df[
        [
            condition_col,
            belonging_col,
        ]
    ].copy()

    diagnostic = diagnostic[
        diagnostic[
            condition_col
        ].isin(
            PRIMARY_CONDITION_ORDER
        )
    ]

    print(
        "\nFinal belonging from condition summary:"
    )

    print(
        diagnostic.to_string(
            index=False
        )
    )


# ============================================================
# FIGURE 1
# BELONGING TRAJECTORY
# ============================================================

def make_fig1_belonging_trajectory(
    data,
    outdir,
    include_controls=False,
):

    df = data["weekly"]

    if df is None:
        return {
            "created": False,
            "reason": "weekly metrics file missing",
        }

    cond_col = find_col(
        df,
        [
            "condition",
            "condition_name",
        ],
    )

    step_col = find_col(
        df,
        [
            "step",
            "week",
            "time_step",
            "t",
            "cycle",
        ],
    )

    mean_col = find_col(
        df,
        [
            "mean_belonging",
            "belonging_mean",
            "avg_belonging",
            "belonging",
        ],
    )

    run_col = find_col(
        df,
        [
            "run",
            "run_id",
            "simulation_run",
        ],
        required=False,
    )

    df = df.copy()

    df[cond_col] = (
        df[cond_col]
        .map(normalize_condition_name)
    )

    preferred = (
        list(CONDITION_LABELS.keys())
        if include_controls
        else PRIMARY_CONDITION_ORDER
    )

    conditions = safe_group_order(
        df[cond_col].unique(),
        preferred,
        include_extras=include_controls,
    )

    df = df[
        df[cond_col].isin(
            conditions
        )
    ].copy()

    if run_col is not None:

        agg = (
            df.groupby(
                [
                    cond_col,
                    step_col,
                ]
            )[mean_col]
            .agg(
                mean="mean",
                ci=ci95,
            )
            .reset_index()
        )

        agg["lower"] = (
            agg["mean"]
            - agg["ci"]
        )

        agg["upper"] = (
            agg["mean"]
            + agg["ci"]
        )

        agg = agg.rename(
            columns={
                cond_col: "condition",
                step_col: "step",
            }
        )

    else:

        agg = df[
            [
                cond_col,
                step_col,
                mean_col,
            ]
        ].copy()

        agg.columns = [
            "condition",
            "step",
            "mean",
        ]

        agg["lower"] = agg["mean"]
        agg["upper"] = agg["mean"]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for cond in conditions:

        sub = (
            agg[
                agg["condition"] == cond
            ]
            .sort_values("step")
        )

        if sub.empty:
            continue

        label = CONDITION_LABELS.get(
            cond,
            cond,
        )

        ax.plot(
            sub["step"],
            sub["mean"],
            label=label,
        )

        ax.fill_between(
            sub["step"],
            sub["lower"],
            sub["upper"],
            alpha=0.20,
        )

    ax.axhline(
        0.50,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Community activity cycle"
    )

    ax.set_ylabel(
        "Mean belonging"
    )

    ax.set_title(
        "Evolution of mean belonging "
        "across simulation steps"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.legend(
        fontsize=8,
        loc="best",
    )

    path = (
        outdir
        / "fig1_belonging_trajectory.png"
    )

    save_figure(
        fig,
        path,
    )

    return {
        "created": True,
        "file": path.name,
        "conditions": conditions,
    }







def make_fig2_helping_reciprocity(
    data,
    outdir,
    include_controls=False,
):

    df = data["condition_summary"]

    if df is None:
        return {
            "created": False,
            "reason": "condition summary file missing",
        }

    cond_col = find_col(
        df,
        [
            "condition",


            "condition_name",
        ],
    )

    help_col = find_col(
        df,
        [
            "mean_helping_rate_mean",
            "mean_helping_rate",
            "helping_rate_mean",
            "help_rate_mean",
            "helping_rate",
            "help_rate",
        ],
    )

    recip_col = find_col(
        df,
        [
            "mean_reciprocal_support_rate_mean",
            "mean_reciprocal_support_rate",
            "reciprocal_support_rate_mean",
            "reciprocity_rate_mean",
            "reciprocal_support_rate",
            "reciprocity_rate",
        ],
    )

    df = df.copy()

    df[cond_col] = (
        df[cond_col]
        .map(normalize_condition_name)
    )

    preferred = (
        list(CONDITION_LABELS.keys())
        if include_controls
        else PRIMARY_CONDITION_ORDER
    )

    conditions = safe_group_order(
        df[cond_col].unique(),
        preferred,
        include_extras=include_controls,
    )

    sub = df[
        df[cond_col].isin(
            conditions
        )
    ].copy()

    order_map = {
        c: i
        for i, c
        in enumerate(conditions)
    }

    sub["_order"] = (
        sub[cond_col]
        .map(order_map)
    )

    sub = sub.sort_values(
        "_order"
    )

    sub["label"] = (
        sub[cond_col]
        .map(
            lambda x:
            CONDITION_LABELS.get(
                x,
                x,
            )
        )
    )

    x = np.arange(
        len(sub)
    )

    width = 0.38

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        sub[help_col],
        width,
        label="Helping rate",
    )

    ax.bar(
        x + width / 2,
        sub[recip_col],
        width,
        label="Reciprocal support",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        sub["label"],
        rotation=20,
        ha="right",
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_xlabel(
        "Experimental condition"
    )

    ax.set_ylabel(
        "Mean rate"
    )

    ax.set_title(
        "Helping and reciprocal support "
        "by condition"
    )

    ax.legend()

    path = (
        outdir
        / "fig2_helping_reciprocity.png"
    )

    save_figure(
        fig,
        path,
    )

    return {
        "created": True,
        "file": path.name,
        "conditions": conditions,
    }



















def make_fig3_participation_violin(
    data,
    outdir,
    include_controls=False,
):

    df = data["run_summary"]

    if df is None:
        return {
            "created": False,
            "reason": "run summary file missing",
        }

    cond_col = find_col(
        df,
        [
            "condition",
            "condition_name",
        ],
    )

    part_col = find_col(
        df,
        [
            "mean_volunteer_retention",
            "participation_rate",
            "mean_participation_rate",
            "participation",
            "request_rate",
        ],
    )

    df = df.copy()

    df[cond_col] = (
        df[cond_col]
        .map(normalize_condition_name)
    )

    preferred = (
        list(CONDITION_LABELS.keys())
        if include_controls
        else PRIMARY_CONDITION_ORDER
    )

    conditions = safe_group_order(
        df[cond_col].unique(),
        preferred,
        include_extras=include_controls,
    )

    data_list = []
    labels = []

    for cond in conditions:

        vals = (
            df.loc[
                df[cond_col] == cond,
                part_col,
            ]
            .dropna()
            .values
        )

        if len(vals) == 0:
            continue

        data_list.append(
            vals
        )

        labels.append(
            CONDITION_LABELS.get(
                cond,
                cond,
            )
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.violinplot(
        data_list,
        showmeans=True,
        showmedians=False,
        showextrema=True,
    )

    ax.set_xticks(
        np.arange(
            1,
            len(labels) + 1,
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=20,
        ha="right",
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_xlabel(
        "Experimental condition"
    )

    ax.set_ylabel(
        "Volunteer retention rate"
    )

    ax.set_title(
        "Participation across simulation runs"
    )

    path = (
        outdir
        / "fig3_participation_violin.png"
    )

    save_figure(
        fig,
        path,
    )

    return {
        "created": True,
        "file": path.name,
        "conditions": conditions,
    }


# ============================================================
# FIGURE 4
# FINAL BELONGING VS PARTICIPATION
# ============================================================

def make_fig4_belonging_participation_scatter(
    data,
    outdir,
    include_controls=False,
):

    cond_df = data["condition_summary"]

    if cond_df is None:
        return {
            "created": False,
            "reason": "condition summary file missing",
        }

    cond_col = find_col(
        cond_df,
        [
            "condition",
            "condition_name",
        ],
    )

    belong_col = find_col(
        cond_df,
        [
            "final_mean_belonging_mean",
            "mean_final_belonging",
            "final_belonging_mean",
            "final_belonging",
            "mean_belonging_final",
        ],
    )

    part_col = find_col(
        cond_df,
        [
            "mean_volunteer_retention_mean",
            "mean_volunteer_retention",
            "mean_participation_rate",
            "participation_rate_mean",
            "participation_rate",
        ],
    )

    breakdown_col = find_col(
        cond_df,
        [
            "community_breakdown_mean",
            "community_breakdown_probability",
            "community_breakdown_rate",
            "breakdown_rate",
            "mean_breakdown_rate",
        ],
        required=False,
    )

    cond_df = cond_df.copy()

    cond_df[cond_col] = (
        cond_df[cond_col]
        .map(normalize_condition_name)
    )

    preferred = (
        list(CONDITION_LABELS.keys())
        if include_controls
        else PRIMARY_CONDITION_ORDER
    )

    conditions = safe_group_order(
        cond_df[cond_col].unique(),
        preferred,
        include_extras=include_controls,
    )

    cond_df = cond_df[
        cond_df[cond_col].isin(
            conditions
        )
    ].copy()

    cols = [
        cond_col,
        belong_col,
        part_col,
    ]

    if breakdown_col:
        cols.append(
            breakdown_col
        )

    plot_df = cond_df[
        cols
    ].copy()

    new_cols = [
        "condition",
        "final_belonging",
        "participation",
    ]

    if breakdown_col:
        new_cols.append(
            "breakdown_rate"
        )

    plot_df.columns = new_cols

    if (
        "breakdown_rate"
        not in plot_df.columns
    ):
        plot_df["breakdown_rate"] = 0.0

    order_map = {
        c: i
        for i, c
        in enumerate(conditions)
    }

    plot_df["_order"] = (
        plot_df["condition"]
        .map(order_map)
    )

    plot_df = plot_df.sort_values(
        "_order"
    )

    label_offsets = {
        "payoff_only": (
            -0.11,
            0.012,
        ),
        "reputation_only": (
            0.012,
            0.012,
        ),
        "belonging_only": (
            0.012,
            0.012,
        ),
        "full_structured_model": (
            0.012,
            0.012,
        ),
        "full_llm_mediated_model": (
            0.012,
            0.012,
        ),
    }

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for _, row in plot_df.iterrows():

        breakdown_rate = (
            row["breakdown_rate"]
            if not pd.isna(
                row["breakdown_rate"]
            )
            else 0.0
        )

        size = (
            180
            + 650
            * float(
                breakdown_rate
            )
        )

        ax.scatter(
            row["final_belonging"],
            row["participation"],
            s=size,
            alpha=0.75,
        )

        dx, dy = label_offsets.get(
            row["condition"],
            (
                0.012,
                0.012,
            ),
        )

        ax.text(
            row["final_belonging"] + dx,
            row["participation"] + dy,
            CONDITION_LABELS.get(
                row["condition"],
                row["condition"],
            ),
            fontsize=8,
        )

    ax.set_xlim(
        0,
        1.05,
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_xlabel(
        "Mean final belonging"
    )

    ax.set_ylabel(
        "Volunteer retention rate"
    )

    ax.set_title(
        "Final belonging and participation "
        "by condition"
    )

    ax.grid(
        True,
        alpha=0.30,
    )

    path = (
        outdir
        / "fig4_belonging_participation_scatter.png"
    )

    save_figure(
        fig,
        path,
    )

    return {
        "created": True,
        "file": path.name,
        "conditions": conditions,
    }

def make_fig5_action_composition(
    data,
    outdir,
    include_controls=False,
):

    df = data["agent_log"]

    if df is None:
        return {
            "created": False,
            "reason": "agent log file missing",
        }

    cond_col = find_col(
        df,
        [
            "condition",
            "condition_name",
        ],
    )

    action_col = find_col(
        df,
        [
            "action",
            "selected_action",
            "formal_action",
            "response_action",
            "helper_action",
        ],
    )

    df = df.copy()

    df[cond_col] = (
        df[cond_col]
        .map(normalize_condition_name)
    )

    df[action_col] = (
        df[action_col]
        .map(standardize_action)
    )

    preferred = (
        list(CONDITION_LABELS.keys())
        if include_controls
        else PRIMARY_CONDITION_ORDER
    )

    conditions = safe_group_order(
        df[cond_col].unique(),
        preferred,
        include_extras=include_controls,
    )

    df = df[
        df[cond_col].isin(
            conditions
        )
    ].copy()

    comp = (
        df
        .groupby(
            [
                cond_col,
                action_col,
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    for action in ACTION_ORDER:

        if action not in comp.columns:
            comp[action] = 0

    comp = comp[
        ACTION_ORDER
    ]

    comp = (
        comp
        .div(
            comp.sum(axis=1),
            axis=0,
        )
        .fillna(0)
    )

    ordered_index = [
        c
        for c in conditions
        if c in comp.index
    ]

    comp = comp.loc[
        ordered_index
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = np.arange(
        len(comp)
    )

    bottom = np.zeros(
        len(comp)
    )

    for action in ACTION_ORDER:

        vals = comp[
            action
        ].values

        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=action.capitalize(),
        )

        bottom += vals

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            CONDITION_LABELS.get(
                c,
                c,
            )
            for c
            in comp.index
        ],
        rotation=20,
        ha="right",
    )

    ax.set_ylim(
        0,
        1.0,
    )

    ax.set_ylabel(
        "Proportion of actions"
    )

    ax.set_title(
        "Action composition by "
        "experimental condition"
    )

    ax.legend(
        title="Action",
        bbox_to_anchor=(
            1.02,
            1,
        ),
        loc="upper left",
    )

    path = (
        outdir
        / "fig5_action_composition.png"
    )

    save_figure(
        fig,
        path,
    )

    return {
        "created": True,
        "file": path.name,
        "conditions": conditions,
    }


























def make_fig6_llm_exact_match(
    data,
    outdir,
):

    df = data["agent_log"]

    if df is None:
        return {
            "created": False,
            "reason": "agent log file missing",
        }

    condition_col = find_col(
        df,
        [
            "condition",
            "condition_name",
        ],
    )

    action_col = find_col(
        df,
        [
            "action",
            "formal_action",
            "intended_action",
        ],
    )

    extracted_col = find_col(
        df,
        [
            "extracted_action",
            "recovered_action",
            "llm_action",
        ],
    )

    llm_df = df[
        df[condition_col]
        == "full_llm_mediated_model"
    ].copy()

    llm_df[action_col] = (
        llm_df[action_col]
        .map(standardize_action)
    )

    llm_df[extracted_col] = (
        llm_df[extracted_col]
        .map(standardize_action)
    )

    llm_df = llm_df[
        llm_df[action_col].notna()
        &
        llm_df[extracted_col].notna()
    ].copy()

    if llm_df.empty:
        return {
            "created": False,
            "reason": "no valid LLM interactions found",
        }

    llm_df["exact_match"] = (
        llm_df[action_col]
        == llm_df[extracted_col]
    ).astype(float)

    by_action = (
        llm_df
        .groupby(
            action_col
        )["exact_match"]
        .mean()
        .reindex(
            ACTION_ORDER
        )
    )

    overall = (
        llm_df["exact_match"]
        .mean()
    )

    labels = (
        ["Overall"]
        +
        [
            a.capitalize()
            for a
            in ACTION_ORDER
        ]
    )

    values = (
        [overall]
        +
        [
            by_action.get(
                a,
                np.nan,
            )
            for a
            in ACTION_ORDER
        ]
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    x = np.arange(
        len(labels)
    )

    ax.bar(
        x,
        values,
    )

    ax.axhline(
        overall,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        "Exact-match rate"
    )

    ax.set_title(
        "LLM action recovery consistency"
    )

    for i, v in enumerate(values):

        if pd.isna(v):
            continue

        ax.text(
            i,
            min(
                v + 0.025,
                1.02,
            ),
            f"{100 * v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    path = (
        outdir
        / "fig6_llm_exact_match.png"
    )

    save_figure(
        fig,
        path,
    )

    print(
        "\n=== LLM EXACT MATCH ==="
    )

    print(
        f"Overall: {overall:.6f}"
    )

    for action in ACTION_ORDER:

        value = by_action.get(
            action,
            np.nan,
        )

        print(
            f"{action}: "
            f"{value:.6f}"
        )

    return {
        "created": True,
        "file": path.name,
        "overall": float(
            overall
        ),
        "by_action": {
            a: (
                None
                if pd.isna(
                    by_action.get(
                        a,
                        np.nan,
                    )
                )
                else float(
                    by_action[
                        a
                    ]
                )
            )
            for a
            in ACTION_ORDER
        },
    }







def make_fig7_llm_cue_distributions(
    data,
    outdir,
):

    df = data["agent_log"]

    if df is None:
        return {
            "created": False,
            "reason": "agent log file missing",
        }

    condition_col = find_col(
        df,
        [
            "condition",
            "condition_name",
        ],
    )

    action_col = find_col(
        df,
        [
            "action",
            "formal_action",
            "intended_action",
        ],
    )

    cue_columns = {
        "acceptance": {
            "llm": find_col(
                df,
                [
                    "acceptance",
                    "llm_acceptance",
                    "extracted_acceptance",
                ],
            ),
            "structured": find_col(
                df,
                [
                    "structured_acceptance",
                    "target_acceptance",
                ],
            ),
        },

        "reciprocity": {
            "llm": find_col(
                df,
                [
                    "reciprocity",
                    "llm_reciprocity",
                    "extracted_reciprocity",
                ],
            ),
            "structured": find_col(
                df,
                [
                    "structured_reciprocity",
                    "target_reciprocity",
                ],
            ),
        },

        "consistency": {
            "llm": find_col(
                df,
                [
                    "consistency",
                    "llm_consistency",
                    "extracted_consistency",
                ],
            ),
            "structured": find_col(


                df,
                [
                    "structured_consistency",
                    "target_consistency",
                ],
            ),
        },

        "utility": {
            "llm": find_col(
                df,
                [
                    "utility",
                    "llm_utility",
                    "extracted_utility",
                ],
            ),
            "structured": find_col(
                df,
                [
                    "structured_utility",
                    "target_utility",
                ],
            ),
        },
    }

    llm_df = df[
        df[condition_col]
        == "full_llm_mediated_model"
    ].copy()

    llm_df[action_col] = (
        llm_df[action_col]
        .map(standardize_action)
    )

    llm_df = llm_df[
        llm_df[action_col].isin(
            ACTION_ORDER
        )
    ].copy()

    if llm_df.empty:
        return {
            "created": False,
            "reason": "no LLM-mediated interactions found",
        }

    # --------------------------------------------------------
    # Build paired long-form dataset
    # --------------------------------------------------------

    long_rows = []

    for _, row in llm_df.iterrows():

        action = row[
            action_col
        ]

        for cue in CUE_ORDER:

            llm_col = cue_columns[
                cue
            ]["llm"]

            structured_col = cue_columns[
                cue
            ]["structured"]

            llm_value = row[
                llm_col
            ]

            structured_value = row[
                structured_col
            ]

            if not pd.isna(
                structured_value
            ):

                long_rows.append(
                    {
                        "action": action,
                        "cue": cue,
                        "source": "Structured",
                        "value": float(
                            structured_value
                        ),
                    }
                )

            if not pd.isna(
                llm_value
            ):

                long_rows.append(
                    {
                        "action": action,
                        "cue": cue,
                        "source": "LLM-extracted",
                        "value": float(
                            llm_value
                        ),
                    }
                )

    long_df = pd.DataFrame(
        long_rows
    )

    if long_df.empty:
        return {
            "created": False,
            "reason": "no valid cue observations found",
        }
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    axes = axes.flatten()

    for ax, cue in zip(
        axes,
        CUE_ORDER,
    ):

        cue_df = long_df[
            long_df["cue"] == cue
        ]

        box_data = []
        positions = []
        labels = []

        pos = 1.0

        for action in ACTION_ORDER:

            sub = cue_df[
                cue_df["action"] == action
            ]

            structured_vals = (
                sub[
                    sub["source"]
                    == "Structured"
                ]["value"]
                .dropna()
                .values
            )

            llm_vals = (
                sub[
                    sub["source"]
                    == "LLM-extracted"
                ]["value"]
                .dropna()
                .values
            )

            if len(
                structured_vals
            ) > 0:

                box_data.append(
                    structured_vals
                )

                positions.append(
                    pos
                )

                labels.append(
                    f"{action.capitalize()}\nS"
                )

            if len(
                llm_vals
            ) > 0:

                box_data.append(
                    llm_vals
                )

                positions.append(
                    pos + 0.65
                )

                labels.append(
                    f"{action.capitalize()}\nL"
                )

            pos += 2.0

        if box_data:

            ax.boxplot(
                box_data,
                positions=positions,
                widths=0.50,
                showfliers=False,
            )

        ax.set_xticks(
            positions
        )

        ax.set_xticklabels(
            labels,
            fontsize=8,
        )

        ax.set_ylim(
            -0.02,
            1.02,
        )

        ax.set_title(
            cue.capitalize()
        )

        ax.set_ylabel(
            "Cue value"
        )

        ax.grid(
            axis="y",
            alpha=0.20,
        )

    fig.suptitle(
        "Distributions of structured and "
        "LLM-extracted social cues by intended action",
        y=1.01,
    )

    fig.text(
        0.5,
        0.01,
        "S = structured counterpart; "
        "L = LLM-extracted value",
        ha="center",
        fontsize=9,
    )

    path = (
        outdir
        / "fig7_llm_cue_distributions.png"
    )

    save_figure(
        fig,
        path,
    )

    # --------------------------------------------------------
    # Distribution summary
    # --------------------------------------------------------

    summary = (
        long_df
        .groupby(
            [
                "action",
                "cue",
                "source",
            ]
        )["value"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    summary_path = (
        outdir
        / "fig7_llm_cue_distribution_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\n=== LLM CUE DISTRIBUTIONS ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return {
        "created": True,
        "file": path.name,
        "summary_file": summary_path.name,
        "n_interactions": int(
            len(
                llm_df
            )
        ),
        "n_long_rows": int(
            len(
                long_df
            )
        ),
    }


def make_fig8_llm_deltaE_by_action(
    data,
    outdir,
):

    candidates = []

    if data["llm_mech"] is not None:

        candidates.append(
            (
                "llm_mech",
                data[
                    "llm_mech"
                ].copy(),
            )
        )

    if data["agent_log"] is not None:

        candidates.append(
            (
                "agent_log",
                data[
                    "agent_log"
                ].copy(),
            )
        )

    for source_name, df in candidates:

        action_col = find_col(
            df,
            [
                "action",
                "formal_action",
                "structured_action",
                "action_type",
                "intended_action",
            ],
            required=False,
        )

        delta_col = find_col(
            df,
            [
                "delta_E_llm_vs_structured",
                "delta_e_llm_vs_structured",
                "delta_E",
                "delta_e",
                "mechanistic_delta_E",
                "interaction_value_difference",
            ],
            required=False,
        )

        if (
            action_col is None
            or delta_col is None
        ):
            continue

        if source_name == "agent_log":

            condition_col = find_col(
                df,
                [
                    "condition",
                    "condition_name",
                ],
                required=False,
            )

            if condition_col is not None:

                df = df[
                    df[condition_col]
                    == "full_llm_mediated_model"
                ].copy()

        df[action_col] = (
            df[action_col]
            .map(standardize_action)
        )

        valid_actions = []
        data_list = []
        mean_values = {}

        for action in ACTION_ORDER:

            vals = (
                df.loc[
                    df[action_col]
                    == action,
                    delta_col,
                ]
                .dropna()
                .astype(float)
                .values
            )

            if len(vals) == 0:
                continue

            valid_actions.append(
                action
            )

            data_list.append(
                vals
            )

            mean_values[
                action
            ] = float(
                np.mean(vals)
            )

        if not data_list:
            continue

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.boxplot(
            data_list,
            labels=[
                a.capitalize()
                for a
                in valid_actions
            ],
            showfliers=True,
        )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_ylabel(
            r"$\Delta E = "
            r"E_{LLM} - "
            r"E_{structured}$"
        )

        ax.set_title(
            "Mechanistic effect of "
            "LLM-extracted cues by "
            "intended action"
        )

        path = (
            outdir
            / "fig8_llm_deltaE_by_action.png"
        )

        save_figure(
            fig,
            path,
        )

        print(
            "\n=== MEAN DELTA E BY ACTION ==="
        )

        for action in valid_actions:

            print(
                f"{action}: "
                f"{mean_values[action]:.6f}"
            )

        return {
            "created": True,
            "file": path.name,
            "source": source_name,
            "delta_column": delta_col,
            "mean_by_action": mean_values,
        }

    return {
        "created": False,
        "reason": (
            "No action-level delta_E "
            "column was found."
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Regenerate quantitative figures "
            "for the revised belonging paper."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Directory containing "
            "the revised CSV files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory in which "
            "the PNG figures will be saved."
        ),
    )

    parser.add_argument(
        "--include-controls",
        action="store_true",
        help=(
            "Include additional control "
            "conditions in Figures 1-5. "
            "Without this flag, only the "
            "five primary conditions are shown."
        ),
    )

    args = parser.parse_args()

    input_dir = Path(
        args.input_dir
    )

    output_dir = Path(
        args.output_dir
    )

    ensure_output_dir(
        output_dir
    )

    data, files = load_inputs(
        input_dir
    )

    print_belonging_diagnostic(
        data
    )

    manifest = {
        "input_dir": str(
            input_dir
        ),

        "output_dir": str(
            output_dir
        ),

        "include_controls": (
            args.include_controls
        ),

        "source_files": {
            key: (
                str(value)
                if value is not None
                else None
            )
            for key, value
            in files.items()
        },

        "figures": {},
    }

    generators = [

        (
            "fig1_belonging_trajectory",
            lambda:
            make_fig1_belonging_trajectory(
                data,
                output_dir,
                include_controls=(
                    args.include_controls
                ),
            ),
        ),

        (
            "fig2_helping_reciprocity",
            lambda:
            make_fig2_helping_reciprocity(
                data,
                output_dir,
                include_controls=(
                    args.include_controls
                ),
            ),
        ),

        (
            "fig3_participation_violin",
            lambda:
            make_fig3_participation_violin(
                data,
                output_dir,
                include_controls=(
                    args.include_controls
                ),
            ),
        ),

        (
            "fig4_belonging_participation_scatter",
            lambda:
            make_fig4_belonging_participation_scatter(
                data,
                output_dir,
                include_controls=(
                    args.include_controls
                ),
            ),
        ),

        (
            "fig5_action_composition",
            lambda:
            make_fig5_action_composition(
                data,
                output_dir,
                include_controls=(
                    args.include_controls
                ),
            ),
        ),

        (
            "fig6_llm_exact_match",
            lambda:
            make_fig6_llm_exact_match(
                data,
                output_dir,
            ),
        ),

        (
            "fig7_llm_cue_distributions",
            lambda:
            make_fig7_llm_cue_distributions(
                data,
                output_dir,
            ),
        ),

        (
            "fig8_llm_deltaE_by_action",
            lambda:
            make_fig8_llm_deltaE_by_action(
                data,
                output_dir,
            ),
        ),
    ]

    print(
        "\n======================================"
    )

    print(
        "GENERATING FIGURES"
    )

    print(
        "======================================\n"
    )

    for name, fn in generators:

        try:

            result = fn()

            manifest[
                "figures"
            ][name] = result

            status = (
                "OK"
                if result.get(
                    "created"
                )
                else "SKIPPED"
            )

            print(
                f"[{status}] "
                f"{name}: "
                f"{result}"
            )

        except Exception as exc:

            manifest[
                "figures"
            ][name] = {
                "created": False,
                "reason": str(
                    exc
                ),
            }

            print(
                f"[ERROR] "
                f"{name}: "
                f"{exc}"
            )

    manifest_path = (
        output_dir
        / "figure_generation_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n======================================"
    )

    print(
        "OK"
    )

    print(
        "======================================"
    )

    print(
        "\nManifest guardado en:"
    )

    print(
        manifest_path
    )


if __name__ == "__main__":
    main()