"""Model Performance — preserved evaluation results from results/, displayed, not recomputed."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import RESULTS
from ui.theme import metric_card


def render(selected: dict) -> None:
    st.markdown("## Model Performance & Research Results")
    st.markdown(
        '<p class="aa-secondary">All numbers on this page come from the pre-registered evaluation '
        "in <code>results/</code> (see <code>REPORT.md</code> and "
        "<code>docs/PREREGISTRATION.md</code>) — nothing here is recomputed live.</p>",
        unsafe_allow_html=True,
    )

    bench_test = pd.read_csv(RESULTS / "benchmark_test.csv")
    ablation = pd.read_csv(RESULTS / "ablation_test.csv")
    conformal = json.loads((RESULTS / "conformal.json").read_text())
    hp = json.loads((RESULTS / "high_pollution.json").read_text())
    boot = json.loads((RESULTS / "bootstrap_mae_diff.json").read_text())

    sel_row = bench_test[(bench_test.model == selected["model"]) & (bench_test.feature_set == selected["feature_set"])].iloc[0]
    pers_row = bench_test[bench_test.model == "persistence"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Selected model", f"{selected['model'].upper()}", "", f"Feature set {selected['feature_set']}")
    with c2:
        metric_card("Test MAE", f"{sel_row['mae']:.2f}", "µg/m³", f"Persistence baseline: {pers_row['mae']:.2f}")
    with c3:
        metric_card("Test RMSE", f"{sel_row['rmse']:.2f}", "µg/m³", "")
    with c4:
        metric_card("Bias", f"{sel_row['bias']:+.2f}", "µg/m³", "Mean (forecast − observed)")

    st.caption(
        f"Persistence-vs-selected MAE difference (block bootstrap, {boot['n_boot']} resamples): "
        f"**{boot['point_estimate_mae_diff']:.2f} µg/m³** (95% CI {boot['ci95_low']:.2f} to "
        f"{boot['ci95_high']:.2f}; positive = persistence has higher error)."
    )

    tabs = st.tabs(["Benchmark & ablation", "Forecast plots", "Conformal coverage", "High-pollution events"])

    with tabs[0]:
        st.markdown("#### Benchmark — test period (all models × feature sets)")
        display = bench_test.copy()
        display["selected"] = (display.model == selected["model"]) & (display.feature_set == selected["feature_set"])
        st.dataframe(display.round(2), use_container_width=True, hide_index=True)
        st.markdown("#### Feature ablation (test MAE, µg/m³)")
        st.dataframe(ablation.round(2), use_container_width=True, hide_index=True)
        st.image(str(RESULTS / "figures" / "ablation.png"), use_container_width=True)

    with tabs[1]:
        st.image(str(RESULTS / "figures" / "actual_vs_predicted.png"), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(str(RESULTS / "figures" / "residuals.png"), use_container_width=True)
        with col_b:
            st.image(str(RESULTS / "figures" / "high_pollution_scatter.png"), use_container_width=True)

    with tabs[2]:
        st.image(str(RESULTS / "figures" / "coverage_by_regime.png"), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Absolute-residual conformal**")
            st.json(conformal["absolute"]["overall"])
        with c2:
            st.markdown("**Log-space conformal (used in production)**")
            st.json(conformal["log"]["overall"])
        st.caption(
            "Coverage is empirical, measured once on the test period. It is not a guarantee of future "
            "coverage; it can drift if pollution patterns, monitor calibration or station siting change."
        )

    with tabs[3]:
        thr = hp["threshold_ugm3"]
        e1, e2, e3, e4 = st.columns(4)
        pr = hp["point_forecast_rule"]
        with e1:
            metric_card("Precision", f"{pr['precision']:.0%}", "", f"Threshold: {thr} µg/m³")
        with e2:
            metric_card("Recall", f"{pr['recall']:.0%}", "", "")
        with e3:
            metric_card("False alarms", f"{pr['false_alarms']}", "", "")
        with e4:
            metric_card("Missed events", f"{pr['missed_events']}", "", "")
        st.caption(
            f"Event-day MAE: **{hp['mae_on_event_days']:.1f} µg/m³** vs. non-event-day MAE "
            f"**{hp['mae_on_nonevent_days']:.1f} µg/m³** — average performance understates error on the "
            "days that matter most for health guidance."
        )

    st.markdown("#### Data split summary (train / validation / calibration / test)")
    split_summary = pd.read_csv(RESULTS / "split_summary.csv")
    st.dataframe(split_summary, use_container_width=True, hide_index=True)
    st.caption(
        "Train and validation decide the model and its hyper-parameters; calibration sets the "
        "conformal interval; test is evaluated exactly once and never used for any decision above."
    )
