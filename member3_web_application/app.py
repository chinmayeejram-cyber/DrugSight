"""DrugSight - AI-assisted EGFR drug discovery web app.

Run from the project root:
    streamlit run member3_web_application/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from rdkit.Chem import Draw

from admet.drug_likeness import admet_profile, summary_row
from explainability.shap_analysis import draw_explanation, explain_molecule
from member2_machine_learning.featurize import mol_from_smiles
from member2_machine_learning.predictor import Predictor
from member3_web_application import dashboard
from member3_web_application.molecule_input import library_upload, smiles_input


TARGET = "EGFR (Epidermal Growth Factor Receptor)"

st.set_page_config(
    page_title="DrugSight | EGFR Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        .stApp {
            background:
                radial-gradient(ellipse at 10% 0%, rgba(20, 75, 105, .28), transparent 36%),
                radial-gradient(ellipse at 90% 8%, rgba(76, 55, 130, .20), transparent 32%),
                #080e1b;
            color: #edf4ff;
            font-family: 'DM Sans', sans-serif;
        }
        [data-testid="stHeader"] { background: rgba(8, 14, 27, .8); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101b30, #090f1d);
            border-right: 1px solid #26364e;
        }
        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            color: #f1f6ff !important;
            letter-spacing: -.025em;
        }
        p, label, .stCaption { color: #c0ccdc; }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, #172940, #101a2c);
            border: 1px solid #30445e;
            border-radius: 16px;
            padding: 16px;
        }
        [data-testid="stMetricValue"] { color: #62e6dc !important; }
        [data-testid="stMetricLabel"] { color: #a7bad2 !important; }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 8px;
            background: #111c2d;
            padding: 9px;
            border-radius: 14px;
        }
        div[data-testid="stTabs"] button {
            border-radius: 10px;
            color: #b8c7dc;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: #243c57;
            color: #61e7dc;
        }
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            background: linear-gradient(100deg, #087e91, #5650a5);
            color: white;
            border: 1px solid #3c9fac;
            border-radius: 10px;
            font-weight: 600;
        }
        div[data-testid="stButton"] button:hover,
        div[data-testid="stDownloadButton"] button:hover {
            border-color: #65e8df;
            color: white;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            background: #142238 !important;
            color: white !important;
            border-radius: 10px !important;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #30445e;
            border-radius: 12px;
        }
        hr { border-color: #30445e; }
        .hero {
            padding: 28px 30px;
            border-radius: 20px;
            background: linear-gradient(110deg, #142c43, #252047);
            border: 1px solid #36516b;
            margin: 4px 0 22px 0;
        }
        .eyebrow {
            color: #60e5d9;
            letter-spacing: 3px;
            font-size: 12px;
            font-weight: 700;
        }
        .hero-title {
            color: #f4f8ff;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 42px;
            font-weight: 700;
            margin: 8px 0;
        }
        .hero-copy { color: #b5c7dc; font-size: 16px; }
        .tech-pill {
            display: inline-block;
            margin: 8px 7px 0 0;
            padding: 6px 10px;
            border: 1px solid rgba(97, 229, 217, .28);
            border-radius: 999px;
            color: #b9f7ef;
            background: rgba(97, 229, 217, .07);
            font-size: 12px;
        }
        .panel {
            background: linear-gradient(145deg, #172940, #101a2c);
            border: 1px solid #30445e;
            border-radius: 17px;
            padding: 18px 20px;
            margin-bottom: 14px;
        }
        .section-kicker {
            color: #60e5d9;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        .notice {
            border-left: 3px solid #eabf62;
            padding: 10px 14px;
            border-radius: 8px;
            background: rgba(234, 191, 98, .08);
            color: #e8d8b2;
            font-size: 13px;
        }
        .result-label { color: #60e5d9; font-weight: 700; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading DrugSight models...")
def get_predictor():
    return Predictor()


def verdict(probability: float):
    if probability >= 0.70:
        return "Likely ACTIVE"
    if probability >= 0.40:
        return "Uncertain"
    return "Likely INACTIVE"


def render_header(predictor):
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AI-POWERED CANCER DRUG DISCOVERY</div>
            <div class="hero-title">🧬 DrugSight</div>
            <div class="hero-copy">
                Discover. Predict. Explain.<br>
                Intelligent molecular bioactivity analysis targeting EGFR.
            </div>
            <span class="tech-pill">Random Forest</span>
            <span class="tech-pill">XGBoost</span>
            <span class="tech-pill">Graph Neural Network</span>
            <span class="tech-pill">SHAP</span>
            <span class="tech-pill">ADMET screening</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if predictor.ready:
        st.caption(f"Target: **{TARGET}** · Model status: 🟢 Models loaded")
    else:
        st.warning(
            "No trained models found. Run `python -m member2_machine_learning.train_all` "
            "from the project root, then refresh this page."
        )


def render_prediction(predictor):
    st.markdown('<div class="section-kicker">01 / MOLECULE ANALYSIS</div>', unsafe_allow_html=True)
    st.header("Predict & explain")
    st.write("Enter a molecule using SMILES or select an example molecule.")

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        smiles = smiles_input("predict")
        molecule = mol_from_smiles(smiles) if smiles else None

        if smiles and molecule is None:
            st.error("That is not a valid SMILES string.")

        if molecule is not None:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**Molecular structure**")
            st.image(
                Draw.MolToImage(molecule, size=(620, 430)),
                use_container_width=True,
            )
            st.caption("2D molecular structure generated using RDKit.")
            st.markdown("</div>", unsafe_allow_html=True)

    if molecule is None:
        with right:
            st.markdown(
                """
                <div class="panel">
                    <div class="section-kicker">READY WHEN YOU ARE</div>
                    <h3>Start with a molecule</h3>
                    <p>Prediction scores, molecular explanations and drug-likeness
                    results will appear here after you provide a valid SMILES.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    if not predictor.ready:
        with right:
            st.warning("Train the models before requesting a prediction.")
        return

    with st.spinner("Calculating predictions and molecular explanations..."):
        probabilities = predictor.predict(molecule)
        main_probability = probabilities.get(
            "Ensemble", next(iter(probabilities.values()))
        )
        explanation = explain_molecule(
            molecule, predictor.explainer_model()
        )
        profile = admet_profile(molecule)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">EGFR BIOACTIVITY RESULT</div>', unsafe_allow_html=True)
        st.markdown(
            f'<h2 class="result-label">{verdict(main_probability)}</h2>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:40px;font-weight:700;color:#ffffff;">'
            f'{main_probability:.0%}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Ensemble model score, not a laboratory measurement.")
        st.progress(float(main_probability))
        st.markdown("</div>", unsafe_allow_html=True)

        model_columns = st.columns(max(1, len(probabilities)))
        for column, (model_name, score) in zip(model_columns, probabilities.items()):
            column.metric(model_name, f"{score:.0%}")

    st.markdown("### Explainable AI")
    explain_col, detail_col = st.columns([1, 1], gap="large")

    with explain_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Molecular contribution map**")
        st.image(
            draw_explanation(molecule, explanation["atom_scores"]),
            caption="Green = contribution towards ACTIVE · Red = contribution towards INACTIVE",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with detail_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Top contributing substructures**")
        contributions = explanation.get("contributions", [])
        if contributions:
            contribution_table = pd.DataFrame(
                [
                    {
                        "SMILES fragment": item["fragment"],
                        "Model direction": (
                            "→ ACTIVE" if item["value"] > 0 else "→ INACTIVE"
                        ),
                        "Contribution": round(abs(item["value"]), 4),
                    }
                    for item in contributions
                ]
            )
            st.dataframe(
                contribution_table,
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No substructure contributions were returned.")
        st.caption(f"Explanation method: {explanation['method'].upper()}.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### ADMET & drug-likeness")
    flag = {
        "green": "🟢 Green",
        "orange": "🟠 Caution",
        "red": "🔴 Review",
    }.get(profile.get("traffic_light"), "⚪ Not available")

    qed_col, flag_col = st.columns(2)
    flag_col.metric("Overall screening flag", flag)
    qed_col.metric("QED drug-likeness", profile.get("qed", "N/A"))

    with st.expander("View drug-likeness rules and molecular properties", expanded=True):
        rules_col, properties_col = st.columns(2)
        with rules_col:
            st.markdown("**Rule-based filters**")
            for rule, (passed, detail) in profile.get("rules", {}).items():
                st.markdown(f"{'✅' if passed else '❌'} **{rule}**")
                st.caption(detail)

        with properties_col:
            st.markdown("**Molecular descriptors**")
            st.dataframe(
                pd.DataFrame(profile.get("properties", {}), index=["Value"]).T,
                use_container_width=True,
            )

        estimates = profile.get("estimates", {})
        if estimates:
            st.markdown("**Computational ADMET estimates**")
            estimate_table = pd.DataFrame(
                [(key, str(value)) for key, value in estimates.items()],
                columns=["Estimate", "Value"],
            )
            st.dataframe(estimate_table, hide_index=True, use_container_width=True)

        for alert_type, items in profile.get("alerts", {}).items():
            if items:
                st.warning(f"**{alert_type}:** " + ", ".join(items))

    st.markdown(
        '<div class="notice">ADMET outputs are computational estimates, not experimental '
        'measurements. Model predictions require validation using curated real EGFR data.</div>',
        unsafe_allow_html=True,
    )


def render_virtual_screening(predictor):
    st.markdown('<div class="section-kicker">02 / LIBRARY EXPLORATION</div>', unsafe_allow_html=True)
    st.header("Virtual screening")
    st.write(
        "Upload a molecule library to calculate model scores, apply drug-likeness "
        "filters and rank compounds for further investigation."
    )

    library = library_upload("screen")
    if library is None:
        st.info("Upload a CSV file containing a `smiles` column to begin.")
        return

    if not predictor.ready:
        st.warning("Train and load the models before screening a molecule library.")
        return

    rows = []
    progress = st.progress(0.0, text="Scoring molecules...")
    total = len(library)

    for index, smiles in enumerate(library["smiles"]):
        molecule = mol_from_smiles(smiles)
        if molecule is None:
            rows.append({})
        else:
            predictions = predictor.predict(molecule)
            row = {
                "score": predictions.get(
                    "Ensemble", next(iter(predictions.values()))
                )
            }
            row.update(summary_row(molecule))
            rows.append(row)

        progress.progress(
            (index + 1) / max(total, 1),
            text=f"Scoring molecules... {index + 1}/{total}",
        )

    progress.empty()

    results = pd.concat(
        [library.reset_index(drop=True), pd.DataFrame(rows)], axis=1
    ).dropna(subset=["score"])

    filter_col, threshold_col = st.columns(2)
    with filter_col:
        only_drug_like = st.checkbox(
            "Only show drug-like hits (Lipinski pass, no PAINS)",
            value=True,
        )
    with threshold_col:
        minimum_score = st.slider(
            "Minimum model score", 0.0, 1.0, 0.5, 0.05
        )

    filtered = results[results["score"] >= minimum_score]
    if only_drug_like:
        filtered = filtered[
            (filtered["Lipinski_pass"]) & (filtered["PAINS_alerts"] == 0)
        ]

    filtered = filtered.sort_values(
        "score", ascending=False
    ).reset_index(drop=True)

    st.metric("Molecules passing current filters", f"{len(filtered)} / {len(results)}")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.download_button(
        "Download ranked hits (CSV)",
        filtered.to_csv(index=False),
        "drugsight_hits.csv",
        mime="text/csv",
    )

    if not filtered.empty:
        top_molecules = [
            mol_from_smiles(smiles)
            for smiles in filtered["smiles"].head(6)
        ]
        top_scores = filtered["score"].head(6).tolist()
        st.markdown("### Top-ranked molecular structures")
        st.image(
            Draw.MolsToGridImage(
                top_molecules,
                molsPerRow=3,
                subImgSize=(260, 200),
                legends=[
                    f"#{index + 1} · score {score:.2f}"
                    for index, score in enumerate(top_scores)
                ],
            ),
            use_container_width=True,
        )


def render_about():
    st.markdown('<div class="section-kicker">05 / PROJECT OVERVIEW</div>', unsafe_allow_html=True)
    st.header("About DrugSight")
    st.write(
        "DrugSight is an academic research prototype for computational "
        "EGFR bioactivity prediction and molecular screening."
    )

    first_col, second_col = st.columns(2)
    with first_col:
        st.markdown(
            """
            <div class="panel">
                <h3>Research workflow</h3>
                <p>ChEMBL bioactivity data → molecular curation → RDKit descriptors
                and fingerprints → machine-learning models → SHAP explanations
                → drug-likeness assessment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with second_col:
        st.markdown(
            """
            <div class="panel">
                <h3>Models and methods</h3>
                <p>Random Forest, XGBoost and a Graph Neural Network are compared
                using held-out molecular scaffolds.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Scientific scope")
    st.markdown(
        f"""
        - **Target:** {TARGET}
        - **Input:** molecular SMILES or uploaded molecule libraries.
        - **Output:** model bioactivity scores, molecular explanations and drug-likeness estimates.
        - **Validation:** performance must be assessed on curated experimental ChEMBL data.
        """
    )
    st.markdown(
        '<div class="notice">DrugSight is a research prototype, not a diagnostic tool '
        'or a substitute for experimental testing. ADMET outputs are computational estimates.</div>',
        unsafe_allow_html=True,
    )


def main():
    inject_styles()
    predictor = get_predictor()

    with st.sidebar:
        st.markdown("## 🧬 DrugSight")
        st.caption("EGFR drug discovery workspace")
        st.divider()

        page = st.radio(
            "WORKSPACE",
            [
                "Predict & explain",
                "Virtual screening",
                "Model dashboard",
                "Explainability",
                "About",
            ],
        )

        st.divider()
        st.markdown("**TARGET PROFILE**")
        st.markdown("EGFR")
        st.caption("Epidermal Growth Factor Receptor")
        st.markdown(
            '<div class="notice">Prototype mode<br>'
            'Demo-trained predictions are not scientifically validated.</div>',
            unsafe_allow_html=True,
        )
        st.caption("DrugSight · Academic research prototype")

    render_header(predictor)

    if page == "Predict & explain":
        render_prediction(predictor)
    elif page == "Virtual screening":
        render_virtual_screening(predictor)
    elif page == "Model dashboard":
        dashboard.render_overview()
        st.divider()
        dashboard.render_models()
    elif page == "Explainability":
        dashboard.render_explainability()
    else:
        render_about()


if __name__ == "__main__":
    main()
