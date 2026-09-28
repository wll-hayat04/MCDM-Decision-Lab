import streamlit as st
import pandas as pd

from ui import (
    load_css, show_header, show_sidebar, show_decision_matrix,
    show_criteria_controls, show_manual_weights, show_ahp_input,
    show_bwm_input, show_weights_result, show_ranking_details, show_results
)
from utils import validate_decision_matrix, ranking_dataframe
from methods.weighting import (
    manual_weights, ahp_weights, bwm_weights, entropy_weights, critic_weights
)
from methods.ranking import wsm, wpm, waspas, topsis, vikor

st.set_page_config(
    page_title="MCDM Decision Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()
show_header()

DEFAULT_DIRECTIONS = {
    "Precision": "Maximiser (+)",
    "Recall": "Maximiser (+)",
    "F1-score": "Maximiser (+)",
    "AUPRC": "Maximiser (+)",
    "Training time (s)": "Minimiser (-)",
    "Prediction time (ms)": "Minimiser (-)",
    "Interpretability": "Maximiser (+)",
}

@st.cache_data
def load_data():
    return pd.read_csv("data/model_selection.csv")

weighting_method, ranking_method, waspas_lambda, vikor_v = show_sidebar()

df = show_decision_matrix(load_data())
criteria, matrix = validate_decision_matrix(df, "Model")
alternatives = df["Model"].tolist()

benefit_flags = show_criteria_controls(criteria, DEFAULT_DIRECTIONS)

# ---------------- PONDERATION ----------------
if weighting_method == "Manual":
    raw = show_manual_weights(criteria)
    weights, w_details = manual_weights(raw)

elif weighting_method == "AHP":
    pairwise = show_ahp_input(criteria)
    weights, w_details = ahp_weights(pairwise)

elif weighting_method == "BWM":
    best_idx, worst_idx, BO, OW = show_bwm_input(criteria)
    weights, w_details = bwm_weights(best_idx, worst_idx, BO, OW)

elif weighting_method == "Entropy":
    # Méthode objective: calcul direct à partir de la matrice.
    weights, w_details = entropy_weights(matrix)

elif weighting_method == "CRITIC":
    weights, w_details = critic_weights(matrix, benefit_flags)

show_weights_result(criteria, weights, w_details)

# ---------------- CLASSEMENT ----------------
if ranking_method == "WSM":
    scores, r_details = wsm(matrix, weights, benefit_flags)
    score_col = "Score WSM"

elif ranking_method == "WPM":
    scores, r_details = wpm(matrix, weights, benefit_flags)
    score_col = "Score WPM"

elif ranking_method == "WASPAS":
    scores, r_details = waspas(matrix, weights, benefit_flags, lam=waspas_lambda)
    score_col = "Score WASPAS"

elif ranking_method == "TOPSIS":
    scores, r_details = topsis(matrix, weights, benefit_flags)
    score_col = "RC TOPSIS"

elif ranking_method == "VIKOR":
    scores, r_details = vikor(matrix, weights, benefit_flags, v=vikor_v)
    score_col = "Q VIKOR"

show_ranking_details(criteria, alternatives, r_details)

run = st.button("Calculer le classement final", type="primary", use_container_width=True)

if run:
    higher_is_better = r_details.get("higher_is_better", True)
    result_df = ranking_dataframe(
        alternatives,
        scores,
        higher_is_better=higher_is_better,
        score_name=score_col
    )
    show_results(result_df, score_col)

st.markdown("---")
st.caption(
    "MCDM Decision Lab — pondération: Manual, AHP, BWM, Entropy, CRITIC • "
    "classement: WSM, WPM, WASPAS, TOPSIS, VIKOR."
)
