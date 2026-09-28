import streamlit as st
import pandas as pd
import numpy as np

from methods.ranking import normalize_wsm, wsm_scores

st.set_page_config(page_title="MCDM Model Selection", page_icon="📊", layout="wide")

st.title("MCDM Model Selection")
st.subheader("Sélection du meilleur modèle de Machine Learning pour la détection de fraude bancaire")

st.markdown(
    """
Cette application compare plusieurs modèles de Machine Learning selon plusieurs critères.
Dans cette première version, les poids sont saisis manuellement et le classement est obtenu avec **WSM (Weighted Sum Method)**.
"""
)

# -------------------------
# 1. Données d'exemple
# -------------------------
@st.cache_data
def load_default_data():
    return pd.read_csv("data/example_data.csv")

if "decision_matrix" not in st.session_state:
    st.session_state.decision_matrix = load_default_data()

st.header("1. Matrice de décision")
st.caption("Vous pouvez modifier directement les valeurs du tableau.")

edited_df = st.data_editor(
    st.session_state.decision_matrix,
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
)

# Identifier alternatives et critères
alternative_col = "Model"
criteria = [c for c in edited_df.columns if c != alternative_col]

st.header("2. Sens des critères")
st.write("Choisissez si chaque critère doit être maximisé (+) ou minimisé (-).")

# Valeurs par défaut adaptées au scénario
DEFAULT_DIRECTIONS = {
    "Precision": "Maximiser (+)",
    "Recall": "Maximiser (+)",
    "F1-score": "Maximiser (+)",
    "AUPRC": "Maximiser (+)",
    "Training time (s)": "Minimiser (-)",
    "Prediction time (ms)": "Minimiser (-)",
    "Interpretability": "Maximiser (+)",
}

directions = {}
cols = st.columns(3)
for i, criterion in enumerate(criteria):
    with cols[i % 3]:
        default = DEFAULT_DIRECTIONS.get(criterion, "Maximiser (+)")
        directions[criterion] = st.selectbox(
            criterion,
            ["Maximiser (+)", "Minimiser (-)"],
            index=0 if default.startswith("Max") else 1,
            key=f"dir_{criterion}",
        )

st.header("3. Poids des critères")
st.write(
    "Attribuez une importance à chaque critère. Les valeurs sont automatiquement normalisées afin que leur somme soit égale à 1."
)

raw_weights = {}
weight_cols = st.columns(3)
for i, criterion in enumerate(criteria):
    with weight_cols[i % 3]:
        raw_weights[criterion] = st.number_input(
            f"Poids - {criterion}",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key=f"weight_{criterion}",
        )

weight_array = np.array([raw_weights[c] for c in criteria], dtype=float)
if weight_array.sum() > 0:
    normalized_weights = weight_array / weight_array.sum()
else:
    normalized_weights = np.ones(len(criteria)) / len(criteria)

weights_df = pd.DataFrame(
    {
        "Critère": criteria,
        "Poids normalisé": normalized_weights,
    }
)
st.dataframe(weights_df.style.format({"Poids normalisé": "{:.3f}"}), use_container_width=True, hide_index=True)

st.header("4. Méthode de classement")
method = st.selectbox("Méthode", ["WSM - Weighted Sum Method"])

st.divider()

if st.button("Calculer le classement", type="primary", use_container_width=True):
    try:
        numeric_matrix = edited_df[criteria].astype(float).to_numpy()
        benefit_flags = [directions[c].startswith("Max") for c in criteria]

        normalized_matrix = normalize_wsm(numeric_matrix, benefit_flags)
        scores = wsm_scores(normalized_matrix, normalized_weights)

        result_df = pd.DataFrame(
            {
                "Alternative": edited_df[alternative_col],
                "Score WSM": scores,
            }
        ).sort_values("Score WSM", ascending=False).reset_index(drop=True)

        result_df.insert(0, "Rang", np.arange(1, len(result_df) + 1))

        st.success(f"Meilleure alternative : {result_df.iloc[0]['Alternative']}")

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.subheader("Classement final")
            st.dataframe(
                result_df.style.format({"Score WSM": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

        with c2:
            st.subheader("Scores")
            chart_df = result_df.set_index("Alternative")[["Score WSM"]]
            st.bar_chart(chart_df)

        with st.expander("Voir la matrice normalisée"):
            normalized_df = pd.DataFrame(normalized_matrix, columns=criteria)
            normalized_df.insert(0, "Model", edited_df[alternative_col].values)
            st.dataframe(normalized_df.style.format(precision=4), use_container_width=True, hide_index=True)

        with st.expander("Comment le score WSM est-il calculé ?"):
            st.latex(r"Q_i = \sum_{j=1}^{n} w_j r_{ij}")
            st.write(
                "Chaque valeur normalisée rᵢⱼ est multipliée par le poids wⱼ du critère. "
                "L'alternative ayant le score Qᵢ le plus élevé est classée première."
            )

    except Exception as exc:
        st.error(f"Erreur lors du calcul : {exc}")

st.divider()
st.caption("V1 — Poids manuels + WSM. Prochaine étape : AHP, BWM, Entropie, CRITIC, TOPSIS, WASPAS et VIKOR.")
