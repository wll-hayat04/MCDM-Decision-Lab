import streamlit as st
import pandas as pd

from ui import (
    load_css,
    show_header,
    show_sidebar,
    show_decision_matrix,
    show_criteria_controls,
    show_manual_weights,
    show_ahp_input,
    show_bwm_input,
    show_weights_result,
    show_ranking_details,
    show_results
)

from utils import (
    prepare_decision_matrix,
    validate_decision_matrix,
    ranking_dataframe
)

from methods.weighting import (
    manual_weights,
    ahp_weights,
    bwm_weights,
    entropy_weights,
    critic_weights
)

from methods.ranking import (
    wsm,
    wpm,
    waspas,
    topsis,
    vikor
)


# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="MCDM Decision Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DESIGN
# ============================================================

load_css()
show_header()


# ============================================================
# CONFIGURATION PAR DEFAUT DES CRITERES
# ============================================================

DEFAULT_DIRECTIONS = {

    "Precision":
        "Maximiser (+)",

    "Recall":
        "Maximiser (+)",

    "F1-score":
        "Maximiser (+)",

    "AUPRC":
        "Maximiser (+)",

    "Training time (s)":
        "Minimiser (-)",

    "Prediction time (ms)":
        "Minimiser (-)",

    "Interpretability":
        "Maximiser (+)"
}


# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/model_selection.csv"
    )


# ============================================================
# SIDEBAR
# ============================================================

(
    weighting_method,
    ranking_method,
    waspas_lambda,
    vikor_v

) = show_sidebar()


# ============================================================
# 1. MATRICE DE DECISION
# ============================================================

raw_df = show_decision_matrix(
    load_data()
)


# ============================================================
# NETTOYAGE DES DONNEES
# ============================================================

try:

    df, ignored_rows = (
        prepare_decision_matrix(
            raw_df,
            alternative_col="Model"
        )
    )

except ValueError as error:

    st.error(
        f"Impossible de préparer la matrice : {error}"
    )

    st.stop()


# ------------------------------------------------------------
# Lignes incomplètes
# ------------------------------------------------------------

if ignored_rows > 0:

    st.warning(
        f"{ignored_rows} ligne(s) incomplète(s) "
        "sont temporairement ignorée(s) dans le calcul. "
        "Complétez toutes les cellules pour les intégrer "
        "à l'analyse."
    )


# ============================================================
# VALIDATION MATRICE
# ============================================================

try:

    criteria, matrix = (
        validate_decision_matrix(
            df,
            alternative_col="Model"
        )
    )

except ValueError as error:

    st.error(
        f"Matrice de décision invalide : {error}"
    )

    st.info(
        "Ajoutez ou complétez les alternatives "
        "avant de poursuivre."
    )

    st.stop()


alternatives = (
    df["Model"]
    .astype(str)
    .tolist()
)


# ============================================================
# 2. NATURE DES CRITERES
# ============================================================

try:

    benefit_flags = (
        show_criteria_controls(
            criteria,
            DEFAULT_DIRECTIONS
        )
    )

except Exception as error:

    st.error(
        f"Erreur dans la configuration des critères : {error}"
    )

    st.stop()


# ============================================================
# 3. PONDERATION
# ============================================================

try:

    # --------------------------------------------------------
    # MANUAL
    # --------------------------------------------------------

    if weighting_method == "Manual":

        raw_weights = (
            show_manual_weights(
                criteria
            )
        )

        weights, w_details = (
            manual_weights(
                raw_weights
            )
        )


    # --------------------------------------------------------
    # AHP
    # --------------------------------------------------------

    elif weighting_method == "AHP":

        pairwise_matrix = (
            show_ahp_input(
                criteria
            )
        )

        weights, w_details = (
            ahp_weights(
                pairwise_matrix
            )
        )


    # --------------------------------------------------------
    # BWM
    # --------------------------------------------------------

    elif weighting_method == "BWM":

        (
            best_index,
            worst_index,
            best_to_others,
            others_to_worst

        ) = show_bwm_input(
            criteria
        )

        weights, w_details = (
            bwm_weights(
                best_index,
                worst_index,
                best_to_others,
                others_to_worst
            )
        )


    # --------------------------------------------------------
    # ENTROPY
    # --------------------------------------------------------

    elif weighting_method == "Entropy":

        weights, w_details = (
            entropy_weights(
                matrix
            )
        )


    # --------------------------------------------------------
    # CRITIC
    # --------------------------------------------------------

    elif weighting_method == "CRITIC":

        weights, w_details = (
            critic_weights(
                matrix,
                benefit_flags
            )
        )


    else:

        raise ValueError(
            "Méthode de pondération inconnue."
        )


except (ValueError, TypeError) as error:

    st.error(
        f"Impossible de calculer les poids : {error}"
    )

    if weighting_method == "Manual":

        st.info(
            "Attribuez au moins un poids "
            "strictement positif."
        )

    elif weighting_method == "AHP":

        st.info(
            "Vérifiez les comparaisons par paires AHP."
        )

    elif weighting_method == "BWM":

        st.info(
            "Vérifiez les choix Best/Worst "
            "et les comparaisons BWM."
        )

    st.stop()


except Exception as error:

    st.error(
        "Une erreur inattendue est survenue "
        "pendant la pondération."
    )

    st.caption(str(error))

    st.stop()


# ============================================================
# 4. AFFICHAGE DES POIDS
# ============================================================

show_weights_result(
    criteria,
    weights,
    w_details
)


# ============================================================
# 5. CLASSEMENT
# ============================================================

try:

    # --------------------------------------------------------
    # WSM
    # --------------------------------------------------------

    if ranking_method == "WSM":

        scores, r_details = wsm(
            matrix,
            weights,
            benefit_flags
        )

        score_col = "Score WSM"


    # --------------------------------------------------------
    # WPM
    # --------------------------------------------------------

    elif ranking_method == "WPM":

        scores, r_details = wpm(
            matrix,
            weights,
            benefit_flags
        )

        score_col = "Score WPM"


    # --------------------------------------------------------
    # WASPAS
    # --------------------------------------------------------

    elif ranking_method == "WASPAS":

        scores, r_details = waspas(
            matrix,
            weights,
            benefit_flags,
            lam=waspas_lambda
        )

        score_col = "Score WASPAS"


    # --------------------------------------------------------
    # TOPSIS
    # --------------------------------------------------------

    elif ranking_method == "TOPSIS":

        scores, r_details = topsis(
            matrix,
            weights,
            benefit_flags
        )

        score_col = "RC TOPSIS"


    # --------------------------------------------------------
    # VIKOR
    # --------------------------------------------------------

    elif ranking_method == "VIKOR":

        scores, r_details = vikor(
            matrix,
            weights,
            benefit_flags,
            v=vikor_v
        )

        score_col = "Q VIKOR"


    else:

        raise ValueError(
            "Méthode de classement inconnue."
        )


except (ValueError, TypeError, ZeroDivisionError) as error:

    st.error(
        f"Impossible d'appliquer "
        f"{ranking_method} : {error}"
    )

    st.info(
        "Vérifiez la matrice de décision, "
        "les poids et le sens des critères."
    )

    st.stop()


except Exception as error:

    st.error(
        f"Une erreur inattendue est survenue "
        f"avec {ranking_method}."
    )

    st.caption(str(error))

    st.stop()


# ============================================================
# DETAILS DES CALCULS
# ============================================================

show_ranking_details(
    criteria,
    alternatives,
    r_details
)


# ============================================================
# 6. CALCUL DU CLASSEMENT FINAL
# ============================================================

run = st.button(
    "Calculer le classement final",
    type="primary",
    width="stretch"
)


if run:

    try:

        higher_is_better = (
            r_details.get(
                "higher_is_better",
                True
            )
        )


        result_df = (
            ranking_dataframe(
                alternatives,
                scores,
                higher_is_better=
                    higher_is_better,
                score_name=
                    score_col
            )
        )


        show_results(
            result_df,
            score_col
        )


    except (ValueError, TypeError) as error:

        st.error(
            f"Impossible de produire "
            f"le classement final : {error}"
        )


    except Exception as error:

        st.error(
            "Une erreur inattendue est survenue "
            "pendant le classement final."
        )

        st.caption(str(error))


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "MCDM Decision Lab — "
    "Pondération : Manual, AHP, BWM, Entropy, CRITIC • "
    "Classement : WSM, WPM, WASPAS, TOPSIS, VIKOR."
)
