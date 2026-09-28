import streamlit as st
import pandas as pd
import numpy as np
import textwrap

# ============================================================
# CONSTANTES
# ============================================================

SAATY_VALUES = {
    "1 — Importance égale": 1.0,
    "2 — Valeur intermédiaire": 2.0,
    "3 — Importance modérée": 3.0,
    "4 — Valeur intermédiaire": 4.0,
    "5 — Importance forte": 5.0,
    "6 — Valeur intermédiaire": 6.0,
    "7 — Très forte importance": 7.0,
    "8 — Valeur intermédiaire": 8.0,
    "9 — Importance extrême": 9.0,
}

def html(content):
    st.markdown(
        textwrap.dedent(content).strip(),
        unsafe_allow_html=True
    )
# ============================================================
# FONCTIONS UTILITAIRES UI
# ============================================================

def number_columns(columns, precision=4):
    """
    Configuration d'affichage numérique pour st.dataframe.
    Évite l'utilisation de pandas Styler.
    """
    fmt = f"%.{precision}f"

    return {
        col: st.column_config.NumberColumn(
            col,
            format=fmt
        )
        for col in columns
    }


# ============================================================
# DESIGN GLOBAL
# ============================================================

def load_css():

    st.markdown(
        """
        <style>

        /* ============================
           GLOBAL
        ============================ */

        .stApp {
            background:
                radial-gradient(
                    circle at 10% 10%,
                    rgba(37, 99, 235, 0.08),
                    transparent 28%
                ),
                linear-gradient(
                    180deg,
                    #f8fafc 0%,
                    #eef2ff 100%
                );
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }


        /* ============================
           HEADER
        ============================ */

        .hero {
            background:
                linear-gradient(
                    135deg,
                    #0f172a 0%,
                    #1d4ed8 100%
                );

            color: white;

            padding: 32px 34px;

            border-radius: 24px;

            box-shadow:
                0 18px 50px
                rgba(15, 23, 42, 0.18);

            margin-bottom: 20px;
        }

        .hero h1 {
            font-size: 2.2rem;
            margin: 0 0 8px 0;
            letter-spacing: -0.04em;
        }

        .hero p {
            margin: 0;
            opacity: 0.92;
            line-height: 1.65;
            font-size: 1rem;
        }

        .eyebrow {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            opacity: 0.75;
            margin-bottom: 8px;
        }


        /* ============================
           CARDS
        ============================ */

        .card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 18px;

            padding: 18px 20px;

            box-shadow:
                0 8px 26px
                rgba(15, 23, 42, 0.05);

            margin-top: 8px;
            margin-bottom: 14px;
        }

        .card-title {
            font-weight: 800;
            color: #0f172a;
            font-size: 1.05rem;
            margin-bottom: 4px;
        }

        .card-text {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.55;
        }


        /* ============================
           NUMEROS ETAPES
        ============================ */

        .step {
            width: 34px;
            height: 34px;

            border-radius: 10px;

            display: inline-flex;

            align-items: center;
            justify-content: center;

            background: #dbeafe;
            color: #1d4ed8;

            font-weight: 800;

            margin-right: 10px;
        }


        /* ============================
           WINNER
        ============================ */

        .winner {
            background:
                linear-gradient(
                    135deg,
                    #dcfce7,
                    #ecfccb
                );

            border: 1px solid #bbf7d0;

            color: #14532d;

            padding: 18px 22px;

            border-radius: 18px;

            font-weight: 800;

            box-shadow:
                0 8px 26px
                rgba(34, 197, 94, 0.10);

            margin: 10px 0 18px;
        }


        /* ============================
           FORMULES
        ============================ */

        .formula-box {
            background: #f8fafc;

            border-left:
                4px solid #2563eb;

            border-radius: 10px;

            padding: 12px 16px;

            margin: 8px 0 14px;

            color: #334155;
        }


        /* ============================
           SIDEBAR
        ============================ */

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    #0f172a,
                    #1e293b
                );
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p {
            color: white !important;
        }


        /* ============================
           BUTTONS
        ============================ */

        div.stButton > button {
            border-radius: 14px;
            font-weight: 800;
            min-height: 46px;
        }


        /* ============================
           DATAFRAMES
        ============================ */

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================
def show_header():

    html("""
    <div class="hero">
        <div class="eyebrow">
            Decision Intelligence • MCDM
        </div>

        <h1>MCDM Decision Lab</h1>

        <p>
            Sélection multicritère du meilleur modèle de
            Machine Learning pour la détection de fraude bancaire.
            L'application détaille la pondération,
            la normalisation, l'agrégation
            et le classement des alternatives.
        </p>
    </div>
    """)


# ============================================================
# TITRES DES SECTIONS
# ============================================================

def section(step, title, text):

    html(f"""
    <div class="card">
        <div class="card-title">
            <span class="step">{step}</span>
            {title}
        </div>

        <div class="card-text">
            {text}
        </div>
    </div>
    """)
# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():

    with st.sidebar:

        st.markdown("## MCDM Decision Lab")

        st.caption(
            "Configuration de l'analyse multicritère."
        )

        st.markdown("---")

        st.markdown("### Pondération")

        weighting = st.selectbox(
            "Méthode de pondération",
            [
                "Manual",
                "AHP",
                "BWM",
                "Entropy",
                "CRITIC"
            ]
        )

        st.markdown("### Classement")

        ranking = st.selectbox(
            "Méthode de classement",
            [
                "WSM",
                "WPM",
                "WASPAS",
                "TOPSIS",
                "VIKOR"
            ]
        )

        lam = 0.5
        vikor_v = 0.5

        if ranking == "WASPAS":

            st.markdown("---")

            lam = st.slider(
                "λ — compromis WSM / WPM",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )

        if ranking == "VIKOR":

            st.markdown("---")

            vikor_v = st.slider(
                "v — utilité de groupe",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )

        st.markdown("---")

        st.caption(
            "Benefit (+) : à maximiser"
        )

        st.caption(
            "Cost (−) : à minimiser"
        )

    return (
        weighting,
        ranking,
        lam,
        vikor_v
    )


# ============================================================
# TELECHARGER MATRICE
# ============================================================

def download_matrix(
    df,
    key="download_matrix"
):

    csv = (
        df
        .to_csv(index=False)
        .encode("utf-8-sig")
    )

    st.download_button(
        label="Télécharger la matrice CSV",
        data=csv,
        file_name="decision_matrix.csv",
        mime="text/csv",
        width="stretch",
        key=key
    )


# ============================================================
# MATRICE DE DECISION
# ============================================================

def show_decision_matrix(default_df):

    section(
        "1",
        "Matrice de décision",
        """
        Utilisez les données d'exemple,
        construisez votre matrice manuellement
        ou importez votre propre fichier CSV.
        """
    )

    source = st.radio(
        "Source des données",
        [
            "Données d'exemple",
            "Saisie manuelle",
            "Importer un CSV"
        ],
        horizontal=True
    )

    st.markdown("")


    # ========================================================
    # DONNEES EXEMPLE
    # ========================================================

    if source == "Données d'exemple":

        st.info(
            "Vous pouvez modifier les valeurs, "
            "ajouter des alternatives ou supprimer des lignes."
        )

        edited_df = st.data_editor(
            default_df.copy(),
            width="stretch",
            hide_index=True,
            num_rows="dynamic",
            key="example_editor"
        )

        download_matrix(
            edited_df,
            key="download_example"
        )

        return edited_df


    # ========================================================
    # SAISIE MANUELLE
    # ========================================================

    elif source == "Saisie manuelle":

        st.info(
            "Remplissez directement votre matrice "
            "de décision."
        )

        if "manual_matrix" not in st.session_state:

            manual_df = default_df.copy()

            manual_df["Model"] = [
                f"Model {i + 1}"
                for i in range(
                    len(manual_df)
                )
            ]

            numeric_cols = [
                col
                for col in manual_df.columns
                if col != "Model"
            ]

            for col in numeric_cols:
                manual_df[col] = 0.0

            st.session_state.manual_matrix = manual_df


        if st.button(
            "Réinitialiser la matrice",
            width="content"
        ):

            manual_df = default_df.copy()

            manual_df["Model"] = [
                f"Model {i + 1}"
                for i in range(
                    len(manual_df)
                )
            ]

            numeric_cols = [
                col
                for col in manual_df.columns
                if col != "Model"
            ]

            for col in numeric_cols:
                manual_df[col] = 0.0

            st.session_state.manual_matrix = manual_df

            if "manual_editor" in st.session_state:
                del st.session_state["manual_editor"]

            st.rerun()


        edited_df = st.data_editor(
            st.session_state.manual_matrix,
            width="stretch",
            hide_index=True,
            num_rows="dynamic",
            key="manual_editor"
        )

        download_matrix(
            edited_df,
            key="download_manual"
        )

        return edited_df


    # ========================================================
    # IMPORT CSV
    # ========================================================

    else:

        st.markdown(
            """
            Le fichier CSV doit contenir :

            - une colonne **Model**
            - une ligne par alternative
            - une colonne numérique par critère
            """
        )

        uploaded_file = st.file_uploader(
            "Importer la matrice",
            type=["csv"]
        )

        if uploaded_file is None:

            st.warning(
                "Importez un fichier CSV pour continuer."
            )

            st.stop()


        try:

            df = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python"
            )

        except Exception as error:

            st.error(
                f"Erreur de lecture du fichier : {error}"
            )

            st.stop()


        if "Model" not in df.columns:

            st.error(
                "Le fichier doit contenir une colonne "
                "nommée exactement 'Model'."
            )

            st.stop()


        st.success(
            f"{len(df)} alternatives et "
            f"{len(df.columns) - 1} critères chargés."
        )


        edited_df = st.data_editor(
            df,
            width="stretch",
            hide_index=True,
            num_rows="dynamic",
            key="uploaded_editor"
        )

        download_matrix(
            edited_df,
            key="download_uploaded"
        )

        return edited_df


# ============================================================
# SENS DES CRITERES
# ============================================================

def show_criteria_controls(
    criteria,
    defaults
):

    section(
        "2",
        "Nature des critères",
        """
        Définissez chaque critère comme
        Benefit (+) lorsqu'il doit être maximisé
        ou Cost (−) lorsqu'il doit être minimisé.
        """
    )

    flags = []

    cols = st.columns(3)

    for i, criterion in enumerate(criteria):

        with cols[i % 3]:

            default = defaults.get(
                criterion,
                "Maximiser (+)"
            )

            choice = st.selectbox(
                criterion,
                [
                    "Maximiser (+)",
                    "Minimiser (-)"
                ],
                index=(
                    0
                    if default.startswith("Max")
                    else 1
                ),
                key=f"direction_{criterion}"
            )

            flags.append(
                choice.startswith("Max")
            )

    return flags


# ============================================================
# POIDS MANUELS
# ============================================================

def show_manual_weights(criteria):

    section(
        "3",
        "Pondération manuelle",
        """
        Saisissez l'importance de chaque critère.
        Les valeurs seront automatiquement normalisées
        afin que leur somme soit égale à 1.
        """
    )

    values = []

    cols = st.columns(3)

    for i, criterion in enumerate(criteria):

        with cols[i % 3]:

            value = st.number_input(
                f"Poids — {criterion}",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key=f"weight_{criterion}"
            )

            values.append(value)

    return values


# ============================================================
# AHP
# ============================================================

def show_ahp_input(criteria):

    section(
        "3",
        "Pondération AHP",
        """
        Comparez les critères deux à deux
        avec l'échelle de Saaty de 1 à 9.
        """
    )

    n = len(criteria)

    A = np.ones(
        (n, n),
        dtype=float
    )

    st.markdown(
        "### Comparaisons par paires"
    )

    st.caption(
        "Choisissez le critère dominant "
        "puis son niveau d'importance."
    )


    for i in range(n):

        for j in range(
            i + 1,
            n
        ):

            c1, c2 = st.columns(
                [1.3, 1]
            )

            with c1:

                dominant = st.selectbox(
                    f"{criteria[i]}  vs  {criteria[j]}",
                    [
                        "Importance égale",
                        criteria[i],
                        criteria[j]
                    ],
                    key=f"dominant_{i}_{j}"
                )

            with c2:

                if dominant == "Importance égale":

                    intensity = 1.0

                    st.text_input(
                        "Intensité",
                        value="1 — Importance égale",
                        disabled=True,
                        key=f"equal_{i}_{j}"
                    )

                else:

                    label = st.selectbox(
                        "Échelle de Saaty",
                        list(
                            SAATY_VALUES.keys()
                        ),
                        index=2,
                        key=f"intensity_{i}_{j}"
                    )

                    intensity = (
                        SAATY_VALUES[label]
                    )


            if dominant == "Importance égale":

                A[i, j] = 1.0
                A[j, i] = 1.0

            elif dominant == criteria[i]:

                A[i, j] = intensity
                A[j, i] = (
                    1.0 / intensity
                )

            else:

                A[i, j] = (
                    1.0 / intensity
                )

                A[j, i] = intensity


    st.markdown(
        "### Matrice de comparaison AHP"
    )

    ahp_df = pd.DataFrame(
        A,
        columns=criteria,
        index=criteria
    )

    ahp_display = (
        ahp_df
        .reset_index()
        .rename(
            columns={
                "index": "Critère"
            }
        )
    )

    st.dataframe(
        ahp_display,
        width="stretch",
        hide_index=True,
        column_config=number_columns(
            criteria,
            3
        )
    )

    return A


# ============================================================
# BWM
# ============================================================

def show_bwm_input(criteria):

    section(
        "3",
        "Pondération BWM",
        """
        Choisissez le critère le plus important (Best)
        et le moins important (Worst),
        puis renseignez les comparaisons.
        """
    )


    best = st.selectbox(
        "Critère le plus important — Best",
        criteria,
        key="bwm_best"
    )


    worst_choices = [
        c
        for c in criteria
        if c != best
    ]


    worst = st.selectbox(
        "Critère le moins important — Worst",
        worst_choices,
        key="bwm_worst"
    )


    best_index = criteria.index(
        best
    )

    worst_index = criteria.index(
        worst
    )


    # ========================================================
    # BEST -> OTHERS
    # ========================================================

    st.markdown(
        "### Best → Others"
    )

    st.caption(
        f"Combien de fois {best} est-il "
        "plus important que les autres critères ?"
    )


    best_to_others = []

    cols = st.columns(3)

    for i, criterion in enumerate(criteria):

        with cols[i % 3]:

            if i == best_index:

                value = 1

                st.number_input(
                    criterion,
                    min_value=1,
                    max_value=1,
                    value=1,
                    disabled=True,
                    key=f"bo_{i}"
                )

            else:

                value = st.slider(
                    criterion,
                    min_value=1,
                    max_value=9,
                    value=3,
                    key=f"bo_{i}"
                )

            best_to_others.append(
                value
            )


    # ========================================================
    # OTHERS -> WORST
    # ========================================================

    st.markdown(
        "### Others → Worst"
    )

    st.caption(
        f"Combien de fois chaque critère "
        f"est-il plus important que {worst} ?"
    )


    others_to_worst = []

    cols = st.columns(3)

    for i, criterion in enumerate(criteria):

        with cols[i % 3]:

            if i == worst_index:

                value = 1

                st.number_input(
                    criterion,
                    min_value=1,
                    max_value=1,
                    value=1,
                    disabled=True,
                    key=f"ow_{i}"
                )

            else:

                value = st.slider(
                    criterion,
                    min_value=1,
                    max_value=9,
                    value=3,
                    key=f"ow_{i}"
                )

            others_to_worst.append(
                value
            )


    return (
        best_index,
        worst_index,
        best_to_others,
        others_to_worst
    )


# ============================================================
# AFFICHAGE DES POIDS
# ============================================================

def show_weights_result(
    criteria,
    weights,
    details
):

    section(
        "4",
        "Poids des critères",
        """
        Les poids obtenus sont normalisés.
        Leur somme est égale à 1.
        """
    )


    weights_df = pd.DataFrame(
        {
            "Critère": criteria,
            "Poids": weights
        }
    )


    st.dataframe(
        weights_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Poids":
                st.column_config.NumberColumn(
                    "Poids",
                    format="%.4f"
                )
        }
    )


    chart_df = (
        weights_df
        .set_index("Critère")
    )

    st.bar_chart(
        chart_df["Poids"]
    )


    method = details.get(
        "method"
    )


    # ========================================================
    # AHP
    # ========================================================

    if method == "AHP":

        st.markdown(
            "### Test de consistance"
        )


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "λmax",
            f"{details['lambda_max']:.4f}"
        )


        c2.metric(
            "CI",
            f"{details['CI']:.4f}"
        )


        c3.metric(
            "CR",
            f"{details['CR']:.4f}"
        )


        if details["consistent"]:

            st.success(
                "CR < 0,10 : "
                "la matrice est cohérente."
            )

        else:

            st.warning(
                "CR ≥ 0,10 : "
                "les comparaisons doivent être révisées."
            )


        with st.expander(
            "Voir la matrice AHP normalisée"
        ):

            normalized_df = pd.DataFrame(
                details[
                    "normalized_pairwise"
                ],
                columns=criteria
            )

            normalized_df.insert(
                0,
                "Critère",
                criteria
            )

            st.dataframe(
                normalized_df,
                width="stretch",
                hide_index=True,
                column_config=number_columns(
                    criteria,
                    4
                )
            )


    # ========================================================
    # BWM
    # ========================================================

    elif method == "BWM":

        st.metric(
            "ξ* — erreur maximale",
            f"{details['xi']:.4f}"
        )

        st.caption(
            "Plus ξ* est proche de 0, "
            "plus les jugements sont cohérents."
        )


    # ========================================================
    # ENTROPY
    # ========================================================

    elif method == "Entropy":

        with st.expander(
            "Voir les calculs d'entropie"
        ):

            entropy_df = pd.DataFrame(
                {
                    "Critère": criteria,

                    "Entropie E_j":
                        details["entropy"],

                    "Diversification 1-E_j":
                        details[
                            "diversification"
                        ]
                }
            )


            st.dataframe(
                entropy_df,
                width="stretch",
                hide_index=True,
                column_config=number_columns(
                    [
                        "Entropie E_j",
                        "Diversification 1-E_j"
                    ],
                    4
                )
            )


        if (
            "normalized_matrix"
            in details
        ):

            with st.expander(
                "Voir la matrice Entropy"
            ):

                matrix_df = pd.DataFrame(
                    details[
                        "normalized_matrix"
                    ],
                    columns=criteria
                )

                st.dataframe(
                    matrix_df,
                    width="stretch",
                    hide_index=True,
                    column_config=number_columns(
                        criteria,
                        4
                    )
                )


    # ========================================================
    # CRITIC
    # ========================================================

    elif method == "CRITIC":

        with st.expander(
            "Voir les indices CRITIC"
        ):

            critic_df = pd.DataFrame(
                {
                    "Critère": criteria,

                    "σ_j":
                        details["sigma"],

                    "C_j":
                        details["C"]
                }
            )


            st.dataframe(
                critic_df,
                width="stretch",
                hide_index=True,
                column_config=number_columns(
                    ["σ_j", "C_j"],
                    4
                )
            )


        with st.expander(
            "Voir la matrice de corrélation"
        ):

            corr_df = pd.DataFrame(
                details["correlation"],
                columns=criteria
            )

            corr_df.insert(
                0,
                "Critère",
                criteria
            )

            st.dataframe(
                corr_df,
                width="stretch",
                hide_index=True,
                column_config=number_columns(
                    criteria,
                    3
                )
            )


        if (
            "normalized_matrix"
            in details
        ):

            with st.expander(
                "Voir la matrice normalisée CRITIC"
            ):

                matrix_df = pd.DataFrame(
                    details[
                        "normalized_matrix"
                    ],
                    columns=criteria
                )

                st.dataframe(
                    matrix_df,
                    width="stretch",
                    hide_index=True,
                    column_config=number_columns(
                        criteria,
                        4
                    )
                )


# ============================================================
# NORMALISATION + DETAILS CLASSEMENT
# ============================================================

def show_ranking_details(
    criteria,
    alternatives,
    details
):

    section(
        "5",
        "Normalisation et calcul",
        """
        Cette partie présente les étapes
        intermédiaires propres à la méthode
        de classement sélectionnée.
        """
    )


    method = details["method"]


    # ========================================================
    # MATRICE NORMALISEE
    # ========================================================

    if (
        "normalized_matrix"
        in details
    ):

        st.markdown(
            "### Matrice normalisée"
        )


        normalized_df = pd.DataFrame(
            details[
                "normalized_matrix"
            ],
            columns=criteria
        )


        normalized_df.insert(
            0,
            "Alternative",
            list(alternatives)
        )


        st.dataframe(
            normalized_df,
            width="stretch",
            hide_index=True,
            column_config=number_columns(
                criteria,
                4
            )
        )


    # ========================================================
    # MATRICE NORMALISEE PONDEREE
    # ========================================================

    if (
        "weighted_matrix"
        in details
    ):

        st.markdown(
            "### Matrice normalisée pondérée"
        )


        weighted_df = pd.DataFrame(
            details[
                "weighted_matrix"
            ],
            columns=criteria
        )


        weighted_df.insert(
            0,
            "Alternative",
            list(alternatives)
        )


        st.dataframe(
            weighted_df,
            width="stretch",
            hide_index=True,
            column_config=number_columns(
                criteria,
                4
            )
        )


    # ========================================================
    # WSM
    # ========================================================

    if method == "WSM":

        st.markdown(
            """
            <div class="formula-box">

            <b>Weighted Sum Method — WSM</b>

            <br><br>

            Qᵢ = Σⱼ wⱼ rᵢⱼ

            <br><br>

            L'alternative avec le score Qᵢ
            le plus élevé est la meilleure.

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # WPM
    # ========================================================

    elif method == "WPM":

        st.markdown(
            """
            <div class="formula-box">

            <b>Weighted Product Method — WPM</b>

            <br><br>

            Qᵢ = ∏ⱼ (rᵢⱼ)<sup>wⱼ</sup>

            <br><br>

            Le score le plus élevé est préféré.

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # WASPAS
    # ========================================================

    elif method == "WASPAS":

        st.markdown(
            f"""
            <div class="formula-box">

            <b>WASPAS</b>

            <br><br>

            Qᵢ =
            λ Qᵢ⁽¹⁾
            +
            (1−λ) Qᵢ⁽²⁾

            <br><br>

            λ = {details["lambda"]:.2f}

            </div>
            """,
            unsafe_allow_html=True
        )


        waspas_df = pd.DataFrame(
            {
                "Alternative":
                    list(alternatives),

                "Q1 — WSM":
                    details["Q1_WSM"],

                "Q2 — WPM":
                    details["Q2_WPM"]
            }
        )


        st.dataframe(
            waspas_df,
            width="stretch",
            hide_index=True,
            column_config=number_columns(
                [
                    "Q1 — WSM",
                    "Q2 — WPM"
                ],
                4
            )
        )


    # ========================================================
    # TOPSIS
    # ========================================================

    elif method == "TOPSIS":

        st.markdown(
            """
            <div class="formula-box">

            <b>TOPSIS</b>

            <br><br>

            RCᵢ =
            Sᵢ⁻ /
            (Sᵢ⁺ + Sᵢ⁻)

            <br><br>

            Plus RCᵢ est élevé,
            meilleure est l'alternative.

            </div>
            """,
            unsafe_allow_html=True
        )


        topsis_df = pd.DataFrame(
            {
                "Alternative":
                    list(alternatives),

                "S+":
                    details["S_plus"],

                "S-":
                    details["S_minus"]
            }
        )


        st.dataframe(
            topsis_df,
            width="stretch",
            hide_index=True,
            column_config=number_columns(
                ["S+", "S-"],
                4
            )
        )


        with st.expander(
            "Voir les solutions idéales TOPSIS"
        ):

            ideal_df = pd.DataFrame(
                {
                    "Critère":
                        criteria,

                    "Idéal positif I+":
                        details[
                            "ideal_positive"
                        ],

                    "Idéal négatif I-":
                        details[
                            "ideal_negative"
                        ]
                }
            )


            st.dataframe(
                ideal_df,
                width="stretch",
                hide_index=True,
                column_config=number_columns(
                    [
                        "Idéal positif I+",
                        "Idéal négatif I-"
                    ],
                    4
                )
            )


    # ========================================================
    # VIKOR
    # ========================================================

    elif method == "VIKOR":

        st.markdown(
            """
            <div class="formula-box">

            <b>VIKOR</b>

            <br><br>

            Sᵢ : utilité globale

            <br>

            Rᵢ : regret individuel

            <br>

            Qᵢ : indice de compromis

            <br><br>

            Plus Qᵢ est faible,
            meilleure est l'alternative.

            </div>
            """,
            unsafe_allow_html=True
        )


        vikor_df = pd.DataFrame(
            {
                "Alternative":
                    list(alternatives),

                "S":
                    details["S"],

                "R":
                    details["R"],

                "Q":
                    details["Q"]
            }
        )


        st.dataframe(
            vikor_df,
            width="stretch",
            hide_index=True,
            column_config=number_columns(
                ["S", "R", "Q"],
                4
            )
        )


        c1, c2 = st.columns(2)


        with c1:

            if details[
                "acceptable_advantage"
            ]:

                st.success(
                    "Avantage acceptable : Oui"
                )

            else:

                st.warning(
                    "Avantage acceptable : Non"
                )


        with c2:

            if details[
                "acceptable_stability"
            ]:

                st.success(
                    "Stabilité acceptable : Oui"
                )

            else:

                st.warning(
                    "Stabilité acceptable : Non"
                )


        with st.expander(
            "Voir les meilleures et mauvaises valeurs"
        ):

            ideal_df = pd.DataFrame(
                {
                    "Critère":
                        criteria,

                    "Meilleure valeur":
                        details["best"],

                    "Mauvaise valeur":
                        details["worst"]
                }
            )


            st.dataframe(
                ideal_df,
                width="stretch",
                hide_index=True,
                column_config=number_columns(
                    [
                        "Meilleure valeur",
                        "Mauvaise valeur"
                    ],
                    4
                )
            )


# ============================================================
# RESULTATS FINAUX
# ============================================================

def show_results(result_df,score_col):

    section(
        "6",
        "Classement final",
        """
        Les alternatives sont classées
        selon la méthode MCDM sélectionnée.
        """
    )

    best = result_df.iloc[0]

    html(f"""
    <div class="winner">
        Meilleure alternative :
        <b>{best["Alternative"]}</b>

        &nbsp;&nbsp; | &nbsp;&nbsp;

        {score_col} =
        <b>{best[score_col]:.4f}</b>
    </div>
    """)

    c1, c2 = st.columns(
        [1.15, 1],
        gap="large"
    )

    with c1:

        st.markdown(
            "### Classement"
        )

        st.dataframe(
            result_df,
            width="stretch",
            hide_index=True,
            column_config={
                score_col:
                    st.column_config.NumberColumn(
                        score_col,
                        format="%.4f"
                    )
            }
        )

    with c2:

        st.markdown(
            "### Scores"
        )

        chart_df = (
            result_df
            .set_index(
                "Alternative"
            )[
                [score_col]
            ]
        )

        st.bar_chart(
            chart_df
        )
