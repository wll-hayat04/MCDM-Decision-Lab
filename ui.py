import streamlit as st
import pandas as pd
import numpy as np

SAATY_OPTIONS = [
    ("1 — Importance égale", 1.0),
    ("2 — Intermédiaire", 2.0),
    ("3 — Importance modérée", 3.0),
    ("4 — Intermédiaire", 4.0),
    ("5 — Importance forte", 5.0),
    ("6 — Intermédiaire", 6.0),
    ("7 — Très forte importance", 7.0),
    ("8 — Intermédiaire", 8.0),
    ("9 — Importance extrême", 9.0),
]

def load_css():
    st.markdown("""
    <style>
      .stApp {
        background:
          radial-gradient(circle at 10% 10%, rgba(37,99,235,.08), transparent 28%),
          linear-gradient(180deg,#f8fafc 0%,#eef2ff 100%);
      }
      .block-container {max-width: 1280px; padding-top: 1.5rem; padding-bottom: 3rem;}
      .hero {
        background: linear-gradient(135deg,#0f172a 0%,#1d4ed8 100%);
        color:#fff; padding:32px 34px; border-radius:24px;
        box-shadow:0 18px 50px rgba(15,23,42,.18); margin-bottom:18px;
      }
      .hero h1 {font-size:2.2rem; margin:0 0 6px 0; letter-spacing:-.04em;}
      .hero p {margin:0; opacity:.92; line-height:1.65; font-size:1rem;}
      .eyebrow {font-size:.78rem; text-transform:uppercase; letter-spacing:.16em; opacity:.75; margin-bottom:8px;}
      .card {
        background:#fff; border:1px solid #e2e8f0; border-radius:18px;
        padding:18px 20px; box-shadow:0 8px 26px rgba(15,23,42,.05); margin-bottom:14px;
      }
      .card-title {font-weight:800; color:#0f172a; font-size:1.05rem; margin-bottom:4px;}
      .card-text {color:#64748b; font-size:.92rem; line-height:1.5;}
      .step {
        width:34px; height:34px; border-radius:10px; display:inline-flex; align-items:center; justify-content:center;
        background:#dbeafe; color:#1d4ed8; font-weight:800; margin-right:10px;
      }
      .winner {
        background:linear-gradient(135deg,#dcfce7,#ecfccb);
        border:1px solid #bbf7d0; color:#14532d; padding:18px 22px; border-radius:18px;
        font-weight:800; box-shadow:0 8px 26px rgba(34,197,94,.10); margin:10px 0 18px;
      }
      .formula-box {
        background:#f8fafc; border-left:4px solid #2563eb; border-radius:10px;
        padding:10px 14px; margin:8px 0 14px;
      }
      section[data-testid="stSidebar"] {background:linear-gradient(180deg,#0f172a,#1e293b);}
      section[data-testid="stSidebar"] * {color:white !important;}
      div.stButton > button {border-radius:14px; font-weight:800; min-height:46px;}
      div[data-testid="stDataFrame"] {border-radius:14px; overflow:hidden;}
    </style>
    """, unsafe_allow_html=True)

def show_header():
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Decision Intelligence • MCDM</div>
      <h1>MCDM Decision Lab</h1>
      <p>
        Sélection multicritère du meilleur modèle de Machine Learning pour la détection de fraude bancaire.
        L'application détaille la pondération, la normalisation et le classement selon les méthodes du cours.
      </p>
    </div>
    """, unsafe_allow_html=True)

def section(step, title, text):
    st.markdown(f"""
    <div class="card">
      <div class="card-title"><span class="step">{step}</span>{title}</div>
      <div class="card-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)

def show_sidebar():
    with st.sidebar:
        st.markdown("## Configuration")
        st.caption("Choisissez la méthode de pondération et la méthode de classement.")

        weighting = st.selectbox(
            "Pondération des critères",
            ["Manual", "AHP", "BWM", "Entropy", "CRITIC"]
        )
        ranking = st.selectbox(
            "Classement des alternatives",
            ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR"]
        )

        lam = 0.5
        vikor_v = 0.5
        if ranking == "WASPAS":
            lam = st.slider("λ — compromis WSM/WPM", 0.0, 1.0, 0.5, 0.05)
        if ranking == "VIKOR":
            vikor_v = st.slider("v — utilité de groupe", 0.0, 1.0, 0.5, 0.05)

        st.markdown("---")
        st.caption("Benefit (+) : à maximiser • Cost (-) : à minimiser")
    return weighting, ranking, lam, vikor_v

def show_decision_matrix(df):
    section("1", "Matrice de décision", "Modifiez les performances des modèles avant l'analyse.")
    return st.data_editor(df, use_container_width=True, hide_index=True, num_rows="fixed", key="decision_editor")

def show_criteria_controls(criteria, defaults):
    section("2", "Sens des critères", "Indiquez si chaque critère est à maximiser (benefit) ou à minimiser (cost).")
    flags = []
    cols = st.columns(3)
    for i, c in enumerate(criteria):
        with cols[i % 3]:
            default = defaults.get(c, "Maximiser (+)")
            choice = st.selectbox(
                c,
                ["Maximiser (+)", "Minimiser (-)"],
                index=0 if default.startswith("Max") else 1,
                key=f"direction_{c}"
            )
            flags.append(choice.startswith("Max"))
    return flags

def show_manual_weights(criteria):
    section("3", "Pondération manuelle", "Saisissez l'importance de chaque critère. Les poids seront automatiquement normalisés.")
    values = []
    cols = st.columns(3)
    for i, c in enumerate(criteria):
        with cols[i % 3]:
            values.append(st.number_input(f"Poids — {c}", min_value=0.0, value=1.0, step=0.1, key=f"w_{c}"))
    return values

def show_ahp_input(criteria):
    section("3", "Pondération AHP", "Comparez les critères deux à deux avec l'échelle de Saaty (1 à 9).")
    n = len(criteria)
    A = np.ones((n, n), dtype=float)

    st.caption("Choisissez l'importance du critère de gauche par rapport au critère de droite.")
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2, c3 = st.columns([1.25, 1, 1.25])
            with c1:
                st.markdown(f"**{criteria[i]}**")
            with c2:
                labels = [x[0] for x in SAATY_OPTIONS]
                label = st.selectbox("vs", labels, index=0, key=f"ahp_{i}_{j}", label_visibility="collapsed")
                val = dict(SAATY_OPTIONS)[label]
            with c3:
                st.markdown(f"**{criteria[j]}**")
            A[i, j] = val
            A[j, i] = 1.0 / val

    return A

def show_bwm_input(criteria):
    section("3", "Pondération BWM", "Choisissez le critère Best, le critère Worst, puis renseignez les deux vecteurs de comparaison.")
    best = st.selectbox("Critère le plus important (Best)", criteria, key="bwm_best")
    worst_choices = [c for c in criteria if c != best]
    worst = st.selectbox("Critère le moins important (Worst)", worst_choices, key="bwm_worst")
    b = criteria.index(best)
    w = criteria.index(worst)

    st.markdown("#### Best → Others")
    BO = []
    cols = st.columns(3)
    for i, c in enumerate(criteria):
        with cols[i % 3]:
            if i == b:
                val = 1
                st.number_input(c, min_value=1, max_value=1, value=1, disabled=True, key=f"bo_{i}")
            else:
                val = st.slider(c, 1, 9, 3, key=f"bo_{i}")
            BO.append(val)

    st.markdown("#### Others → Worst")
    OW = []
    cols2 = st.columns(3)
    for i, c in enumerate(criteria):
        with cols2[i % 3]:
            if i == w:
                val = 1
                st.number_input(c, min_value=1, max_value=1, value=1, disabled=True, key=f"ow_{i}")
            else:
                val = st.slider(c, 1, 9, 3, key=f"ow_{i}")
            OW.append(val)

    return b, w, BO, OW

def show_weights_result(criteria, weights, details):
    section("4", "Poids des critères", "Les poids obtenus sont normalisés et leur somme vaut 1.")
    df = pd.DataFrame({"Critère": criteria, "Poids": weights})
    st.dataframe(df.style.format({"Poids": "{:.4f}"}), use_container_width=True, hide_index=True)

    method = details.get("method")
    if method == "AHP":
        c1, c2, c3 = st.columns(3)
        c1.metric("λmax", f"{details['lambda_max']:.4f}")
        c2.metric("CI", f"{details['CI']:.4f}")
        c3.metric("CR", f"{details['CR']:.4f}")
        if details["consistent"]:
            st.success("CR < 0,10 : les jugements AHP sont cohérents.")
        else:
            st.warning("CR ≥ 0,10 : les comparaisons AHP doivent être révisées.")
        with st.expander("Voir la matrice AHP normalisée"):
            st.dataframe(pd.DataFrame(details["normalized_pairwise"], index=criteria, columns=criteria).style.format("{:.4f}"))
    elif method == "BWM":
        st.metric("ξ*", f"{details['xi']:.4f}")
    elif method == "Entropy":
        with st.expander("Voir les entropies"):
            e = pd.DataFrame({"Critère": criteria, "Entropie E_j": details["entropy"], "Diversification 1-E_j": details["diversification"]})
            st.dataframe(e.style.format(precision=4), use_container_width=True, hide_index=True)
    elif method == "CRITIC":
        with st.expander("Voir les indices CRITIC"):
            cdf = pd.DataFrame({"Critère": criteria, "σ_j": details["sigma"], "C_j": details["C"]})
            st.dataframe(cdf.style.format(precision=4), use_container_width=True, hide_index=True)
        with st.expander("Voir la matrice de corrélation"):
            st.dataframe(pd.DataFrame(details["correlation"], index=criteria, columns=criteria).style.format("{:.3f}"))

def show_ranking_details(criteria, alternatives, details):
    section("5", "Normalisation et calcul", "Cette partie affiche les matrices intermédiaires propres à la méthode sélectionnée.")
    method = details["method"]

    if "normalized_matrix" in details:
        st.markdown("#### Matrice normalisée")
        ndf = pd.DataFrame(details["normalized_matrix"], columns=criteria)
        ndf.insert(0, "Alternative", list(alternatives))
        st.dataframe(ndf.style.format(precision=4), use_container_width=True, hide_index=True)

    if "weighted_matrix" in details:
        st.markdown("#### Matrice normalisée pondérée")
        wdf = pd.DataFrame(details["weighted_matrix"], columns=criteria)
        wdf.insert(0, "Alternative", list(alternatives))
        st.dataframe(wdf.style.format(precision=4), use_container_width=True, hide_index=True)

    if method == "WSM":
        st.markdown('<div class="formula-box">WSM : Qᵢ = Σⱼ wⱼ rᵢⱼ</div>', unsafe_allow_html=True)
    elif method == "WPM":
        st.markdown('<div class="formula-box">WPM : Qᵢ = ∏ⱼ (rᵢⱼ)<sup>wⱼ</sup></div>', unsafe_allow_html=True)
    elif method == "WASPAS":
        st.markdown(f'<div class="formula-box">WASPAS : Qᵢ = λQᵢ⁽¹⁾ + (1−λ)Qᵢ⁽²⁾, λ={details["lambda"]:.2f}</div>', unsafe_allow_html=True)
        waspas_df = pd.DataFrame({
            "Alternative": list(alternatives),
            "Q1 (WSM)": details["Q1_WSM"],
            "Q2 (WPM)": details["Q2_WPM"],
        })
        st.dataframe(waspas_df.style.format(precision=4), use_container_width=True, hide_index=True)
    elif method == "TOPSIS":
        st.markdown('<div class="formula-box">TOPSIS : RCᵢ = Sᵢ⁻ / (Sᵢ⁺ + Sᵢ⁻)</div>', unsafe_allow_html=True)
        d = pd.DataFrame({
            "Alternative": list(alternatives),
            "S+": details["S_plus"],
            "S-": details["S_minus"],
        })
        st.dataframe(d.style.format(precision=4), use_container_width=True, hide_index=True)
    elif method == "VIKOR":
        st.markdown('<div class="formula-box">VIKOR : plus Qᵢ est faible, meilleure est l’alternative.</div>', unsafe_allow_html=True)
        vdf = pd.DataFrame({
            "Alternative": list(alternatives),
            "S": details["S"],
            "R": details["R"],
            "Q": details["Q"],
        })
        st.dataframe(vdf.style.format(precision=4), use_container_width=True, hide_index=True)
        st.write(
            f"Avantage acceptable : **{'Oui' if details['acceptable_advantage'] else 'Non'}** • "
            f"Stabilité acceptable : **{'Oui' if details['acceptable_stability'] else 'Non'}**"
        )

def show_results(result_df, score_col):
    section("6", "Classement final", "Le classement est construit à partir du score final de la méthode choisie.")
    best = result_df.iloc[0]
    st.markdown(
        f'<div class="winner">🏆 Meilleure alternative : <b>{best["Alternative"]}</b> '
        f'— {score_col} = <b>{best[score_col]:.4f}</b></div>',
        unsafe_allow_html=True
    )
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        st.dataframe(result_df.style.format({score_col: "{:.4f}"}), use_container_width=True, hide_index=True)
    with c2:
        chart = result_df.set_index("Alternative")[[score_col]]
        st.bar_chart(chart)
