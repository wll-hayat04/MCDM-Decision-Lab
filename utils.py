import numpy as np
import pandas as pd


# ============================================================
# NETTOYAGE DE LA MATRICE
# ============================================================

def prepare_decision_matrix(df, alternative_col="Model"):
    """
    Nettoie la matrice provenant de st.data_editor.

    Une ligne est considérée comme incomplète si :
    - le nom de l'alternative est vide
    - au moins un critère est vide
    - une valeur de critère n'est pas numérique

    Les lignes incomplètes sont ignorées pour le calcul
    au lieu de faire planter l'application.

    Returns
    -------
    cleaned_df : DataFrame
        Matrice contenant uniquement les lignes complètes.

    ignored_rows : int
        Nombre de lignes ignorées.
    """

    if df is None:
        raise ValueError(
            "Aucune matrice de décision n'a été fournie."
        )

    if alternative_col not in df.columns:
        raise ValueError(
            f"La colonne '{alternative_col}' est absente."
        )

    cleaned = df.copy()

    criteria = [
        col
        for col in cleaned.columns
        if col != alternative_col
    ]

    if len(criteria) < 2:
        raise ValueError(
            "Il faut au moins deux critères."
        )

    # Nettoyage du nom des alternatives
    cleaned[alternative_col] = (
        cleaned[alternative_col]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    # Conversion des critères en nombres
    for criterion in criteria:

        cleaned[criterion] = pd.to_numeric(
            cleaned[criterion],
            errors="coerce"
        )

    # Détection des lignes incomplètes
    incomplete_mask = (
        cleaned[alternative_col].isna()
        |
        cleaned[criteria]
        .isna()
        .any(axis=1)
    )

    ignored_rows = int(
        incomplete_mask.sum()
    )

    # On ne garde que les lignes complètes
    cleaned = (
        cleaned.loc[~incomplete_mask]
        .reset_index(drop=True)
    )

    return cleaned, ignored_rows


# ============================================================
# NORMALISATION DES POIDS
# ============================================================

def normalize_weights(weights):

    weights = np.asarray(
        weights,
        dtype=float
    )

    if np.any(weights < 0):

        raise ValueError(
            "Les poids ne peuvent pas être négatifs."
        )

    total = weights.sum()

    if total <= 0:

        raise ValueError(
            "La somme des poids doit être "
            "strictement supérieure à 0."
        )

    return weights / total


# ============================================================
# VALIDATION MATRICE
# ============================================================

def validate_decision_matrix(
    df,
    alternative_col="Model"
):

    if alternative_col not in df.columns:

        raise ValueError(
            f"La colonne '{alternative_col}' est absente."
        )

    criteria = [
        col
        for col in df.columns
        if col != alternative_col
    ]

    if len(criteria) < 2:

        raise ValueError(
            "Il faut au moins deux critères."
        )

    if len(df) < 2:

        raise ValueError(
            "Il faut au moins deux alternatives "
            "complètes pour effectuer une analyse MCDM."
        )

    numeric = (
        df[criteria]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    if numeric.isna().any().any():

        raise ValueError(
            "Toutes les performances des alternatives "
            "doivent être numériques."
        )

    return (
        criteria,
        numeric.to_numpy(dtype=float)
    )


# ============================================================
# CLASSEMENT
# ============================================================

def ranking_dataframe(
    alternatives,
    scores,
    higher_is_better=True,
    score_name="Score"
):

    scores = np.asarray(
        scores,
        dtype=float
    )

    if np.any(~np.isfinite(scores)):

        raise ValueError(
            "Le calcul a produit un score invalide "
            "(NaN ou infini). Vérifiez vos données."
        )

    result = pd.DataFrame(
        {
            "Alternative":
                list(alternatives),

            score_name:
                scores
        }
    )

    result = (
        result
        .sort_values(
            score_name,
            ascending=not higher_is_better
        )
        .reset_index(drop=True)
    )

    result.insert(
        0,
        "Rang",
        np.arange(
            1,
            len(result) + 1
        )
    )

    return result
