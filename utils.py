import numpy as np
import pandas as pd

def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("Les poids doivent être positifs ou nuls.")
    s = weights.sum()
    if s <= 0:
        raise ValueError("La somme des poids doit être strictement positive.")
    return weights / s

def validate_decision_matrix(df, alternative_col="Model"):
    if alternative_col not in df.columns:
        raise ValueError(f"La colonne '{alternative_col}' est absente.")
    criteria = [c for c in df.columns if c != alternative_col]
    if len(criteria) < 2:
        raise ValueError("Il faut au moins deux critères.")
    numeric = df[criteria].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Toutes les valeurs des critères doivent être numériques.")
    if len(df) < 2:
        raise ValueError("Il faut au moins deux alternatives.")
    return criteria, numeric.to_numpy(dtype=float)

def ranking_dataframe(alternatives, scores, higher_is_better=True, score_name="Score"):
    out = pd.DataFrame({
        "Alternative": list(alternatives),
        score_name: np.asarray(scores, dtype=float)
    })
    out = out.sort_values(score_name, ascending=not higher_is_better).reset_index(drop=True)
    out.insert(0, "Rang", np.arange(1, len(out) + 1))
    return out
