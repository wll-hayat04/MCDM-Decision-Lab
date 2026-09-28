import numpy as np


def normalize_wsm(matrix, benefit_flags):
    """
    Normalisation pour WSM.

    Critère bénéfice (+): r_ij = x_ij / max(x_j)
    Critère coût (-):     r_ij = min(x_j) / x_ij
    """
    x = np.asarray(matrix, dtype=float)
    r = np.zeros_like(x, dtype=float)

    for j in range(x.shape[1]):
        col = x[:, j]

        if benefit_flags[j]:
            max_val = np.max(col)
            if max_val == 0:
                r[:, j] = 0
            else:
                r[:, j] = col / max_val
        else:
            min_val = np.min(col)
            if np.any(col == 0):
                raise ValueError(
                    "Un critère à minimiser contient une valeur nulle. "
                    "La normalisation min(x)/x serait impossible."
                )
            r[:, j] = min_val / col

    return r


def wsm_scores(normalized_matrix, weights):
    """Calcule les scores WSM Q_i = somme_j w_j * r_ij."""
    r = np.asarray(normalized_matrix, dtype=float)
    w = np.asarray(weights, dtype=float)
    return r @ w
