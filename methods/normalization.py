import numpy as np

EPS = 1e-12

def ideal_solutions(matrix, benefit_flags):
    x = np.asarray(matrix, dtype=float)
    best, worst = [], []
    for j, benefit in enumerate(benefit_flags):
        col = x[:, j]
        if benefit:
            best.append(np.max(col))
            worst.append(np.min(col))
        else:
            best.append(np.min(col))
            worst.append(np.max(col))
    return np.array(best, dtype=float), np.array(worst, dtype=float)

def normalize_wsm(matrix, benefit_flags):
    """
    Normalisation du cours pour WSM/WPM/WASPAS:
    benefit: x_ij / max(x_j)
    cost:    min(x_j) / x_ij
    """
    x = np.asarray(matrix, dtype=float)
    r = np.zeros_like(x, dtype=float)
    for j, benefit in enumerate(benefit_flags):
        col = x[:, j]
        if benefit:
            denom = np.max(col)
            if abs(denom) < EPS:
                r[:, j] = 1.0
            else:
                r[:, j] = col / denom
        else:
            minv = np.min(col)
            if np.any(np.abs(col) < EPS):
                # Fallback robuste pour éviter division par zéro.
                maxv = np.max(col)
                span = maxv - minv
                r[:, j] = 1.0 if abs(span) < EPS else (maxv - col) / span
            else:
                r[:, j] = minv / col
    return r

def normalize_topsis(matrix):
    """
    Normalisation vectorielle TOPSIS:
    r_ij = x_ij / sqrt(sum_i x_ij^2)
    """
    x = np.asarray(matrix, dtype=float)
    denom = np.sqrt(np.sum(x ** 2, axis=0))
    denom = np.where(np.abs(denom) < EPS, 1.0, denom)
    return x / denom

def normalize_critic(matrix, benefit_flags):
    """
    Normalisation min-max CRITIC
    benefit: (x-min)/(max-min)
    cost:    (max-x)/(max-min)
    """
    x = np.asarray(matrix, dtype=float)
    r = np.zeros_like(x, dtype=float)
    for j, benefit in enumerate(benefit_flags):
        col = x[:, j]
        mn, mx = np.min(col), np.max(col)
        span = mx - mn
        if abs(span) < EPS:
            r[:, j] = 0.0
        elif benefit:
            r[:, j] = (col - mn) / span
        else:
            r[:, j] = (mx - col) / span
    return r

def normalize_entropy(matrix):
    """
    Transformation positive pour la méthode d'entropie.
    Les valeurs sont ramenées dans [0,1] par min-max, puis légèrement
    décalées afin d'éviter log(0).
    """
    x = np.asarray(matrix, dtype=float)
    r = np.zeros_like(x, dtype=float)
    for j in range(x.shape[1]):
        col = x[:, j]
        mn, mx = np.min(col), np.max(col)
        span = mx - mn
        if abs(span) < EPS:
            r[:, j] = 1.0
        else:
            r[:, j] = (col - mn) / span
    return r
