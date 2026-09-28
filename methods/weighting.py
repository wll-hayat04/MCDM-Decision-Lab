import numpy as np
from scipy.optimize import linprog
from .normalization import normalize_critic, normalize_entropy
from utils import normalize_weights

RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.56
}

def manual_weights(values):
    w = normalize_weights(values)
    return w, {"method": "Manual"}

def ahp_weights(pairwise_matrix):
    A = np.asarray(pairwise_matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("La matrice AHP doit être carrée.")
    n = A.shape[0]

    col_sums = A.sum(axis=0)
    norm = A / col_sums
    w = norm.mean(axis=1)
    w = normalize_weights(w)

    # Méthode du cours: lambda_max = somme(s_j * w_j)
    lambda_max = float(np.dot(col_sums, w))
    ci = 0.0 if n <= 1 else (lambda_max - n) / (n - 1)
    ri = RI_TABLE.get(n, 1.56)
    cr = 0.0 if ri == 0 else ci / ri

    details = {
        "method": "AHP",
        "normalized_pairwise": norm,
        "column_sums": col_sums,
        "lambda_max": lambda_max,
        "CI": ci,
        "RI": ri,
        "CR": cr,
        "consistent": cr < 0.10
    }
    return w, details

def bwm_weights(best_index, worst_index, best_to_others, others_to_worst):
    """
    BWM linéaire du cours.
    Variables: w_1..w_n, xi
    Min xi
    s.c. |w_B - a_Bj w_j| <= xi
         |w_j - a_jW w_W| <= xi
         sum w_j = 1, w_j >= 0, xi >= 0
    """
    a_B = np.asarray(best_to_others, dtype=float)
    a_W = np.asarray(others_to_worst, dtype=float)
    n = len(a_B)
    if len(a_W) != n:
        raise ValueError("Les vecteurs BWM doivent avoir la même taille.")
    if best_index == worst_index:
        raise ValueError("Best et Worst doivent être différents.")

    # x = [w1,...,wn,xi]
    c = np.zeros(n + 1)
    c[-1] = 1.0

    A_ub, b_ub = [], []

    for j in range(n):
        # wB - aBj*wj - xi <= 0
        row = np.zeros(n + 1)
        row[best_index] = 1
        row[j] -= a_B[j]
        row[-1] = -1
        A_ub.append(row); b_ub.append(0)

        # -wB + aBj*wj - xi <= 0
        row2 = np.zeros(n + 1)
        row2[best_index] = -1
        row2[j] += a_B[j]
        row2[-1] = -1
        A_ub.append(row2); b_ub.append(0)

        # wj - ajW*wW - xi <= 0
        row3 = np.zeros(n + 1)
        row3[j] = 1
        row3[worst_index] -= a_W[j]
        row3[-1] = -1
        A_ub.append(row3); b_ub.append(0)

        # -wj + ajW*wW - xi <= 0
        row4 = np.zeros(n + 1)
        row4[j] = -1
        row4[worst_index] += a_W[j]
        row4[-1] = -1
        A_ub.append(row4); b_ub.append(0)

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * (n + 1)

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )
    if not res.success:
        raise ValueError("L'optimisation BWM n'a pas convergé: " + res.message)

    w = normalize_weights(res.x[:n])
    xi = float(res.x[-1])
    return w, {"method": "BWM", "xi": xi, "success": res.success}

def entropy_weights(matrix):
    x = normalize_entropy(matrix)
    m, n = x.shape
    # p_ij = d_ij / sum_i d_ij
    col_sums = x.sum(axis=0)
    p = np.zeros_like(x, dtype=float)
    for j in range(n):
        if col_sums[j] <= 1e-12:
            p[:, j] = 1.0 / m
        else:
            p[:, j] = x[:, j] / col_sums[j]

    k = 1.0 / np.log(m)
    # Convention 0*log(0)=0, sans évaluer log(0).
    plogp = np.zeros_like(p, dtype=float)
    mask = p > 0
    plogp[mask] = p[mask] * np.log(p[mask])
    E = -k * np.sum(plogp, axis=0)
    d = 1.0 - E
    if d.sum() <= 1e-12:
        w = np.ones(n) / n
    else:
        w = d / d.sum()

    details = {
        "method": "Entropy",
        "normalized_matrix": x,
        "p_matrix": p,
        "entropy": E,
        "diversification": d
    }
    return w, details

def critic_weights(matrix, benefit_flags):
    r = normalize_critic(matrix, benefit_flags)
    # Ecart-type échantillonnal comme dans le cours (m-1)
    sigma = np.std(r, axis=0, ddof=1)
    corr = np.corrcoef(r, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    independence = np.sum(1.0 - np.abs(corr), axis=1)
    C = sigma * independence
    if C.sum() <= 1e-12:
        w = np.ones(r.shape[1]) / r.shape[1]
    else:
        w = C / C.sum()

    details = {
        "method": "CRITIC",
        "normalized_matrix": r,
        "sigma": sigma,
        "correlation": corr,
        "C": C
    }
    return w, details
