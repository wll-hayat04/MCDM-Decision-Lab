import numpy as np
from .normalization import (
    normalize_wsm,
    normalize_topsis,
    ideal_solutions
)

EPS = 1e-12

def wsm(matrix, weights, benefit_flags):
    r = normalize_wsm(matrix, benefit_flags)
    weighted = r * weights
    scores = weighted.sum(axis=1)
    details = {
        "method": "WSM",
        "normalized_matrix": r,
        "weighted_matrix": weighted,
        "higher_is_better": True
    }
    return scores, details

def wpm(matrix, weights, benefit_flags):
    r = normalize_wsm(matrix, benefit_flags)
    safe = np.clip(r, EPS, None)
    scores = np.prod(safe ** weights, axis=1)
    details = {
        "method": "WPM",
        "normalized_matrix": r,
        "higher_is_better": True
    }
    return scores, details

def waspas(matrix, weights, benefit_flags, lam=0.5):
    r = normalize_wsm(matrix, benefit_flags)
    q1 = np.sum(r * weights, axis=1)
    safe = np.clip(r, EPS, None)
    q2 = np.prod(safe ** weights, axis=1)
    q = lam * q1 + (1.0 - lam) * q2
    details = {
        "method": "WASPAS",
        "normalized_matrix": r,
        "Q1_WSM": q1,
        "Q2_WPM": q2,
        "lambda": lam,
        "higher_is_better": True
    }
    return q, details

def topsis(matrix, weights, benefit_flags):
    r = normalize_topsis(matrix)
    v = r * weights

    positive = np.zeros(v.shape[1], dtype=float)
    negative = np.zeros(v.shape[1], dtype=float)

    for j, benefit in enumerate(benefit_flags):
        if benefit:
            positive[j] = np.max(v[:, j])
            negative[j] = np.min(v[:, j])
        else:
            positive[j] = np.min(v[:, j])
            negative[j] = np.max(v[:, j])

    s_plus = np.sqrt(np.sum((v - positive) ** 2, axis=1))
    s_minus = np.sqrt(np.sum((v - negative) ** 2, axis=1))
    closeness = s_minus / np.clip(s_plus + s_minus, EPS, None)

    details = {
        "method": "TOPSIS",
        "normalized_matrix": r,
        "weighted_matrix": v,
        "ideal_positive": positive,
        "ideal_negative": negative,
        "S_plus": s_plus,
        "S_minus": s_minus,
        "higher_is_better": True
    }
    return closeness, details

def vikor(matrix, weights, benefit_flags, v=0.5):
    x = np.asarray(matrix, dtype=float)
    best, worst = ideal_solutions(x, benefit_flags)

    gap = np.zeros_like(x, dtype=float)
    for j in range(x.shape[1]):
        denom = best[j] - worst[j]
        if abs(denom) < EPS:
            gap[:, j] = 0.0
        else:
            gap[:, j] = weights[j] * (best[j] - x[:, j]) / denom

    S = np.sum(gap, axis=1)
    R = np.max(gap, axis=1)

    S_star, S_minus = np.min(S), np.max(S)
    R_star, R_minus = np.min(R), np.max(R)

    s_term = np.zeros_like(S) if abs(S_minus - S_star) < EPS else (S - S_star) / (S_minus - S_star)
    r_term = np.zeros_like(R) if abs(R_minus - R_star) < EPS else (R - R_star) / (R_minus - R_star)

    Q = v * s_term + (1.0 - v) * r_term

    order = np.argsort(Q)
    m = len(Q)
    dq = 1.0 / (m - 1) if m > 1 else 0.0
    advantage = True if m < 2 else (Q[order[1]] - Q[order[0]] >= dq)
    stable = (order[0] == int(np.argmin(S))) or (order[0] == int(np.argmin(R)))

    details = {
        "method": "VIKOR",
        "best": best,
        "worst": worst,
        "gap_matrix": gap,
        "S": S,
        "R": R,
        "Q": Q,
        "v": v,
        "acceptable_advantage": advantage,
        "acceptable_stability": stable,
        "higher_is_better": False
    }
    return Q, details
