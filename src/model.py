"""Modelo teorico alpha-beta del KNN-MPI: el costo final es comunicacion mas computo.

Idea del curso: Tp = Ts/p (computo ~ 1/p) + log2(p)*(n_col*alpha + beta*bytes).
Para calibrar saco Ts del run secuencial y (alpha,beta) por minimos cuadrados,
o sea normalizo la teoria a los datos que medi.
"""
import numpy as np

N_COL = 3              # las 3 colectivas que cuentan: bcast, scatter, gather
BYTES = 8             # float64 / int64


def bytes_total(p, n_tr, n_te, d, k):
    """Cuenta los bytes que viajan en las 3 colectivas con p procesos."""
    # el broadcast copia el test a todos
    b_bcast = n_te * (d + 1) * BYTES
    # el scatter reparte el train: a cada proceso le toca n_tr/p
    b_scatter = (n_tr / np.maximum(p, 1)) * (d + 1) * BYTES
    # el gather junta de todos a uno los k vecinos (distancia + etiqueta)
    b_gather = n_te * k * 2 * BYTES
    return b_bcast + b_scatter + b_gather


def t_comm(p, n_tr, n_te, d, k, alpha, beta):
    """Comunicacion teorica: latencia (alpha) mas ancho de banda (beta), proporcional al log P."""
    p = np.asarray(p, dtype=float)
    # con p=1 no hay comunicacion, asi que el log queda en 0
    logp = np.where(p > 1, np.log2(p), 0.0)
    return logp * (N_COL * alpha + beta * bytes_total(p, n_tr, n_te, d, k))


def t_comp(p, ts):
    """Computo teorico: Ts repartido entre los p procesos (proporcional a 1/p)."""
    return ts / np.asarray(p, dtype=float)


def t_total(p, ts, n_tr, n_te, d, k, alpha, beta):
    return t_comp(p, ts) + t_comm(p, n_tr, n_te, d, k, alpha, beta)


def calibrate_comm(p_arr, tcomm_arr, n_tr, n_te, d, k):
    """Saca alpha y beta por minimos cuadrados: normaliza el modelo al T_comm medido.

    Es lineal en (alpha,beta), asi que armo la matriz A y resuelvo con lstsq.
    """
    p_arr = np.asarray(p_arr, float)
    tcomm_arr = np.asarray(tcomm_arr, float)
    # con p=1 el log es 0 y no aporta nada, lo dejo fuera del ajuste
    mask = p_arr > 1
    logp = np.log2(p_arr[mask])
    bt = bytes_total(p_arr[mask], n_tr, n_te, d, k)
    A = np.column_stack([logp * N_COL, logp * bt])
    sol, *_ = np.linalg.lstsq(A, tcomm_arr[mask], rcond=None)
    alpha, beta = float(sol[0]), float(sol[1])
    # alpha y beta son constantes fisicas, no pueden salir negativas
    return max(alpha, 1e-9), max(beta, 1e-12)


def optimal_p(ts, n_tr, n_te, d, k, alpha, beta, p_max=64):
    """Busca el punto de cruce: el p que minimiza el tiempo total."""
    grid = np.arange(1, p_max + 1)
    tt = t_total(grid, ts, n_tr, n_te, d, k, alpha, beta)
    return int(grid[int(np.argmin(tt))])


def speedup(t1, tp):
    return np.asarray(t1, float) / np.asarray(tp, float)


def efficiency(t1, tp, p):
    return speedup(t1, tp) / np.asarray(p, float)
