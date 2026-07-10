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
    p = np.asarray(p, dtype=float)
    # el broadcast copia el test a todos: no depende de p
    b_bcast = n_te * (d + 1) * BYTES
    # el scatter reparte el train: a cada proceso le toca n_tr/p, o sea DECRECE con p
    b_scatter = (n_tr / np.maximum(p, 1)) * (d + 1) * BYTES
    # el gather: cada uno de los p procesos manda sus (n_te x k) candidatos (distancia
    # + etiqueta) y la raiz los recibe TODOS. O sea el volumen CRECE con p. Aca esta
    # la clave de por que la eficiencia se cae: el maestro se vuelve el cuello.
    b_gather = p * n_te * k * 2 * BYTES
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


def karp_flatt(s, p):
    """Fraccion serie experimental e = (1/S - 1/p) / (1 - 1/p).

    Si e se mantiene CONSTANTE al crecer p, lo que frena es una parte
    intrinsecamente secuencial. Si e CRECE con p, lo que frena es el overhead
    (comunicacion), que es nuestro caso.
    """
    s = np.asarray(s, float)
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = (1.0 / s - 1.0 / p) / (1.0 - 1.0 / p)
    return np.where(p > 1, e, np.nan)


# ---------------------------------------------------------------------------
# Iso-eficiencia: "que tan rapido tiene que crecer N para que E no se caiga"
# ---------------------------------------------------------------------------
def constante_computo(ts_ref, n_tr, n_te, d):
    """El Ts medido dividido por el trabajo, o sea los segundos por unidad de
    trabajo. Con esto puedo evaluar Ts(N) para cualquier N sin volver a medir."""
    return ts_ref / (n_tr * n_te * d)


def eficiencia_modelo(p, N, frac_te, d, k, c, alpha, beta):
    """E(N,p) segun el modelo: E = Ts / (p*Tp), con Tp = Ts/p + T_comm."""
    n_te = frac_te * N
    n_tr = N - n_te
    ts = c * n_tr * n_te * d                       # el computo va como N^2
    tp = ts / p + t_comm(p, n_tr, n_te, d, k, alpha, beta)
    return ts / (p * tp)


def iso_eficiencia_N(p_arr, e_obj, frac_te, d, k, c, alpha, beta,
                     n_lo=1e3, n_hi=1e10, tol=1e-4):
    """Para cada p, el N que hace E(N,p) = e_obj. Es la funcion de iso-eficiencia.

    La deduccion es la que pide el profe: se escribe E = Ts/(Ts + To) con
    To = p*T_comm(p) el overhead, se la iguala a una constante y se despeja N.
    Como E crece de forma monotona con N (mas computo por la misma comunicacion),
    lo resuelvo por biseccion en vez de despejar la cuadratica a mano.

    Asintotica: en el termino de comunicacion manda el gather, que va como
    beta*p*N, asi que To ~ p*log2(p) * beta*p*N = beta*N*p^2*log2(p). Igualando
    con Ts ~ c*N^2 queda N ~ (beta/c) * p^2 * log2(p): nuestro KNN necesita que
    N crezca como p^2*log p, MAS rapido que el p*log p del Random Forest que el
    profe derivo en clase (S13). Justamente porque alla el maestro sincroniza un
    resultado chico y aca recibe p*k*n_te candidatos.
    """
    out = []
    for p in np.atleast_1d(p_arr):
        if p <= 1:
            out.append(np.nan)
            continue
        lo, hi = float(n_lo), float(n_hi)
        if eficiencia_modelo(p, hi, frac_te, d, k, c, alpha, beta) < e_obj:
            out.append(np.nan)                     # ni con N enorme se alcanza
            continue
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if eficiencia_modelo(p, mid, frac_te, d, k, c, alpha, beta) < e_obj:
                lo = mid
            else:
                hi = mid
            if (hi - lo) / hi < tol:
                break
        out.append(0.5 * (lo + hi))
    return np.array(out)
