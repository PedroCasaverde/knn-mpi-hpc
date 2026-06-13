"""
KNN secuencial sobre load_digits. Es el baseline del proyecto, la version
contra la que voy a comparar el speedup de la version MPI.
La parte pesada es la distancia euclidiana (la calculo a mano), que despues
reparto entre los procesos.
"""
import argparse
import csv
import os
import time

import numpy as np


# dataset load_digits: 1797 imagenes 8x8, o sea 64 atributos, 10 clases
def _load_base():
    """Carga la data. Primero busca el digits.npz porque en Khipu no siempre
    tengo sklearn; si no esta el npz recien uso sklearn.load_digits."""
    here = os.path.dirname(os.path.abspath(__file__))
    npz = os.path.join(here, "digits.npz")
    if os.path.exists(npz):
        dat = np.load(npz)
        return dat["X"].astype(np.float64), dat["y"]
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    return X.astype(np.float64), y


def build_dataset(n, seed=42):
    """Arma un dataset de tamaño n. Como load_digits solo trae 1797, si quiero
    escalar el problema hacia arriba replico la data con un ruido gaussiano chico
    en los pixeles (asi no estoy inventando muestras de la nada, lo explico en
    el informe). Si n es menor agarro un subconjunto al azar."""
    X, y = _load_base()
    base = X.shape[0]
    rng = np.random.default_rng(seed)

    if n <= base:
        idx = rng.permutation(base)[:n]
        return X[idx], y[idx]

    reps = int(np.ceil(n / base))
    Xs, ys = [X], [y]
    for _ in range(1, reps):
        noise = rng.normal(0.0, 0.5, size=X.shape)        # ruidito
        Xs.append(np.clip(X + noise, 0.0, 16.0))
        ys.append(y)
    Xb = np.vstack(Xs)[:n]
    yb = np.concatenate(ys)[:n]
    return Xb, yb


def train_test_split(X, y, test_frac=0.25, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_te = max(1, int(round(test_frac * n)))
    te, tr = perm[:n_te], perm[n_te:]
    return X[tr], y[tr], X[te], y[te]


def build_split(n_tr, n_te, seed=42):
    """Igual que el split pero con los tamaños puestos a mano. Lo necesito para
    la escalabilidad debil: dejo n_te fijo y hago n_tr proporcional a p, asi la
    carga por proceso queda constante."""
    need = n_tr + n_te
    Xp, yp = build_dataset(need, seed=seed)
    return Xp[:n_tr], yp[:n_tr], Xp[n_tr:n_tr + n_te], yp[n_tr:n_tr + n_te]


# esta es la parte que despues paralelizo: distancia euclidiana + k vecinos
def euclidean_block(Xte, Xtr):
    """Devuelve la matriz de distancias euclidianas, de tamaño (n_te x n_tr).

    En vez de hacer la resta par por par uso la identidad
    ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, asi el producto Xte @ Xtr.T se lo dejo
    a BLAS (GEMM) y no tengo que armar el tensor (n_te, n_tr, d) que se come la
    memoria. Para contar FLOPs igual uso la formula directa: por cada par hay
    d restas + d cuadrados + (d-1) sumas + 1 raiz, mas o menos 3d.
    """
    te2 = np.einsum("ij,ij->i", Xte, Xte)[:, None]        # ||a||^2
    tr2 = np.einsum("ij,ij->i", Xtr, Xtr)[None, :]        # ||b||^2
    d2 = te2 + tr2 - 2.0 * (Xte @ Xtr.T)
    np.maximum(d2, 0.0, out=d2)                           # a veces sale negativo chiquito por redondeo
    return np.sqrt(d2)


def knn_predict(Xtr, ytr, Xte, k):
    """KNN secuencial, predice por voto de mayoria."""
    dist = euclidean_block(Xte, Xtr)                      # (n_te, n_tr)
    knn_idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]   # me quedo con los k mejores
    knn_labels = ytr[knn_idx]                             # (n_te, k)
    n_classes = int(ytr.max()) + 1
    preds = np.empty(Xte.shape[0], dtype=ytr.dtype)
    for i in range(Xte.shape[0]):
        preds[i] = np.bincount(knn_labels[i], minlength=n_classes).argmax()
    return preds


def flops_region(n_tr, n_te, d):
    """Cuenta los FLOPs de la zona que paralelizo (la distancia, ~3d por par)."""
    return float(n_te) * float(n_tr) * (3.0 * d)


# main
def main():
    ap = argparse.ArgumentParser(description="KNN secuencial sobre load_digits")
    ap.add_argument("--n", type=int, default=1797, help="tamaño total del dataset")
    ap.add_argument("--k", type=int, default=5, help="numero de vecinos")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--reps", type=int, default=5, help="repeticiones (se promedian)")
    ap.add_argument("--warmup", type=int, default=1, help="corridas iniciales descartadas")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=None, help="archivo CSV donde anexar la fila")
    args = ap.parse_args()

    X, y = build_dataset(args.n, seed=args.seed)
    Xtr, ytr, Xte, yte = train_test_split(X, y, args.test_frac, seed=args.seed)
    n_tr, d = Xtr.shape
    n_te = Xte.shape[0]

    times = []
    acc = 0.0
    for r in range(args.reps + args.warmup):
        t0 = time.perf_counter()
        preds = knn_predict(Xtr, ytr, Xte, args.k)
        t1 = time.perf_counter()
        if r >= args.warmup:
            times.append(t1 - t0)
        acc = float((preds == yte).mean())

    t_comp = float(np.mean(times))
    fl = flops_region(n_tr, n_te, d)
    gflops = fl / t_comp / 1e9

    print(f"[SEC] n={args.n} n_tr={n_tr} n_te={n_te} d={d} k={args.k}")
    print(f"      t_comp={t_comp:.6f}s  acc={acc:.4f}  "
          f"FLOPs={fl:.3e}  {gflops:.3f} GFLOP/s  (reps={args.reps})")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["impl", "n", "p", "k", "d", "n_tr", "n_te",
                            "t_total", "t_comp", "t_comm", "acc", "flops",
                            "gflops_total", "gflops_comp"])
            w.writerow(["sec", args.n, 1, args.k, d, n_tr, n_te,
                        t_comp, t_comp, 0.0, acc, fl, gflops, gflops])


if __name__ == "__main__":
    main()
