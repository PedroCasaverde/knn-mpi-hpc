"""
knn_mpi.py - KNN paralelo con mpi4py (entregable a).

La idea del DAG es bien simple: se le manda el test a todos con bcast y se
reparte el train entre los procesos (scatter, mas o menos n_tr/p a cada uno).
Cada proceso saca sus k vecinos locales contra el bloque que le toco, y al
final el proceso 0 junta todo con gather y decide la clase por voto de mayoria
(los k mejores de los k mejores).

        bcast(test) + scatter(train)
                |
        cada proceso -> k vecinos locales
                |
        gather -> proceso 0 -> voto

Las tres directivas que pide el enunciado: bcast (test), scatter (train) y
gather (vecinos locales). Cronometro computo y comunicacion por separado con
MPI.Wtime (entregable c).

Uso (en Khipu):
    module load gnu12 openmpi4 py3-mpi4py
    mpiexec -n 8 python knn_mpi.py --n 7188 --k 7 --csv results/mpi.csv
"""
import argparse
import csv
import os

import numpy as np
from mpi4py import MPI

# jalo todo del secuencial para no repetir codigo (mismo dataset y misma distancia)
from knn_digits_sec import (
    build_dataset,
    build_split,
    train_test_split,
    euclidean_block,
    flops_region,
)


def local_knn(Xte, Xtr, ytr, k):
    """Los k vecinos locales de cada test contra el bloque de train que le toco
    a este proceso.

    Devuelve (dist_k, lab_k) con forma (n_te, k). Si al proceso le cayo un bloque
    con menos de k filas, relleno con inf y etiqueta -1 asi el merge del final
    las bota solas.
    """
    n_te = Xte.shape[0]
    m = Xtr.shape[0]
    if m == 0:
        return (np.full((n_te, k), np.inf),
                np.full((n_te, k), -1, dtype=np.int64))
    dist = euclidean_block(Xte, Xtr)                 # matriz (n_te, m)
    kk = min(k, m)
    idx = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
    dk = np.take_along_axis(dist, idx, axis=1)
    lk = ytr[idx].astype(np.int64)
    if kk < k:
        pad = k - kk
        dk = np.hstack([dk, np.full((n_te, pad), np.inf)])
        lk = np.hstack([lk, np.full((n_te, pad), -1, dtype=np.int64)])
    return dk, lk


def merge_vote(gathered_d, gathered_l, k, n_classes):
    """Junta los p*k candidatos que llegaron por gather y vota por mayoria.
    Esto corre solo en la raiz."""
    D = np.hstack(gathered_d)                         # (n_te, p*k)
    L = np.hstack(gathered_l)                         # (n_te, p*k)
    # de todos los candidatos me quedo con los k mejores (los k mejores de los k mejores)
    sel = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    Lk = np.take_along_axis(L, sel, axis=1)           # (n_te, k)
    preds = np.empty(D.shape[0], dtype=np.int64)
    for i in range(D.shape[0]):
        lab = Lk[i]
        lab = lab[lab >= 0]
        preds[i] = 0 if lab.size == 0 else np.bincount(lab, minlength=n_classes).argmax()
    return preds


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    p = comm.Get_size()

    ap = argparse.ArgumentParser(description="KNN paralelo con mpi4py")
    ap.add_argument("--n", type=int, default=1797, help="tamaño total del dataset")
    ap.add_argument("--n-tr", type=int, default=None, help="tamaño train explicito (weak scaling)")
    ap.add_argument("--n-te", type=int, default=None, help="tamaño test explicito (weak scaling)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--reps", type=int, default=10, help="repeticiones promediadas")
    ap.add_argument("--warmup", type=int, default=2, help="corridas iniciales descartadas")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    # ---- solo el proceso 0 arma los datos y los parte para repartir ----
    if rank == 0:
        if args.n_tr is not None and args.n_te is not None:
            Xtr, ytr, Xte, yte = build_split(args.n_tr, args.n_te, seed=args.seed)
        else:
            X, y = build_dataset(args.n, seed=args.seed)
            Xtr, ytr, Xte, yte = train_test_split(X, y, args.test_frac, seed=args.seed)
        n_tr, d = Xtr.shape
        n_te = Xte.shape[0]
        if args.n_tr is not None and args.n_te is not None:
            args.n = n_tr + n_te                     # el n total real que va al csv
        n_classes = int(ytr.max()) + 1
        chunksX = np.array_split(Xtr, p)             # parto el train en p bloques (~n_tr/p) para el scatter
        chunksY = np.array_split(ytr, p)
        payload = list(zip(chunksX, chunksY))
        meta = (n_tr, n_te, d, n_classes)
    else:
        Xte = yte = payload = None
        meta = None

    meta = comm.bcast(meta, root=0)
    n_tr, n_te, d, n_classes = meta

    t_comp_list, t_comm_list, t_tot_list = [], [], []
    acc = 0.0

    for rep in range(args.reps + args.warmup):
        comm.Barrier()
        t0 = MPI.Wtime()
        # comunicacion: mando el test a todos (bcast) y reparto el train (scatter)
        Xte_b = comm.bcast(Xte, root=0)
        yte_b = comm.bcast(yte, root=0)
        myX, myY = comm.scatter(payload, root=0)
        t1 = MPI.Wtime()
        # computo: aca cada proceso saca sus distancias y sus k vecinos locales
        dk, lk = local_knn(Xte_b, myX, myY, args.k)
        t2 = MPI.Wtime()
        # comunicacion: la raiz junta los vecinos locales con gather
        gd = comm.gather(dk, root=0)
        gl = comm.gather(lk, root=0)
        t3 = MPI.Wtime()

        if rank == 0:
            preds = merge_vote(gd, gl, args.k, n_classes)
            acc = float((preds == yte).mean())

        # me quedo con el max entre procesos, que es el tiempo de pared real (el mas lento manda)
        comp = comm.reduce(t2 - t1, op=MPI.MAX, root=0)
        commt = comm.reduce((t1 - t0) + (t3 - t2), op=MPI.MAX, root=0)
        total = comm.reduce(t3 - t0, op=MPI.MAX, root=0)
        if rank == 0 and rep >= args.warmup:
            t_comp_list.append(comp)
            t_comm_list.append(commt)
            t_tot_list.append(total)

    # ---- el proceso 0 imprime los resultados y los guarda al csv ----
    if rank == 0:
        t_comp = float(np.mean(t_comp_list))
        t_comm = float(np.mean(t_comm_list))
        t_total = float(np.mean(t_tot_list))
        fl = flops_region(n_tr, n_te, d)
        gflops_total = fl / t_total / 1e9
        gflops_comp = fl / t_comp / 1e9

        print(f"[MPI] p={p:>3} n={args.n} n_tr={n_tr} n_te={n_te} d={d} k={args.k}")
        print(f"      t_total={t_total:.6f}s  t_comp={t_comp:.6f}s  "
              f"t_comm={t_comm:.6f}s  acc={acc:.4f}")
        print(f"      FLOPs={fl:.3e}  {gflops_total:.3f} GFLOP/s (sostenido)  "
              f"{gflops_comp:.3f} GFLOP/s (computo)")

        if args.csv:
            os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
            new = not os.path.exists(args.csv)
            with open(args.csv, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["impl", "n", "p", "k", "d", "n_tr", "n_te",
                                "t_total", "t_comp", "t_comm", "acc", "flops",
                                "gflops_total", "gflops_comp"])
                w.writerow(["mpi", args.n, p, args.k, d, n_tr, n_te,
                            t_total, t_comp, t_comm, acc, fl,
                            gflops_total, gflops_comp])


if __name__ == "__main__":
    main()
