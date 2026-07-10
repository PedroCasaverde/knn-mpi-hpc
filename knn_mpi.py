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
gather (vecinos locales). Cronometro cada colectiva por separado con MPI.Wtime
(entregable c) y guardo tambien la desviacion estandar de las repeticiones.

Dos variantes de las colectivas (--impl):
  v2 : las minusculas de mpi4py (bcast/scatter/gather). Serializan con pickle,
       asi que el costo lo domina la serializacion del payload y no la red.
  v3 : las de buffer (Bcast/Scatterv/Gather) sobre arrays numpy contiguos.
       No hay pickle: se mide la comunicacion de verdad. Es la mejora.

Hibrido MPI+OpenMP: el computo pasa por un GEMM de BLAS, asi que la cantidad de
hilos sale de OMP_NUM_THREADS. Se lanza con p rangos x T hilos y se anota T en
el csv para poder comparar (p*T = cores usados).

Uso (en Khipu):
    module load gnu12 openmpi4 python3/3.11.11
    mpiexec -n 8 python knn_mpi.py --n 7188 --k 7 --impl v3 --csv results/strong.csv
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
    return np.ascontiguousarray(dk, dtype=np.float64), np.ascontiguousarray(lk, dtype=np.int64)


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


def contar_filas(n_tr, p):
    """Cuantas filas de train le tocan a cada rango. np.array_split reparte
    los sobrantes en los primeros bloques, asi que los bloques son desiguales
    y por eso el Scatterv necesita counts y displs a mano."""
    return [len(c) for c in np.array_split(np.arange(n_tr), p)]


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
    ap.add_argument("--impl", choices=("v2", "v3"), default="v2",
                    help="v2 = colectivas con pickle; v3 = colectivas con buffer")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    hilos = int(os.environ.get("OMP_NUM_THREADS", "1"))
    nodos = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))

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
        meta = (n_tr, n_te, d, n_classes)
    else:
        Xtr = ytr = Xte = yte = None
        meta = None

    meta = comm.bcast(meta, root=0)
    n_tr, n_te, d, n_classes = meta
    filas = contar_filas(n_tr, p)                     # igual en todos los rangos (es deterministico)
    mis_filas = filas[rank]

    # ---- preparacion (fuera del cronometro: es setup, no comunicacion) ----
    if args.impl == "v2":
        if rank == 0:
            payload = list(zip(np.array_split(Xtr, p), np.array_split(ytr, p)))
        else:
            payload = None
    else:
        # buffers contiguos, se reservan una sola vez. Esa es justamente la ventaja
        # de las colectivas con buffer: no se serializa ni se realoca en cada rep.
        if rank == 0:
            Xtr_c = np.ascontiguousarray(Xtr, dtype=np.float64)
            ytr_c = np.ascontiguousarray(ytr, dtype=np.int64)
            Xte_c = np.ascontiguousarray(Xte, dtype=np.float64)
            yte_c = np.ascontiguousarray(yte, dtype=np.int64)
            cnt_x = [f * d for f in filas]
            dsp_x = np.concatenate(([0], np.cumsum(cnt_x)[:-1])).astype(int)
            dsp_y = np.concatenate(([0], np.cumsum(filas)[:-1])).astype(int)
            env_x = [Xtr_c, cnt_x, dsp_x, MPI.DOUBLE]
            env_y = [ytr_c, filas, dsp_y, MPI.INT64_T]
        else:
            Xte_c = np.empty((n_te, d), dtype=np.float64)
            yte_c = np.empty(n_te, dtype=np.int64)
            env_x = env_y = None
        myX = np.empty((mis_filas, d), dtype=np.float64)
        myY = np.empty(mis_filas, dtype=np.int64)
        recvD = np.empty((p, n_te, args.k), dtype=np.float64) if rank == 0 else None
        recvL = np.empty((p, n_te, args.k), dtype=np.int64) if rank == 0 else None

    t_b_l, t_s_l, t_g_l, t_comp_l, t_comm_l, t_tot_l = [], [], [], [], [], []
    t_tmax_l, t_cmax_l, t_mmax_l = [], [], []      # tiempos de pared (max entre rangos)
    acc = 0.0

    for rep in range(args.reps + args.warmup):
        comm.Barrier()
        t0 = MPI.Wtime()

        if args.impl == "v2":
            # comunicacion: mando el test a todos (bcast) y reparto el train (scatter)
            Xte_b = comm.bcast(Xte, root=0)
            yte_b = comm.bcast(yte, root=0)
            t1 = MPI.Wtime()
            myX_r, myY_r = comm.scatter(payload, root=0)
            t2 = MPI.Wtime()
            dk, lk = local_knn(Xte_b, myX_r, myY_r, args.k)
            t3 = MPI.Wtime()
            gd = comm.gather(dk, root=0)
            gl = comm.gather(lk, root=0)
            t4 = MPI.Wtime()
        else:
            comm.Bcast(Xte_c, root=0)
            comm.Bcast(yte_c, root=0)
            t1 = MPI.Wtime()
            comm.Scatterv(env_x, myX, root=0)
            comm.Scatterv(env_y, myY, root=0)
            t2 = MPI.Wtime()
            dk, lk = local_knn(Xte_c, myX, myY, args.k)
            t3 = MPI.Wtime()
            comm.Gather(dk, recvD, root=0)
            comm.Gather(lk, recvL, root=0)
            t4 = MPI.Wtime()
            gd = list(recvD) if rank == 0 else None
            gl = list(recvL) if rank == 0 else None
            yte_b = yte_c

        if rank == 0:
            preds = merge_vote(gd, gl, args.k, n_classes)
            acc = float((preds == yte_b).mean())

        # Dos relojes, y hay que tener cuidado de no mezclarlos:
        #  - el del MAESTRO (rank 0): es el que pidio el profe y, sobre todo, sus fases
        #    SUMAN exacto (t_bcast + t_scatter + t_comp + t_gather = t_total). Si en vez
        #    de esto uno reduce cada fase con MAX por separado y despues las suma, los
        #    maximos caen en rangos distintos y el total no cierra.
        #  - el de PARED (max entre rangos): el proceso mas lento manda. Lo guardo aparte
        #    como referencia, pero no lo uso para descomponer.
        t_t_max = comm.reduce(t4 - t0, op=MPI.MAX, root=0)
        t_c_max = comm.reduce(t3 - t2, op=MPI.MAX, root=0)
        t_m_max = comm.reduce((t1 - t0) + (t2 - t1) + (t4 - t3), op=MPI.MAX, root=0)
        if rank == 0 and rep >= args.warmup:
            t_b_l.append(t1 - t0)
            t_s_l.append(t2 - t1)
            t_comp_l.append(t3 - t2)
            t_g_l.append(t4 - t3)
            t_comm_l.append((t1 - t0) + (t2 - t1) + (t4 - t3))
            t_tot_l.append(t4 - t0)
            t_tmax_l.append(t_t_max); t_cmax_l.append(t_c_max); t_mmax_l.append(t_m_max)

    # ---- el proceso 0 imprime los resultados y los guarda al csv ----
    if rank == 0:
        t_bcast = float(np.mean(t_b_l))
        t_scatter = float(np.mean(t_s_l))
        t_gather = float(np.mean(t_g_l))
        t_comp = float(np.mean(t_comp_l))
        t_comm = float(np.mean(t_comm_l))
        t_total = float(np.mean(t_tot_l))
        sd_comp = float(np.std(t_comp_l, ddof=1)) if len(t_comp_l) > 1 else 0.0
        sd_comm = float(np.std(t_comm_l, ddof=1)) if len(t_comm_l) > 1 else 0.0
        sd_tot = float(np.std(t_tot_l, ddof=1)) if len(t_tot_l) > 1 else 0.0
        t_total_max = float(np.mean(t_tmax_l))
        t_comp_max = float(np.mean(t_cmax_l))
        t_comm_max = float(np.mean(t_mmax_l))

        fl = flops_region(n_tr, n_te, d)
        gflops_total = fl / t_total / 1e9
        gflops_comp = fl / t_comp / 1e9

        print(f"[MPI-{args.impl}] p={p:>3} hilos={hilos} nodos={nodos} n={args.n} "
              f"n_tr={n_tr} n_te={n_te} d={d} k={args.k}")
        print(f"      t_total={t_total:.6f}s (sd {sd_tot:.6f})  t_comp={t_comp:.6f}s  "
              f"t_comm={t_comm:.6f}s (sd {sd_comm:.6f})  acc={acc:.4f}")
        print(f"      bcast={t_bcast:.6f}s  scatter={t_scatter:.6f}s  gather={t_gather:.6f}s"
              f"   [suman {t_bcast + t_scatter + t_comp + t_gather:.6f}s = t_total]")
        print(f"      pared(max): total={t_total_max:.6f}s comp={t_comp_max:.6f}s comm={t_comm_max:.6f}s")
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
                                "gflops_total", "gflops_comp",
                                "variant", "threads", "nodes", "reps",
                                "t_bcast", "t_scatter", "t_gather",
                                "t_total_std", "t_comp_std", "t_comm_std",
                                "t_total_max", "t_comp_max", "t_comm_max"])
                w.writerow(["mpi", args.n, p, args.k, d, n_tr, n_te,
                            t_total, t_comp, t_comm, acc, fl,
                            gflops_total, gflops_comp,
                            args.impl, hilos, nodos, args.reps,
                            t_bcast, t_scatter, t_gather,
                            sd_tot, sd_comp, sd_comm,
                            t_total_max, t_comp_max, t_comm_max])


if __name__ == "__main__":
    main()
