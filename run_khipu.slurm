#!/bin/bash
# ---------------------------------------------------------------------------
# run_khipu.slurm - barrido de benchmarks KNN-MPI en el cluster Khipu (UTEC).
#
#   sbatch run_khipu.slurm
#
# Mide tres escenarios (todos con reps promediadas y warmup descartado):
#   1) Escalabilidad FUERTE : n fijo, p = 1,2,4,8,16,32   -> speedup/eficiencia/FLOPs
#   2) TAMAÑO del problema  : p = 8 fijo, n creciente     -> validar teoria vs n
#   3) Escalabilidad DEBIL  : n = 1797*p (n/p constante)  -> eficiencia debil
#
# Cada rango MPI corre 1 hilo (OMP/BLAS = 1) para que el escalado sea limpio.
# ---------------------------------------------------------------------------
#SBATCH --job-name=knn_mpi
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=00:40:00
#SBATCH --output=logs/knn_mpi_%j.out
#SBATCH --error=logs/knn_mpi_%j.err

set -e
# python3/3.11.11 vive en /opt/ohpc (compartido) y es consistente login<->compute;
# es el mismo con el que se creo el venv (evita el python 3.6 del nodo de computo).
module load gnu12 openmpi4 python3/3.11.11

PY="$HOME/knnmpi-venv/bin/python"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p results logs

PROCS="1 2 4 8 16 32"
K=7
REPS=10
WARMUP=2

echo "================ NODO: $(hostname) | $(date) ================"
echo "python: $PY"; $PY --version

echo "==== [1] Escalabilidad FUERTE (n=14376 fijo) ===="
rm -f results/strong.csv
for p in $PROCS; do
    mpiexec -n $p $PY knn_mpi.py --n 14376 --k $K --reps $REPS --warmup $WARMUP --csv results/strong.csv
done

echo "==== [2] TAMAÑO del problema (p=8 fijo) ===="
rm -f results/size.csv
for n in 1797 3594 7188 14376 28752; do
    mpiexec -n 8 $PY knn_mpi.py --n $n --k $K --reps $REPS --warmup $WARMUP --csv results/size.csv
done

echo "==== [3] Escalabilidad DEBIL (n_te fijo, n_tr = 2000*p) ===="
# Carga por proceso constante: cada uno calcula n_te x (n_tr/p) = 2000 x 2000.
rm -f results/weak.csv
NTE=2000
for p in $PROCS; do
    ntr=$((2000 * p))
    mpiexec -n $p $PY knn_mpi.py --n-tr $ntr --n-te $NTE --k $K --reps $REPS --warmup $WARMUP --csv results/weak.csv
done

echo "DONE_KNN_SWEEP $(date)"
