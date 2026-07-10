#!/bin/bash
# ---------------------------------------------------------------------------
# run_khipu.slurm - barrido de benchmarks KNN-MPI en el cluster Khipu (UTEC).
#
#   sbatch run_khipu.slurm
#
# Corre, en orden:
#   0) VALIDACION   : v3 (buffers) tiene que dar el mismo accuracy que v2 (pickle)
#                     con p que NO divide a n_tr (bloques desiguales). Si no, aborta.
#   1) FUERTE       : n fijo, p = 1..32, con v2 y v3  -> speedup / eficiencia / FLOPs
#   2) TAMAÑO       : p = 8 fijo, n creciente         -> validar teoria vs n
#   3) DEBIL        : carga por proceso constante, TRES cargas (1000/2000/4000)
#                     -> tres curvas, como pide el profe
#   4) HIBRIDO      : p rangos x T hilos con p*T = 32 -> MPI+OpenMP (BLAS)
#   5) ENERGIA      : RAPL antes/despues, mismo metodo que el codigo de la S6
#
# MULTI-NODO: se piden 2 nodos y se usa `--map-by node` para que los rangos se
# repartan en round-robin entre nodos. Asi TODA colectiva con p>=2 cruza la red y
# el termino de latencia (alpha * log2 p) se ve de verdad; con un solo nodo la
# comunicacion es un memcpy y la curva sale plana o cayendo.
# ---------------------------------------------------------------------------
#SBATCH --job-name=knn_mpi
#SBATCH --partition=standard
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:30:00
#SBATCH --output=logs/knn_mpi_%j.out
#SBATCH --error=logs/knn_mpi_%j.err
# OJO: la particion por defecto es `debug`, y su QOS (p-debug) tiene MaxWall=01:00:00,
# asi que un job de 1h30 se queda colgado en PD con QOSMaxWallDurationPerJobLimit.
# `standard` (n003-n006) no tiene ese tope; es la que usa el profe en sus scripts.

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
REPS=15
WARMUP=2
# reparte los rangos entre nodos (round-robin) para forzar comunicacion por red
SPREAD="--map-by node"

echo "================ NODOS: $SLURM_JOB_NODELIST | $(date) ================"
echo "python: $PY"; $PY --version

# ---------------------------------------------------------------------------
echo ""
echo "==== [0] VALIDACION: v3 debe dar el mismo accuracy que v2 ===="
# p=3 y p=7 no dividen a n_tr, asi que los bloques del Scatterv quedan desiguales:
# es justo el caso donde un error de counts/displs corrompe el train.
rm -f results/_val.csv
for pv in 3 7; do
    for impl in v2 v3; do
        mpiexec $SPREAD -n $pv $PY knn_mpi.py --n 1797 --k $K --reps 2 --warmup 1 \
            --impl $impl --csv results/_val.csv
    done
done
$PY - <<'PYVAL'
import csv, sys
from collections import defaultdict
acc = defaultdict(dict)
with open("results/_val.csv") as f:
    for r in csv.DictReader(f):
        acc[int(r["p"])][r["variant"]] = float(r["acc"])
malo = False
for p, d in sorted(acc.items()):
    ok = abs(d.get("v2", -1) - d.get("v3", -2)) < 1e-12
    print(f"  p={p}: acc v2={d.get('v2')} v3={d.get('v3')} -> {'OK' if ok else 'DIFIEREN'}")
    malo |= not ok
if malo:
    sys.exit("[ABORT] v3 no reproduce a v2: revisar counts/displs del Scatterv")
print("  validacion OK: las colectivas con buffer dan el mismo resultado")
PYVAL

# ---------------------------------------------------------------------------
echo ""
echo "==== [1] Escalabilidad FUERTE (n=14376 fijo, multi-nodo, v2 y v3) ===="
rm -f results/strong.csv
for impl in v2 v3; do
    for p in $PROCS; do
        mpiexec $SPREAD -n $p $PY knn_mpi.py --n 14376 --k $K \
            --reps $REPS --warmup $WARMUP --impl $impl --csv results/strong.csv
    done
done

# ---------------------------------------------------------------------------
echo ""
echo "==== [2] TAMAÑO del problema (p=8 fijo, v3) ===="
rm -f results/size.csv
for n in 1797 3594 7188 14376 28752; do
    mpiexec $SPREAD -n 8 $PY knn_mpi.py --n $n --k $K \
        --reps $REPS --warmup $WARMUP --impl v3 --csv results/size.csv
done

# ---------------------------------------------------------------------------
echo ""
echo "==== [3] Escalabilidad DEBIL (tres cargas por proceso) ===="
# carga por proceso constante: cada rango calcula n_te x (n_tr/p) = CARGA x CARGA.
# Tres cargas => tres curvas, que es lo que pide el profe (fuerte = una sola curva).
rm -f results/weak.csv
for carga in 1000 2000 4000; do
    for p in $PROCS; do
        ntr=$((carga * p))
        mpiexec $SPREAD -n $p $PY knn_mpi.py --n-tr $ntr --n-te $carga --k $K \
            --reps $REPS --warmup $WARMUP --impl v3 --csv results/weak.csv
    done
done

# ---------------------------------------------------------------------------
echo ""
echo "==== [4] HIBRIDO MPI + OpenMP (p rangos x T hilos, p*T = 32) ===="
# el computo es un GEMM de BLAS, asi que los hilos salen de OMP_NUM_THREADS.
# --bind-to none evita que OpenMPI clave cada rango a un solo core y ahogue a los hilos.
rm -f results/hybrid.csv
for combo in "32 1" "16 2" "8 4" "4 8"; do
    set -- $combo; ph=$1; th=$2
    OMP_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th MKL_NUM_THREADS=$th \
    mpiexec $SPREAD -n $ph --bind-to none $PY knn_mpi.py --n 14376 --k $K \
        --reps $REPS --warmup $WARMUP --impl v3 --csv results/hybrid.csv
done

# ---------------------------------------------------------------------------
echo ""
echo "==== [5] ENERGIA (RAPL, mismo metodo que el codigo de la S6) ===="
# RAPL cuenta la energia del socket del nodo donde corre el script, asi que este
# bloque se mide en UN nodo para que el numero tenga sentido.
RAPL0="/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
RAPL1="/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj"
RANGE0="/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj"
ENER="results/energia.csv"
rm -f "$ENER"
echo "p,tiempo_s,energia_J,potencia_W,J_por_consulta" > "$ENER"

leer_rapl() {  # suma los sockets disponibles, en microjoules
    local tot=0
    [ -f "$RAPL0" ] && tot=$(( tot + $(cat "$RAPL0") ))
    [ -f "$RAPL1" ] && tot=$(( tot + $(cat "$RAPL1") ))
    echo "$tot"
}

if [ -f "$RAPL0" ]; then
    NTE=3594   # las consultas del caso n=14376
    WRAP=0; [ -f "$RANGE0" ] && WRAP=$(cat "$RANGE0")
    # OJO: RAPL solo cuenta la energia del nodo donde corre este script, asi que
    # este bloque se fuerza a UN solo nodo con `-N $p` (por eso llega hasta p=16,
    # que es --ntasks-per-node). Con p=32 el trabajo se iria a dos nodos y la
    # energia quedaria a la mitad.
    for p in 1 2 4 8 16; do
        E0=$(leer_rapl); T0=$(date +%s.%N)
        mpiexec -n $p -N $p $PY knn_mpi.py --n 14376 --k $K --reps 5 --warmup 1 --impl v3 >/dev/null
        T1=$(date +%s.%N); E1=$(leer_rapl)
        # el contador da la vuelta al llegar a max_energy_range_uj; si pasa, se suma el rango
        if [ "$E1" -lt "$E0" ]; then E1=$(( E1 + 2 * WRAP )); fi
        TIME=$(echo "$T1 - $T0" | bc -l)
        EJ=$(echo "($E1 - $E0)/1000000" | bc -l)
        PW=$(echo "$EJ / $TIME" | bc -l)
        JQ=$(echo "$EJ / $NTE" | bc -l)
        echo "$p,$TIME,$EJ,$PW,$JQ" >> "$ENER"
        echo "  p=$p  t=${TIME}s  E=${EJ}J  P=${PW}W  J/consulta=${JQ}"
    done
else
    echo "  [aviso] no hay RAPL en este nodo; se omite el bloque de energia"
fi

echo ""
echo "DONE_KNN_SWEEP $(date)"
