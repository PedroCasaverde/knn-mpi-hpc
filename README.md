# KNN paralelo con MPI - Applied High Performance Computing (UTEC, 2026-I)

Paralelización del clasificador **K-Nearest Neighbors** sobre el dataset
`load_digits` (dígitos manuscritos 8×8, 64 atributos) usando **mpi4py** con las
directivas colectivas `comm.bcast`, `comm.scatter` y `comm.gather`, siguiendo el
DAG del enunciado. Las mediciones reales se ejecutan en el clúster **Khipu**
(UTEC) y se contrastan con un **modelo teórico α-β**.

> Proyecto del curso *Applied High Performance Computing* (Prof. José Fiestas).
> Entrega parcial: informe en `informe/` + presentación.

## DAG / esquema de paralelización

```
              [parámetros iniciales]
        bcast(test) /          \ scatter(train, n_tr/p)
                   v            v
   [proceso i: distancia euclidiana(test, su bloque de train)
               -> sus k vecinos locales]        <- cómputo O(n_tr·n_te/p)
                   \            /
                    gather(k locales)
                         v
        [maestro: merge de p·k candidatos -> voto por mayoría]
```

- `comm.bcast` → replica el conjunto de **test** en todos los procesos.
- `comm.scatter` → reparte el **train** en bloques de ≈ `n_tr/p`.
- `comm.gather` → recolecta los `k` vecinos locales en el maestro, que hace el
  merge global y el voto por mayoría.

## Estructura

| Ruta | Qué es |
|------|--------|
| `knn_digits_sec.py` | Baseline **secuencial** (el código que se paraleliza). |
| `knn_mpi.py` | Versión **MPI** (bcast/scatter/gather + cronómetro por fase). |
| `digits.npz` | Dataset `load_digits` empaquetado (portable, sin sklearn en Khipu). |
| `run_khipu.slurm` | Job SLURM: barridos fuerte / tamaño / débil. |
| `tools/khipu_run.py` | Orquestador SSH: sube, lanza, espera y descarga resultados. |
| `src/model.py` | Modelo teórico **α-β** y su calibración a los datos. |
| `src/make_figures.py` | Genera las figuras y métricas (medido + modelo). |
| `results/khipu/` | CSVs y logs descargados del clúster. |
| `results/figs/` | Figuras de resultados (speedup, eficiencia, FLOP/s, escalabilidad). |

> El **informe** (PDF) y la **presentación** son entregables de documento que se
> suben a Canvas; no forman parte de este repositorio de código. Las figuras se
> regeneran desde los CSV con `python src/make_figures.py`.

## Reproducir

### Local (baseline + regenerar dataset)
```bash
pip install -r requirements.txt
python knn_digits_sec.py --n 1797 --k 7        # KNN secuencial
```

### En Khipu (mediciones MPI reales)
```bash
# crea el venv (numpy+mpi4py contra OpenMPI), sube todo, lanza SLURM y descarga CSVs
export KHIPU_PASSWORD=...      # o usa --password-file
python tools/khipu_run.py
```
El job ejecuta tres barridos con `module load gnu12 openmpi4 python3/3.11.11`:
1. **Escalabilidad fuerte** - `n` fijo, `p = 1,2,4,8,16,32`.
2. **Tamaño del problema** - `p = 8`, `n` creciente.
3. **Escalabilidad débil** - `n = 1797·p` (carga por proceso constante).

### Figuras
```bash
python src/make_figures.py     # lee results/khipu/*.csv -> results/figs/*.pdf
```

## Notas de método
- El escalado es limpio porque cada rango MPI usa **1 hilo** (`OMP/BLAS=1`).
- Se promedian ≥10 repeticiones y se descartan las primeras (warmup).
- La paralelización **no degrada el accuracy** (idéntico al secuencial, verificado).
- `n` se escala hacia abajo por submuestreo y hacia arriba por replicado con
  ruido gaussiano leve (documentado en el informe).
