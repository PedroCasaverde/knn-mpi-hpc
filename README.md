# KNN paralelo con MPI - Applied High Performance Computing (UTEC, 2026-I)

Paralelización del clasificador **K-Nearest Neighbors** sobre el dataset
`load_digits` (dígitos manuscritos 8×8, 64 atributos) usando **mpi4py** con las
directivas colectivas `comm.bcast`, `comm.scatter` y `comm.gather`, siguiendo el
DAG del enunciado. Las mediciones reales se ejecutan en el clúster **Khipu**
(UTEC) y se contrastan con un **modelo teórico α-β**.

> Proyecto del curso *Applied High Performance Computing* (Prof. José Fiestas).
> Entrega final: informe + presentación (van a Canvas, no a este repo).

## Resultado principal

Con `n = 14376` en **dos nodos** de Khipu, el tiempo baja hasta un mínimo en
**`p = 16`** (0.163 s) y con `p = 32` **vuelve a subir** (0.433 s): el speedup pasa
de 3.45× a 1.29×. La causa está medida: la comunicación pasa de pesar 0.3 % a
**84 %** del total, y la fracción serie de Karp-Flatt **crece** (0.22 → 0.77), lo que
descarta un cuello secuencial fijo y apunta al overhead.

El motivo de fondo es que las tres colectivas no son simétricas: el `bcast` manda un
volumen fijo, el `scatter` manda cada vez menos, pero el **`gather` manda cada vez más**
(el maestro recibe `p·k·n_te` candidatos). De ahí sale la función de iso-eficiencia:
para sostener `E` constante hace falta **`N ~ p²·log₂p`**, peor que el `N ~ p·log₂p`
de un Random Forest. El ajuste numérico sobre los datos da `p^2.01`.

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
El job corre en la partición `standard` (la `debug` tiene `MaxWall` de 1 h y el job se
queda en `PD`), con `module load gnu12 openmpi4 python3/3.11.11`, y ejecuta:

0. **Validación** - la `v3` tiene que reproducir el accuracy de la `v2` con `p = 3` y
   `p = 7` (que no dividen a `n_tr`, así que los bloques del `Scatterv` quedan
   desiguales). Si no coinciden, **aborta**.
1. **Escalabilidad fuerte** - `n` fijo, `p = 1..32`, con `v2` y `v3`.
2. **Tamaño del problema** - `p = 8`, `n` creciente.
3. **Escalabilidad débil** - tres cargas por proceso (1000/2000/4000) → **tres curvas**.
4. **Híbrido MPI+OpenMP** - `p × hilos = 32`.
5. **Energía** - contadores **RAPL** del socket, leídos antes y después.

### Figuras
```bash
python src/make_figures.py     # lee results/khipu/*.csv -> results/figs/*.pdf
```

## Versiones (tags de Git)

| Tag | Qué trae |
|-----|----------|
| `v0-secuencial` | Baseline con la distancia a mano. |
| `v1-scatter-gather` | Versión MPI funcionando con el DAG. |
| `v2-optimizada` | Cronómetro por fase, promediado y descarte de warmup. |
| `v3-buffers` | Colectivas con buffer (`Bcast`/`Scatterv`/`Gather`), sin `pickle`. |

## Notas de método
- **Se mide en dos nodos** con `--map-by node`, para que toda colectiva con `p ≥ 2`
  cruce la red. Con un solo nodo la comunicación es un `memcpy` y la curva no crece:
  el ajuste de `β` se va al piso y el modelo α-β deja de decir nada.
- Los tiempos que se reportan son los del **maestro** (rango 0), cuyas fases suman
  exactamente el total. Sumar los `MAX` de cada fase por separado **no** da `t_total`,
  porque los máximos caen en rangos distintos.
- El escalado es limpio porque cada rango MPI usa **1 hilo** (`OMP/BLAS=1`), salvo en
  el barrido híbrido.
- Se promedian **15 repeticiones** y se descartan las 2 primeras (warmup). Los CSV
  guardan la desviación estándar.
- La paralelización **no degrada el accuracy** (idéntico al secuencial, verificado).
- `n` se escala hacia abajo por submuestreo y hacia arriba por replicado con
  ruido gaussiano leve. Por eso el accuracy sale 1.0000: **se mide costo, no precisión**.
- `results/parcial/` guarda las mediciones mono-nodo de la entrega parcial, para poder
  comparar el antes y el después.
