"""
Saca las figuras del informe usando los CSV que dejo el benchmark en Khipu.
Encima dibujo la curva teorica alpha-beta para comparar teoria vs experimento.

    python src/make_figures.py

Los CSV vienen del barrido multi-nodo (2 nodos, --map-by node), asi que la
comunicacion es de RED y no un memcpy. La variante que se grafica por defecto
es la v3 (colectivas con buffer); la v2 (pickle) se usa solo para la figura
comparativa de la mejora.
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "khipu")
FIGS = os.path.join(ROOT, "results", "figs")     # las figuras cuentan como resultado, asi que al repo
INFORME = os.path.join(ROOT, "informe")          # los .tex chiquitos los dejo local nomas
os.makedirs(FIGS, exist_ok=True)
os.makedirs(INFORME, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (6.4, 4.2),
    "figure.dpi": 130,
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "savefig.bbox": "tight",
})
C = {"exp": "#1f77b4", "comp": "#2ca02c", "comm": "#d62728",
     "model": "#ff7f0e", "ideal": "#7f7f7f", "extra": "#9467bd",
     "bcast": "#17becf", "scatter": "#bcbd22", "gather": "#e377c2"}

TEXTO = {"impl", "variant"}     # columnas que NO son numeros


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for kk, vv in list(r.items()):
            if kk not in TEXTO:
                try:
                    r[kk] = float(vv)
                except (TypeError, ValueError):
                    pass
    return rows


def col(rows, name):
    return np.array([r[name] for r in rows], dtype=float)


def save(fig, base):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{base}.{ext}"))
    plt.close(fig)
    print(f"  figura -> figs/{base}.pdf/.png")


def main():
    todo = load("strong.csv")
    # la v3 es la implementacion final; si el csv es viejo y no trae variant, uso todo
    strong = sorted([r for r in todo if r.get("variant", "v3") == "v3"], key=lambda r: r["p"])
    v2 = sorted([r for r in todo if r.get("variant") == "v2"], key=lambda r: r["p"])
    size = sorted(load("size.csv"), key=lambda r: r["n"])
    weak = sorted(load("weak.csv"), key=lambda r: (r["n_te"], r["p"]))
    hyb = sorted(load("hybrid.csv"), key=lambda r: -r["p"])
    ener = sorted(load("energia.csv"), key=lambda r: r["p"])

    if not strong:
        sys.exit(f"[ERR] no encuentro {RES}/strong.csv - corre primero el benchmark en Khipu.")

    p = col(strong, "p")
    t_tot, t_cmp, t_cmm = col(strong, "t_total"), col(strong, "t_comp"), col(strong, "t_comm")
    gflz, acc = col(strong, "gflops_total"), col(strong, "acc")
    sd_tot = col(strong, "t_total_std") if "t_total_std" in strong[0] else np.zeros_like(p)
    n_tr, n_te = strong[0]["n_tr"], strong[0]["n_te"]
    d, k = strong[0]["d"], strong[0]["k"]
    ts = t_tot[0]                                   # el caso p=1 es mi tiempo secuencial de referencia

    alpha, beta = M.calibrate_comm(p, t_cmm, n_tr, n_te, d, k)
    c_comp = M.constante_computo(ts, n_tr, n_te, d)
    p_opt = M.optimal_p(ts, n_tr, n_te, d, k, alpha, beta)
    p_opt_exp = int(p[int(np.argmin(t_tot))])       # el optimo que se ve en los datos
    pg = np.linspace(1, p.max(), 200)
    tt_model = M.t_total(pg, ts, n_tr, n_te, d, k, alpha, beta)
    print(f"Calibracion: Ts={ts:.4f}s  alpha={alpha*1e6:.2f} us  "
          f"beta={beta*1e9:.3f} ns/byte (BW~{1/beta/1e9:.2f} GB/s)  "
          f"p_opt(modelo)={p_opt}  p_opt(medido)={p_opt_exp}")

    s_exp = ts / t_tot
    e_exp = s_exp / p
    s_mod = ts / M.t_total(pg, ts, n_tr, n_te, d, k, alpha, beta)
    e_mod = s_mod / pg
    frac = t_cmm / t_tot * 100.0

    # === FIG 1: como se reparte el tiempo entre computo y comunicacion ===
    fig, ax = plt.subplots()
    ax.errorbar(p, t_tot, yerr=sd_tot, fmt="o-", color=C["exp"], capsize=3, label="Total (medido)")
    ax.loglog(p, t_cmp, "s--", color=C["comp"], label="Computo (medido)")
    ax.loglog(p, np.maximum(t_cmm, 1e-6), "^--", color=C["comm"], label="Comunicacion (medido)")
    ax.loglog(pg, tt_model, "-", color=C["model"], lw=2.4, alpha=0.85, label="Total (modelo $\\alpha\\!-\\!\\beta$)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(p_opt_exp, color=C["extra"], ls=":", lw=1.8)
    ax.text(p_opt_exp, ax.get_ylim()[1] * 0.55, f" $p_{{opt}}={p_opt_exp}$", color=C["extra"], fontsize=10)
    ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("tiempo [s]")
    ax.set_title(f"Tiempo de ejecucion vs procesos (n={int(strong[0]['n'])}, k={int(k)})")
    ax.set_xticks(p); ax.set_xticklabels([int(x) for x in p]); ax.legend()
    save(fig, "fig_tiempos")

    # === FIG 2: speedup ===
    fig, ax = plt.subplots()
    ax.plot(p, p, "--", color=C["ideal"], label="Speedup ideal ($S=p$)")
    ax.plot(pg, s_mod, "-", color=C["model"], alpha=0.85, label="Modelo $\\alpha\\!-\\!\\beta$")
    ax.errorbar(p, s_exp, yerr=s_exp * sd_tot / t_tot, fmt="o-", color=C["exp"], capsize=3, label="Medido (Khipu)")
    ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("speedup $S(p)=T_s/T_p$")
    ax.set_title("Speedup vs procesos")
    ax.set_xticks(p); ax.set_xticklabels([int(x) for x in p]); ax.legend()
    save(fig, "fig_speedup")

    # === FIG 3: eficiencia ===
    fig, ax = plt.subplots()
    ax.axhline(1.0, color=C["ideal"], ls="--", label="Ideal ($E=1$)")
    ax.axhline(0.8, color="#bbbbbb", ls=":", label="Umbral 80%")
    ax.plot(pg, e_mod, "-", color=C["model"], alpha=0.85, label="Modelo $\\alpha\\!-\\!\\beta$")
    ax.plot(p, e_exp, "o-", color=C["exp"], label="Medido (Khipu)")
    ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("eficiencia $E(p)=S/p$")
    ax.set_title("Eficiencia vs procesos")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(p); ax.set_xticklabels([int(x) for x in p]); ax.legend()
    save(fig, "fig_eficiencia")

    # === FIG 4: FLOP/s ===
    gf_mod = (n_te * n_tr * 3 * d) / M.t_total(pg, ts, n_tr, n_te, d, k, alpha, beta) / 1e9
    fig, ax = plt.subplots()
    ax.plot(pg, gf_mod, "-", color=C["model"], alpha=0.85, label="Modelo $\\alpha\\!-\\!\\beta$")
    ax.plot(p, gflz, "o-", color=C["exp"], label="Medido (Khipu)")
    ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("rendimiento sostenido [GFLOP/s]")
    ax.set_title("FLOP/s vs procesos (region paralelizable)")
    ax.set_xticks(p); ax.set_xticklabels([int(x) for x in p]); ax.legend()
    save(fig, "fig_flops")

    # === FIG 5: tamano del problema (p=8) ===
    if size:
        ns = col(size, "n"); tts = col(size, "t_total"); accs = col(size, "acc")
        n_te_s, n_tr_s = col(size, "n_te"), col(size, "n_tr")
        work = n_tr_s * n_te_s
        kfit = np.sum(tts * work) / np.sum(work * work)
        fig, ax = plt.subplots()
        ax.loglog(ns, tts, "o-", color=C["exp"], label="Medido (p=8)")
        ax.loglog(ns, kfit * work, "--", color=C["model"], label="Modelo $\\propto n_{tr}\\,n_{te}/p$")
        ax.set_xlabel("tamano del problema $n$"); ax.set_ylabel("tiempo total [s]")
        ax.set_title("Escalabilidad con el tamano del problema (p=8)")
        ax2 = ax.twinx(); ax2.plot(ns, accs, "^:", color=C["comp"], label="accuracy")
        ax2.set_ylabel("accuracy"); ax2.set_ylim(0.9, 1.02); ax2.grid(False)
        l1, lab1 = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, loc="upper left")
        save(fig, "fig_tamano")
        gf2 = col(size, "gflops_total")
        fig, ax = plt.subplots()
        ax.semilogx(ns, gf2, "o-", color=C["comp"])
        ax.set_xlabel("tamano del problema $n$"); ax.set_ylabel("rendimiento sostenido [GFLOP/s]")
        ax.set_title("Rendimiento vs tamano del problema (p=8)")
        save(fig, "fig_gflops_n")

    # === FIG 6: escalabilidad debil, UNA CURVA POR CARGA ===
    ew_min = np.nan
    if weak:
        cargas = sorted({r["n_te"] for r in weak})
        fig, ax = plt.subplots()
        ax.axhline(1.0, color=C["ideal"], ls="--", label="Ideal (debil)")
        for i, carga in enumerate(cargas):
            g = [r for r in weak if r["n_te"] == carga]
            pw = col(g, "p"); tw = col(g, "t_total")
            ew = tw[0] / tw
            ax.plot(pw, ew, "o-", color=plt.cm.viridis(i / max(len(cargas) - 1, 1)),
                    label=f"carga/proc = {int(carga)}")
            ew_min = np.nanmin([ew_min, ew.min()]) if not np.isnan(ew_min) else ew.min()
        ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("eficiencia debil $T(1)/T(p)$")
        ax.set_title("Escalabilidad debil: una curva por carga por proceso")
        ax.set_ylim(0, 1.15); ax.set_xscale("log", base=2)
        ax.set_xticks(pw); ax.set_xticklabels([int(x) for x in pw]); ax.legend()
        save(fig, "fig_debil")

    # === FIG 7: peso de la comunicacion ===
    fig, ax = plt.subplots()
    bars = ax.bar([str(int(x)) for x in p], frac, color=C["comm"], alpha=0.85)
    for b, v in zip(bars, frac):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("comunicacion / total [%]")
    ax.set_title("Peso de la comunicacion en el tiempo total")
    ax.set_ylim(0, max(frac) * 1.25)
    save(fig, "fig_fraccion_comm")

    # === FIG 8: Karp-Flatt ===
    m = p > 1
    e_kf = M.karp_flatt(s_exp, p)[m]
    fig, ax = plt.subplots()
    ax.plot(p[m], e_kf, "o-", color=C["extra"])
    ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("fraccion serie experimental $e$")
    ax.set_title("Metrica de Karp--Flatt (si $e$ crece, manda el overhead)")
    ax.set_xticks(p[m]); ax.set_xticklabels([int(x) for x in p[m]])
    ax.set_ylim(0, max(e_kf) * 1.4)
    save(fig, "fig_karpflatt")

    # === FIG 10 (nueva): desglose por colectiva ===
    if "t_bcast" in strong[0]:
        tb, tsc, tg = col(strong, "t_bcast"), col(strong, "t_scatter"), col(strong, "t_gather")
        fig, ax = plt.subplots()
        ax.plot(p, tb, "o-", color=C["bcast"], label="bcast (test)")
        ax.plot(p, tsc, "s-", color=C["scatter"], label="scatter (train)")
        ax.plot(p, tg, "^-", color=C["gather"], label="gather ($p\\cdot k\\cdot n_{te}$)")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("tiempo [s]")
        ax.set_title("Costo de cada colectiva (reloj del maestro)")
        ax.set_xticks(p); ax.set_xticklabels([int(x) for x in p]); ax.legend()
        save(fig, "fig_colectivas")

    # === FIG 11 (nueva): iso-eficiencia N(p) ===
    frac_te = n_te / (n_tr + n_te)
    pg2 = np.array([2, 4, 8, 16, 32, 64, 128, 256], dtype=float)
    fig, ax = plt.subplots()
    iso_exp = np.nan
    for eo, cl in ((0.8, C["exp"]), (0.5, C["comm"])):
        N = M.iso_eficiencia_N(pg2, eo, frac_te, d, k, c_comp, alpha, beta)
        ok = ~np.isnan(N)
        ax.loglog(pg2[ok], N[ok], "o-", color=cl, label=f"$E={eo}$")
        q = np.polyfit(np.log(pg2[ok]), np.log(N[ok]), 1)[0]
        if eo == 0.8:
            iso_exp = q
    ref = pg2 ** 2 * np.log2(pg2)
    ax.loglog(pg2, ref / ref[0] * 6e3, "--", color=C["ideal"], label="referencia $p^2\\log_2 p$")
    ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("$N$ necesario")
    ax.set_title(f"Iso-eficiencia: $N(p)\\sim p^{{{iso_exp:.2f}}}$")
    ax.legend()
    save(fig, "fig_isoeficiencia")

    # === FIG 12 (nueva): la mejora v3 vs v2 ===
    if v2:
        p2 = col(v2, "p"); c2 = col(v2, "t_comm")
        fig, ax = plt.subplots()
        ax.loglog(p2, c2, "s--", color=C["ideal"], label="v2 (colectivas con pickle)")
        ax.loglog(p, t_cmm, "o-", color=C["exp"], label="v3 (colectivas con buffer)")
        ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("$T_{comm}$ [s]")
        ax.set_title("Mejora: colectivas con buffer vs serializadas")
        ax.set_xticks(p); ax.set_xticklabels([int(x) for x in p]); ax.legend()
        save(fig, "fig_v2_v3")

    # === FIG 13 (nueva): hibrido MPI+OpenMP con 32 cores ===
    if hyb:
        etiq = [f"{int(r['p'])}x{int(r['threads'])}" for r in hyb]
        th = col(hyb, "t_total")
        fig, ax = plt.subplots()
        b = ax.bar(etiq, th, color=C["extra"], alpha=0.85)
        for bb, v in zip(b, th):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.008, f"{v:.3f}s", ha="center", fontsize=9)
        ax.axhline(t_tot[int(np.argmin(t_tot))], color=C["comp"], ls="--",
                   label=f"MPI puro $p={p_opt_exp}$ (16 cores)")
        ax.set_xlabel("rangos MPI $\\times$ hilos OpenMP (siempre 32 cores)")
        ax.set_ylabel("tiempo total [s]")
        ax.set_title("Hibrido: mismo presupuesto de cores, distinta reparticion")
        ax.legend()
        save(fig, "fig_hibrido")

    # === FIG 14 (nueva): energia (RAPL) ===
    if ener:
        pe = col(ener, "p"); ej = col(ener, "energia_J"); jq = col(ener, "J_por_consulta")
        p_ener = int(pe[int(np.argmin(ej))])
        fig, ax = plt.subplots()
        ax.plot(pe, ej, "o-", color=C["comm"], label="energia total [J]")
        ax.axvline(p_ener, color=C["comm"], ls=":", lw=1.6)
        ax.text(p_ener, ej.max() * 0.95, f" min energia $p={p_ener}$", color=C["comm"], fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("energia [J]")
        ax.set_title("Consumo energetico (RAPL) vs procesos")
        ax2 = ax.twinx(); ax2.plot(pe, jq, "^:", color=C["extra"], label="J / consulta")
        ax2.set_ylabel("J por consulta"); ax2.grid(False)
        l1, a1 = ax.get_legend_handles_labels(); l2, a2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, a1 + a2, loc="upper center")
        ax.set_xticks(pe); ax.set_xticklabels([int(x) for x in pe])
        save(fig, "fig_energia")

        # === FIG 15 (nueva): ¿la energia va como 1/eficiencia o como 1/speedup? ===
        # En clase se dice que el consumo es inversamente proporcional a la eficiencia.
        # Con nuestros datos no cierra: E = P*t = P*Ts/S, asi que va como 1/SPEEDUP.
        # Se ve solo: J*S queda casi plano (lo que se mueve es la potencia), J*E se derrumba.
        te = col(ener, "tiempo_s")
        Se = te[0] / te
        Ee = Se / pe
        fig, ax = plt.subplots()
        ax.plot(pe, ej * Se, "o-", color=C["exp"], label="$J\\cdot S$  (si $J\\propto 1/S$, es constante)")
        ax.plot(pe, ej * Ee, "s--", color=C["comm"], label="$J\\cdot E$  (si $J\\propto 1/E$, seria constante)")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("[J]")
        ax.set_title("La energia va como $1/S$, no como $1/E$")
        ax.set_xticks(pe); ax.set_xticklabels([int(x) for x in pe]); ax.legend()
        save(fig, "fig_energia_ley")

    # ----------------------------- tablas -----------------------------
    def tabla(nombre, cab, filas, fmt):
        with open(os.path.join(INFORME, nombre), "w", encoding="utf-8") as f:
            f.write(f"\\begin{{tabular}}{{{fmt}}}\n\\toprule\n{cab} \\\\\n\\midrule\n")
            for fila in filas:
                f.write(fila + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        print(f"  tabla  -> informe/{nombre}")

    tabla("tabla_resultados.tex",
          "$p$ & $T_{tot}$ [s] & $T_{comp}$ [s] & $T_{comm}$ [s] & $S(p)$ & $E(p)$ & acc",
          [f"{int(p[i])} & {t_tot[i]:.4f} & {t_cmp[i]:.4f} & {t_cmm[i]:.4f} & "
           f"{s_exp[i]:.2f} & {e_exp[i]:.2f} & {acc[i]:.4f}" for i in range(len(p))],
          "rrrrrrr")

    if "t_bcast" in strong[0]:
        tabla("tabla_colectivas.tex",
              "$p$ & $T_{bcast}$ & $T_{scatter}$ & $T_{comp}$ & $T_{gather}$ & suma & $T_{tot}$",
              [f"{int(p[i])} & {tb[i]:.4f} & {tsc[i]:.4f} & {t_cmp[i]:.4f} & {tg[i]:.4f} & "
               f"{tb[i]+tsc[i]+t_cmp[i]+tg[i]:.4f} & {t_tot[i]:.4f}" for i in range(len(p))],
              "rrrrrrr")

    if size:
        tabla("tabla_size.tex", "$n$ & $n_{tr}$ & $n_{te}$ & $T_{tot}$ [s] & GFLOP/s & acc",
              [f"{int(r['n'])} & {int(r['n_tr'])} & {int(r['n_te'])} & {r['t_total']:.4f} & "
               f"{r['gflops_total']:.1f} & {r['acc']:.4f}" for r in size], "rrrrrr")

    if weak:
        filas = []
        for carga in sorted({r["n_te"] for r in weak}):
            g = [r for r in weak if r["n_te"] == carga]
            t1 = g[0]["t_total"]
            for r in g:
                filas.append(f"{int(carga)} & {int(r['p'])} & {int(r['n_tr'])} & "
                             f"{r['t_total']:.4f} & {r['t_comm']:.4f} & {t1/r['t_total']:.2f}")
        tabla("tabla_weak.tex", "carga/proc & $p$ & $n_{tr}$ & $T_{tot}$ [s] & $T_{comm}$ [s] & $E_{debil}$",
              filas, "rrrrrr")

    if hyb:
        tabla("tabla_hibrido.tex", "rangos $\\times$ hilos & cores & $T_{tot}$ [s] & $T_{comp}$ & $T_{comm}$ & GFLOP/s",
              [f"{int(r['p'])} $\\times$ {int(r['threads'])} & {int(r['p']*r['threads'])} & "
               f"{r['t_total']:.4f} & {r['t_comp']:.4f} & {r['t_comm']:.4f} & {r['gflops_total']:.1f}"
               for r in hyb], "rrrrrr")

    if ener:
        tabla("tabla_energia.tex", "$p$ & $t$ [s] & $E$ [J] & $P$ [W] & J/consulta",
              [f"{int(r['p'])} & {r['tiempo_s']:.2f} & {r['energia_J']:.1f} & "
               f"{r['potencia_W']:.1f} & {r['J_por_consulta']:.4f}" for r in ener], "rrrrr")

    # ------------------- numeritos que cito en el texto -------------------
    with open(os.path.join(INFORME, "modelo_params.tex"), "w", encoding="utf-8") as f:
        f.write(f"\\newcommand{{\\Ts}}{{{ts:.4f}}}\n")
        f.write(f"\\newcommand{{\\alphaus}}{{{alpha*1e6:.0f}}}\n")
        f.write(f"\\newcommand{{\\betans}}{{{beta*1e9:.2f}}}\n")
        f.write(f"\\newcommand{{\\bw}}{{{1/beta/1e9:.2f}}}\n")
        f.write(f"\\newcommand{{\\popt}}{{{p_opt}}}\n")
        f.write(f"\\newcommand{{\\poptexp}}{{{p_opt_exp}}}\n")
        f.write(f"\\newcommand{{\\accbase}}{{{acc[0]:.4f}}}\n")
        f.write(f"\\newcommand{{\\smax}}{{{s_exp.max():.2f}}}\n")
        f.write(f"\\newcommand{{\\sfinal}}{{{s_exp[-1]:.2f}}}\n")
        f.write(f"\\newcommand{{\\pmax}}{{{int(p.max())}}}\n")
        f.write(f"\\newcommand{{\\tmin}}{{{t_tot.min():.4f}}}\n")
        f.write(f"\\newcommand{{\\tfinal}}{{{t_tot[-1]:.4f}}}\n")
    print("  params -> informe/modelo_params.tex")

    with open(os.path.join(INFORME, "modelo_extra.tex"), "w", encoding="utf-8") as f:
        f.write(f"\\newcommand{{\\fraccommax}}{{{frac.max():.0f}}}\n")
        f.write(f"\\newcommand{{\\ekfmin}}{{{np.nanmin(e_kf):.3f}}}\n")
        f.write(f"\\newcommand{{\\ekfmax}}{{{np.nanmax(e_kf):.3f}}}\n")
        f.write(f"\\newcommand{{\\isoexp}}{{{iso_exp:.2f}}}\n")
        f.write(f"\\newcommand{{\\emin}}{{{e_exp.min():.3f}}}\n")
        if size:
            f.write(f"\\newcommand{{\\gfmaxn}}{{{gf2.max():.0f}}}\n")
            f.write(f"\\newcommand{{\\nmax}}{{{int(ns.max())}}}\n")
        if weak:
            f.write(f"\\newcommand{{\\ewmin}}{{{ew_min:.2f}}}\n")
        if hyb:
            mejor = min(hyb, key=lambda r: r["t_total"])
            peor = max(hyb, key=lambda r: r["t_total"])
            f.write(f"\\newcommand{{\\hybmejor}}{{{int(mejor['p'])}\\times{int(mejor['threads'])}}}\n")
            f.write(f"\\newcommand{{\\hybgain}}{{{peor['t_total']/mejor['t_total']:.2f}}}\n")
        if ener:
            # OJO: el bloque de energia corre en UN nodo (RAPL solo ve el nodo local), asi que
            # su optimo de tiempo NO es el del barrido fuerte multi-nodo. Hay que compararlos
            # dentro del mismo experimento o no vale.
            te = col(ener, "tiempo_s"); pw = col(ener, "potencia_W")
            p_t_ener = int(pe[int(np.argmin(te))])
            Se = te[0] / te                      # speedup dentro del bloque de energia
            Ee = Se / pe
            js, jee = ej * Se, ej * Ee           # si energia ~ 1/S, entonces J*S es constante
            f.write(f"\\newcommand{{\\eneropt}}{{{p_ener}}}\n")
            f.write(f"\\newcommand{{\\enerptiempo}}{{{p_t_ener}}}\n")
            f.write(f"\\newcommand{{\\enerjmin}}{{{ej.min():.0f}}}\n")
            f.write(f"\\newcommand{{\\enerjq}}{{{jq.min():.4f}}}\n")
            f.write(f"\\newcommand{{\\enerpotpct}}{{{100*(pw.max()/pw.min()-1):.0f}}}\n")
            f.write(f"\\newcommand{{\\enerjspct}}{{{100*(js.max()/js.min()-1):.0f}}}\n")
            f.write(f"\\newcommand{{\\enerjepct}}{{{100*(jee.max()/jee.min()-1):.0f}}}\n")
        if v2:
            f.write(f"\\newcommand{{\\vtresganancia}}{{{(c2[0]/t_cmm[0]):.1f}}}\n")
    print("  params -> informe/modelo_extra.tex")


if __name__ == "__main__":
    main()
