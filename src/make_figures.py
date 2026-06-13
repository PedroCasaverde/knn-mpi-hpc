"""
Saca las figuras del informe usando los CSV que dejo el benchmark en Khipu.
Encima dibujo la curva teorica alpha-beta para comparar teoria vs experimento.

    python src/make_figures.py
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

# dejo los plots un poco mas presentables para que no se vean por defecto
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
     "model": "#ff7f0e", "ideal": "#7f7f7f", "extra": "#9467bd"}


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for kk, vv in r.items():
            if kk != "impl":
                r[kk] = float(vv)
    return rows


def save(fig, base):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{base}.{ext}"))
    plt.close(fig)
    print(f"  figura -> figs/{base}.pdf/.png")


def main():
    strong = sorted(load("strong.csv"), key=lambda r: r["p"])
    size = sorted(load("size.csv"), key=lambda r: r["n"])
    weak = sorted(load("weak.csv"), key=lambda r: r["p"])

    if not strong:
        sys.exit(f"[ERR] no encuentro {RES}/strong.csv - corre primero el benchmark en Khipu.")

    p = np.array([r["p"] for r in strong])
    t_tot = np.array([r["t_total"] for r in strong])
    t_cmp = np.array([r["t_comp"] for r in strong])
    t_cmm = np.array([r["t_comm"] for r in strong])
    gflz = np.array([r["gflops_total"] for r in strong])
    acc = np.array([r["acc"] for r in strong])
    n_tr, n_te = strong[0]["n_tr"], strong[0]["n_te"]
    d, k = strong[0]["d"], strong[0]["k"]
    ts = t_tot[0]                                   # el caso p=1 es mi tiempo secuencial de referencia

    # ajusto alpha y beta del modelo a lo que medi (calibracion a los datos)
    alpha, beta = M.calibrate_comm(p, t_cmm, n_tr, n_te, d, k)
    p_opt = M.optimal_p(ts, n_tr, n_te, d, k, alpha, beta)
    pg = np.linspace(1, p.max(), 200)
    tt_model = M.t_total(pg, ts, n_tr, n_te, d, k, alpha, beta)
    print(f"Calibracion: Ts={ts:.4f}s  alpha={alpha*1e6:.2f} us  "
          f"beta={beta*1e9:.3f} ns/byte (BW~{1/beta/1e9:.2f} GB/s)  p_opt={p_opt}")

    # === FIG 1: como se reparte el tiempo entre computo y comunicacion ===
    fig, ax = plt.subplots()
    ax.loglog(p, t_tot, "o-", color=C["exp"], label="Total (medido)")
    ax.loglog(p, t_cmp, "s--", color=C["comp"], label="Computo (medido)")
    ax.loglog(p, np.maximum(t_cmm, 1e-6), "^--", color=C["comm"], label="Comunicacion (medido)")
    ax.loglog(pg, tt_model, "-", color=C["model"], lw=2.4, alpha=0.85, label="Total (modelo $\\alpha\\!-\\!\\beta$)")
    ax.axvline(p_opt, color=C["extra"], ls=":", lw=1.8)
    ax.text(p_opt, ax.get_ylim()[1]*0.6, f" $p_{{opt}}\\approx{p_opt}$", color=C["extra"], fontsize=10)
    ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("tiempo [s]")
    ax.set_title(f"Tiempo de ejecucion vs procesos (n={int(strong[0]['n'])}, k={int(k)})")
    ax.set_xticks(p); ax.set_xticklabels([int(x) for x in p]); ax.legend()
    save(fig, "fig_tiempos")

    # === FIG 2: speedup, lo medido contra la curva ideal y el modelo ===
    s_exp = ts / t_tot
    s_mod = ts / M.t_total(pg, ts, n_tr, n_te, d, k, alpha, beta)
    fig, ax = plt.subplots()
    ax.plot(p, p, "--", color=C["ideal"], label="Speedup ideal ($S=p$)")
    ax.plot(pg, s_mod, "-", color=C["model"], alpha=0.85, label="Modelo $\\alpha\\!-\\!\\beta$")
    ax.plot(p, s_exp, "o-", color=C["exp"], label="Medido (Khipu)")
    ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("speedup $S(p)=T_s/T_p$")
    ax.set_title("Speedup vs procesos")
    ax.set_xticks(p); ax.set_xticklabels([int(x) for x in p]); ax.legend()
    save(fig, "fig_speedup")

    # === FIG 3: eficiencia, con la rayita del 80% para tener referencia ===
    e_exp = s_exp / p
    e_mod = s_mod / pg
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

    # === FIG 4: rendimiento sostenido en GFLOP/s al subir p ===
    gf_mod = (n_te * n_tr * 3 * d) / M.t_total(pg, ts, n_tr, n_te, d, k, alpha, beta) / 1e9
    fig, ax = plt.subplots()
    ax.plot(pg, gf_mod, "-", color=C["model"], alpha=0.85, label="Modelo $\\alpha\\!-\\!\\beta$")
    ax.plot(p, gflz, "o-", color=C["exp"], label="Medido (Khipu)")
    ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("rendimiento sostenido [GFLOP/s]")
    ax.set_title("FLOP/s vs procesos (region paralelizable)")
    ax.set_xticks(p); ax.set_xticklabels([int(x) for x in p]); ax.legend()
    save(fig, "fig_flops")

    # === FIG 5: que pasa al agrandar el problema con p=8 fijo ===
    if size:
        ns = np.array([r["n"] for r in size])
        tts = np.array([r["t_total"] for r in size])
        accs = np.array([r["acc"] for r in size])
        # el computo va como n_tr*n_te/p (o sea ~n^2); solo me falta la constante, la saco de los datos
        n_te_s = np.array([r["n_te"] for r in size]); n_tr_s = np.array([r["n_tr"] for r in size])
        work = n_tr_s * n_te_s
        kfit = np.sum(tts * work) / np.sum(work * work)   # minimos cuadrados forzando que pase por el origen
        fig, ax = plt.subplots()
        ax.loglog(ns, tts, "o-", color=C["exp"], label="Medido (p=8)")
        ax.loglog(ns, kfit * work, "--", color=C["model"], label="Modelo $\\propto n_{tr}\\,n_{te}/p$")
        ax.set_xlabel("tamano del problema $n$"); ax.set_ylabel("tiempo total [s]")
        ax.set_title("Escalabilidad con el tamano del problema (p=8)")
        ax2 = ax.twinx(); ax2.plot(ns, accs, "^:", color=C["comp"], label="accuracy")
        ax2.set_ylabel("accuracy"); ax2.set_ylim(0.9, 1.0); ax2.grid(False)
        l1, lab1 = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, loc="upper left")
        save(fig, "fig_tamano")

    # === FIG 6: escalabilidad debil ===
    if weak:
        pw = np.array([r["p"] for r in weak])
        ew = (weak[0]["t_total"]) / np.array([r["t_total"] for r in weak])  # eficiencia debil = T(1)/T(p), manteniendo n/p constante
        fig, ax = plt.subplots()
        ax.axhline(1.0, color=C["ideal"], ls="--", label="Ideal (debil)")
        ax.plot(pw, ew, "o-", color=C["extra"], label="Medido ($n_{tr}=2000\\,p,\\ n_{te}=2000$)")
        ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("eficiencia debil $T(1)/T(p)$")
        ax.set_title("Escalabilidad debil (n/p constante)")
        ax.set_ylim(0, 1.15)
        ax.set_xticks(pw); ax.set_xticklabels([int(x) for x in pw]); ax.legend()
        save(fig, "fig_debil")

    # === FIG 7: cuanto pesa la comunicacion sobre el total, en % ===
    frac = t_cmm / t_tot * 100.0
    fig, ax = plt.subplots()
    bars = ax.bar([str(int(x)) for x in p], frac, color=C["comm"], alpha=0.85)
    for b, v in zip(bars, frac):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("comunicacion / total [%]")
    ax.set_title("Peso de la comunicacion en el tiempo total")
    ax.set_ylim(0, max(frac) * 1.25)
    save(fig, "fig_fraccion_comm")

    # === FIG 8: Karp-Flatt, para estimar la fraccion serie a partir de lo medido ===
    m = p > 1
    e_kf = (1.0 / s_exp[m] - 1.0 / p[m]) / (1.0 - 1.0 / p[m])
    fig, ax = plt.subplots()
    ax.plot(p[m], e_kf, "o-", color=C["extra"])
    ax.set_xlabel("numero de procesos $p$"); ax.set_ylabel("fraccion serie experimental $e$")
    ax.set_title("Metrica de Karp--Flatt")
    ax.set_xticks(p[m]); ax.set_xticklabels([int(x) for x in p[m]])
    ax.set_ylim(0, max(e_kf) * 1.4)
    save(fig, "fig_karpflatt")

    # === FIG 9: rendimiento sostenido conforme crece el problema (p=8) ===
    if size:
        ns2 = np.array([r["n"] for r in size]); gf2 = np.array([r["gflops_total"] for r in size])
        fig, ax = plt.subplots()
        ax.semilogx(ns2, gf2, "o-", color=C["comp"])
        ax.set_xlabel("tamano del problema $n$"); ax.set_ylabel("rendimiento sostenido [GFLOP/s]")
        ax.set_title("Rendimiento vs tamano del problema (p=8)")
        save(fig, "fig_gflops_n")

    # === tabla LaTeX con el barrido de tamano (p=8) ===
    if size:
        with open(os.path.join(INFORME, "tabla_size.tex"), "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{rrrrrr}\n\\toprule\n")
            f.write("$n$ & $n_{tr}$ & $n_{te}$ & $T_{tot}$ [s] & GFLOP/s & acc \\\\\n\\midrule\n")
            for r in size:
                f.write(f"{int(r['n'])} & {int(r['n_tr'])} & {int(r['n_te'])} & "
                        f"{r['t_total']:.4f} & {r['gflops_total']:.1f} & {r['acc']:.4f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        print("  tabla  -> informe/tabla_size.tex")

    # === tabla LaTeX de la escalabilidad debil ===
    if weak:
        t1w = weak[0]["t_total"]
        with open(os.path.join(INFORME, "tabla_weak.tex"), "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{rrrrr}\n\\toprule\n")
            f.write("$p$ & $n_{tr}$ & $T_{tot}$ [s] & $T_{comm}$ [s] & $E_{debil}$ \\\\\n\\midrule\n")
            for r in weak:
                ew = t1w / r["t_total"]
                f.write(f"{int(r['p'])} & {int(r['n_tr'])} & {r['t_total']:.4f} & "
                        f"{r['t_comm']:.4f} & {ew:.2f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        print("  tabla  -> informe/tabla_weak.tex")

    # numeritos sueltos que despues cito directo en el texto del informe
    with open(os.path.join(INFORME, "modelo_extra.tex"), "w", encoding="utf-8") as f:
        f.write(f"\\newcommand{{\\fraccommax}}{{{frac.max():.0f}}}\n")
        f.write(f"\\newcommand{{\\ekfmin}}{{{e_kf.min():.3f}}}\n")
        f.write(f"\\newcommand{{\\ekfmax}}{{{e_kf.max():.3f}}}\n")
        if size:
            f.write(f"\\newcommand{{\\gfmaxn}}{{{gf2.max():.0f}}}\n")
            f.write(f"\\newcommand{{\\nmax}}{{{int(ns2.max())}}}\n")
        if weak:
            f.write(f"\\newcommand{{\\ewmin}}{{{(weak[0]['t_total']/weak[-1]['t_total']):.2f}}}\n")
    print("  params -> informe/modelo_extra.tex")

    # === tabla LaTeX del barrido fuerte (escalabilidad fuerte) ===
    tex = os.path.join(ROOT, "informe", "tabla_resultados.tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{rrrrrrr}\n\\toprule\n")
        f.write("$p$ & $T_{tot}$ [s] & $T_{comp}$ [s] & $T_{comm}$ [s] & "
                "$S(p)$ & $E(p)$ & acc \\\\\n\\midrule\n")
        for i in range(len(p)):
            f.write(f"{int(p[i])} & {t_tot[i]:.4f} & {t_cmp[i]:.4f} & {t_cmm[i]:.4f} & "
                    f"{s_exp[i]:.2f} & {e_exp[i]:.2f} & {acc[i]:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  tabla  -> informe/tabla_resultados.tex")

    # parametros del modelo alpha-beta calibrado, para citarlos en el texto
    with open(os.path.join(ROOT, "informe", "modelo_params.tex"), "w", encoding="utf-8") as f:
        f.write(f"\\newcommand{{\\Ts}}{{{ts:.4f}}}\n")
        f.write(f"\\newcommand{{\\alphaus}}{{{alpha*1e6:.2f}}}\n")
        f.write(f"\\newcommand{{\\betans}}{{{beta*1e9:.3f}}}\n")
        f.write(f"\\newcommand{{\\bw}}{{{1/beta/1e9:.1f}}}\n")
        f.write(f"\\newcommand{{\\popt}}{{{p_opt}}}\n")
        f.write(f"\\newcommand{{\\accbase}}{{{acc[0]:.4f}}}\n")
        f.write(f"\\newcommand{{\\smax}}{{{s_exp.max():.2f}}}\n")
        f.write(f"\\newcommand{{\\pmax}}{{{int(p.max())}}}\n")
    print("  params -> informe/modelo_params.tex")


if __name__ == "__main__":
    main()
