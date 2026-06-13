"""
Script para correr todo el barrido de KNN-MPI en Khipu sin tener que hacerlo a mano.

Basicamente sube los archivos, arma el venv, lanza el job y baja los resultados.
La contraseña la saco de KHIPU_PASSWORD o de --password-file para no dejarla en el repo.

Como lo corro:
    set KHIPU_PASSWORD=...           &&  python tools/khipu_run.py
    python tools/khipu_run.py --password-file ../../access.txt
"""
import argparse
import os
import posixpath
import sys
import time

import paramiko

HOST = "khipu.utec.edu.pe"
REMOTE_DIR = "knn-mpi"            # cuelga de $HOME en Khipu
VENV = "knnmpi-venv"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UPLOADS = ["knn_digits_sec.py", "knn_mpi.py", "digits.npz", "run_khipu.slurm"]


def read_password(args):
    if os.environ.get("KHIPU_PASSWORD"):
        return os.environ["KHIPU_PASSWORD"]
    if args.password_file and os.path.exists(args.password_file):
        txt = open(args.password_file, encoding="utf-8").read()
        return txt.split("password:", 1)[1].strip() if "password:" in txt else txt.strip()
    sys.exit("[ERR] define KHIPU_PASSWORD o pasa --password-file")


def connect(user, pwd):
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, 22, user, pwd, look_for_keys=False, allow_agent=False, timeout=25)
    return cli


def run(cli, cmd, timeout=600, log=True):
    if log:
        print(f"[remote]$ {cmd}")
    _, so, se = cli.exec_command(cmd, timeout=timeout)
    rc = so.channel.recv_exit_status()
    out, err = so.read().decode(errors="replace"), se.read().decode(errors="replace")
    if log and out.strip():
        print(out.rstrip())
    if log and err.strip():
        print("[stderr]", err.rstrip())
    return rc, out, err


def ensure_venv(cli):
    rc, out, _ = run(cli, f"test -x $HOME/{VENV}/bin/python && echo OK || echo MISSING", log=False)
    if "OK" in out:
        print("venv ya existe.")
        return
    print("Creando venv (numpy + mpi4py contra OpenMPI)...")
    # uso este python3 (el modulo de ohpc) pq es el mismo en login y en los nodos de computo
    run(cli, "module load gnu12 openmpi4 python3/3.11.11 && python3 -m venv $HOME/{v} && "
             "$HOME/{v}/bin/pip install -q --upgrade pip && "
             "MPICC=$(which mpicc) $HOME/{v}/bin/pip install -q numpy mpi4py".format(v=VENV),
        timeout=600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="pedro.casaverde")
    ap.add_argument("--password-file", default=os.path.join(ROOT, "..", "..", "..", "access.txt"))
    ap.add_argument("--wait-min", type=int, default=30, help="minutos maximos de espera")
    args = ap.parse_args()

    pwd = read_password(args)
    cli = connect(args.user, pwd)
    print(f"Conectado a {HOST} como {args.user}.")

    run(cli, f"mkdir -p {REMOTE_DIR}/results {REMOTE_DIR}/logs", log=False)
    ensure_venv(cli)

    print("\n== Subiendo archivos ==")
    sftp = cli.open_sftp()
    for fn in UPLOADS:
        lp = os.path.join(ROOT, fn)
        sftp.put(lp, posixpath.join(REMOTE_DIR, fn))
        print(f"  subido {fn}")

    print("\n== Lanzando job SLURM ==")
    rc, out, _ = run(cli, f"cd {REMOTE_DIR} && sbatch run_khipu.slurm")
    jid = "".join(c for c in out.strip().split()[-1] if c.isdigit()) if out.strip() else ""
    if not jid:
        sys.exit("[ERR] no se obtuvo job id")
    print(f"  jobId = {jid}")

    print("\n== Esperando ==")
    deadline = time.time() + args.wait_min * 60
    while time.time() < deadline:
        rc, out, _ = run(cli, f"squeue -j {jid} --noheader -o '%t %M %R'", log=False)
        if not out.strip():
            print("  job terminado.")
            break
        print(f"  [{time.strftime('%H:%M:%S')}] estado: {out.strip()}")
        time.sleep(15)
    else:
        print("  [TIMEOUT] el job sigue en cola; vuelve a correr para descargar luego.")

    print("\n== Descargando resultados ==")
    local = os.path.join(ROOT, "results", "khipu")
    os.makedirs(local, exist_ok=True)
    for sub in ("results", "logs"):
        rc, out, _ = run(cli, f"ls -1 {REMOTE_DIR}/{sub}/ 2>/dev/null", log=False)
        for name in out.split():
            try:
                sftp.get(posixpath.join(REMOTE_DIR, sub, name), os.path.join(local, name))
                print(f"  descargado {sub}/{name}")
            except IOError as e:
                print(f"  no se pudo {sub}/{name}: {e}")
    sftp.close()
    cli.close()
    print(f"\nLISTO. Resultados en {local}")


if __name__ == "__main__":
    main()
