"""Langkah 2 - Bagi train/val berdasarkan SESI (tanggal + kondisi cahaya).

  python split.py                     val = sesi lab_redup untuk SEMUA kelas (default)
  python split.py --val lab_terang    val = sesi lab_terang untuk SEMUA kelas
  python split.py --mode blok         tiap sesi: 80% awal -> train, 20% akhir -> val

Mode sesi memakai kondisi val yang SAMA di semua kelas, sehingga model tidak bisa
menebak kelas dari tingkat kecerahan (shortcut learning).
"""
import argparse, csv, shutil
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["sesi", "blok"], default="sesi")
ap.add_argument("--val", default="lab_redup", help="kondisi cahaya untuk val (mode sesi)")
ap.add_argument("--frac", type=float, default=0.2, help="porsi val (mode blok)")
a = ap.parse_args()

RAW, OUT = Path("dataset_raw"), Path("dataset")
rows = list(csv.DictReader(open(RAW / "metadata.csv", encoding="utf-8")))
data = defaultdict(lambda: defaultdict(list))            # kelas -> kondisi -> [file]
for r in rows:
    data[r["kelas"]][(r["tanggal"], r["kondisi_cahaya"])].append(r["nama_file"])

if OUT.exists():
    shutil.rmtree(OUT)

def blok(sesi):
    val = set()
    for files in sesi.values():
        files = sorted(files)
        val |= set(files[len(files) - max(1, round(len(files) * a.frac)):])
    return val

print(f"mode={a.mode}" + (f"  val={a.val}" if a.mode == "sesi" else f"  frac={a.frac}"))
print(f"{'KELAS':12s} {'TRAIN':>5s} {'VAL':>5s}  METODE")
for kelas, sesi in sorted(data.items()):
    metode = a.mode
    if a.mode == "sesi":
        val = {f for (tgl, kond), fs in sesi.items() if kond == a.val for f in fs}
        semua = {f for fs in sesi.values() for f in fs}
        if not val or val == semua:                      # kelas tak punya 2 kondisi
            val, metode = blok(sesi), "blok (KELAS INI TIDAK PUNYA 2 KONDISI)"
    else:
        val = blok(sesi)
    n_tr = n_va = 0
    for fs in sesi.values():
        for f in fs:
            split = "val" if f in val else "train"
            dst = OUT / split / kelas
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RAW / kelas / f, dst / f)
            n_va += split == "val"; n_tr += split == "train"
    print(f"{kelas:12s} {n_tr:5d} {n_va:5d}  {metode}")
