"""Menyusun dataset_raw/<kelas>/ dan dataset_raw/metadata.csv dari citra
bernama: <kelas>_<tanggal 8 digit>_<kondisi>_<nomor>.png
Contoh : rfid_rc522_20261006_lab_terang_070.png
         -> kelas=rfid_rc522, tanggal=20261006, kondisi=lab_terang

Pemakaian (dari folder ret503_p2):
    python make_metadata.py                 # cari citra di folder ini dan subfoldernya
    python make_metadata.py C:\\lokasi\\foto  # cari citra di folder tertentu
"""
import csv, re, shutil, sys
from collections import Counter
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
RAW = Path("dataset_raw")
EXT = {".png", ".jpg", ".jpeg"}
SKIP = {"dataset_raw", "dataset", "venv", "results", "models", ".git"}
POLA = re.compile(r"^(?P<kelas>.+?)_(?P<tgl>\d{8})_(?P<kondisi>.+)_(?P<no>\d+)$")

rows, salah, terlihat = [], [], set()
for f in sorted(SRC.rglob("*")):
    if f.suffix.lower() not in EXT:
        continue
    if SKIP & set(f.relative_to(SRC).parts[:-1]):
        continue
    m = POLA.match(f.stem)
    if not m:
        salah.append(f.name)
        continue
    kelas, tgl, kondisi = m["kelas"], m["tgl"], m["kondisi"]
    if f.name in terlihat:                        # lewati nama file kembar
        continue
    terlihat.add(f.name)
    tujuan = RAW / kelas
    tujuan.mkdir(parents=True, exist_ok=True)
    if not (tujuan / f.name).exists():
        shutil.copy2(f, tujuan / f.name)          # salin, file asli tidak dihapus
    rows.append([f.name, kelas, tgl, kondisi])

if not rows:
    sys.exit("Tidak ada citra dengan pola nama yang cocok. Cek folder sumber / nama file.")

with open(RAW / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["nama_file", "kelas", "tanggal", "kondisi_cahaya"])
    w.writerows(rows)

print(f"\n{len(rows)} citra masuk ke dataset_raw/ dan dataset_raw/metadata.csv\n")
print(f"{'KELAS':15s} {'JUMLAH':>6s}  SESI (tanggal, kondisi)")
per = Counter((r[1], r[2], r[3]) for r in rows)
for kelas in sorted({r[1] for r in rows}):
    n = sum(v for k, v in per.items() if k[0] == kelas)
    sesi = ", ".join(f"{k[1]}/{k[2]}={v}" for k, v in per.items() if k[0] == kelas)
    print(f"{kelas:15s} {n:6d}  {sesi}")
if salah:
    print(f"\n[DILEWATI] {len(salah)} file tidak sesuai pola, contoh: {salah[:3]}")
