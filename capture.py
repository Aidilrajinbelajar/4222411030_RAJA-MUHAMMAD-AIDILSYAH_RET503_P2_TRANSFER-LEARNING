"""Langkah 1 - Ambil citra terkoreksi (undistort) dari kamera robot.
Pemakaian : python capture.py baut terang
Tombol    : SPASI = simpan citra, Q = keluar
"""
import argparse, csv, datetime, sys
from pathlib import Path
import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("kelas")
p.add_argument("kondisi", help="mis. terang, redup, jendela, bayangan")
p.add_argument("--cam", type=int, default=0)
p.add_argument("--width", type=int, default=640)   # samakan dengan resolusi kalibrasi!
p.add_argument("--height", type=int, default=480)
p.add_argument("--calib", default="calib.npz")
p.add_argument("--out", default="dataset_raw")
a = p.parse_args()

# Muat parameter kalibrasi dari Pertemuan 2 (nama kunci dibuat toleran)
K = D = None
if Path(a.calib).exists():
    c = np.load(a.calib)
    for kk in ("K", "mtx", "camera_matrix"):
        if kk in c: K = c[kk]
    for dk in ("dist", "D", "dist_coeffs"):
        if dk in c: D = c[dk]
if K is None or D is None:
    print("[PERINGATAN] calib.npz tidak ditemukan/kunci tidak cocok -> citra TIDAK di-undistort")

cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW)       # CAP_DSHOW: backend DirectShow khusus Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
if not cap.isOpened():
    sys.exit("Kamera tidak dapat dibuka. Coba --cam 1")

tgl = datetime.date.today().strftime("%Y%m%d")
folder = Path(a.out) / a.kelas
folder.mkdir(parents=True, exist_ok=True)
meta = Path(a.out) / "metadata.csv"
if not meta.exists():
    meta.write_text("nama_file,kelas,tanggal,kondisi_cahaya\n", encoding="utf-8")

prefix = f"{a.kelas}_{tgl}_{a.kondisi}_"
n = len(list(folder.glob(prefix + "*.png")))
print(f"Mulai dari nomor {n+1:03d}. SPASI=simpan, Q=keluar")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if K is not None and D is not None:
        frame = cv2.undistort(frame, K, D)
    view = frame.copy()
    cv2.putText(view, f"{a.kelas}/{a.kondisi}  tersimpan: {n}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("capture", view)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        n += 1
        nama = f"{prefix}{n:03d}.png"
        cv2.imwrite(str(folder / nama), frame)
        with open(meta, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([nama, a.kelas, tgl, a.kondisi])
    elif k in (ord("q"), ord("Q")):
        break
cap.release(); cv2.destroyAllWindows()
