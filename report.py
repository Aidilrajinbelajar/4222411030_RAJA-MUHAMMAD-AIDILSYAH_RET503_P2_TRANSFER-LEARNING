"""Gabungkan hasil 3 mode -> results/tabel_hasil.md dan results/grafik_akurasi.png"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = {m: json.load(open(f"results/{m}.json")) for m in ("feature", "partial", "scratch")
     if Path(f"results/{m}.json").exists()}
lines = ["| Mode | Akurasi val terbaik | Waktu pelatihan (s) | Epoch akurasi >= 90% |", "|---|---|---|---|"]
for m, r in R.items():
    lines.append(f"| {m} | {r['best_val_acc']*100:.1f}% | {r['train_time_s']:.0f} | {r['epoch_acc90'] or '-'} |")
Path("results/tabel_hasil.md").write_text("\n".join(lines), encoding="utf-8")

plt.figure(figsize=(7, 4))
for m, r in R.items():
    plt.plot([h["epoch"] for h in r["history"]], [h["val_acc"] for h in r["history"]], marker="o", label=m)
plt.xlabel("Epoch"); plt.ylabel("Akurasi validasi"); plt.ylim(0, 1.05); plt.grid(alpha=.3); plt.legend()
plt.title("Akurasi validasi per epoch - ResNet-18")
plt.tight_layout(); plt.savefig("results/grafik_akurasi.png", dpi=150)
print("\n".join(lines))
