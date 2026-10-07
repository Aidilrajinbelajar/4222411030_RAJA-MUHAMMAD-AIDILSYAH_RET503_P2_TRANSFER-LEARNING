"""Langkah 4 - Ukur latensi ResNet-18 vs MobileNetV3-Small (anggaran inferensi slide: 35 ms).
Pemakaian : python latency.py [jumlah_kelas]
"""
import json, sys, time
import cv2, numpy as np, torch, torch.nn as nn
from torchvision import models

NCLS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)       # frame BGR dari kamera

def preprocess(bgr):
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)                       # OpenCV = BGR -> RGB
    x = cv2.resize(x, (224, 224)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dev)

def sync():
    if dev.type == "cuda": torch.cuda.synchronize()

def bench(fn, n=100, warm=20):
    for _ in range(warm): fn()
    sync(); t = []
    for _ in range(n):
        s = time.perf_counter(); fn(); sync(); t.append((time.perf_counter() - s) * 1000)
    return float(np.mean(t)), float(np.std(t))

res = {"device": str(dev)}
pre_ms, _ = bench(lambda: preprocess(frame)); res["preprocess_ms"] = pre_ms
for name, ctor in (("ResNet-18", models.resnet18), ("MobileNetV3-Small", models.mobilenet_v3_small)):
    m = ctor(weights=None)                                         # latensi tidak bergantung nilai bobot
    if name == "ResNet-18": m.fc = nn.Linear(512, NCLS)
    else: m.classifier[3] = nn.Linear(m.classifier[3].in_features, NCLS)
    m = m.to(dev).eval(); x = preprocess(frame)
    with torch.inference_mode():
        mean, std = bench(lambda: m(x))
    res[name] = dict(infer_ms=mean, std_ms=std, fps_model_saja=1000 / mean,
                     pipeline_ms=mean + pre_ms, fps_pipeline=1000 / (mean + pre_ms),
                     params_juta=sum(p.numel() for p in m.parameters()) / 1e6)
    print(f"{name:18s} infer {mean:6.2f}±{std:.2f} ms | pipeline {mean+pre_ms:6.2f} ms | "
          f"{1000/(mean+pre_ms):5.1f} FPS | anggaran 35 ms: {'LOLOS' if mean <= 35 else 'TIDAK'}")
json.dump(res, open("results/latency.json", "w"), indent=2)
