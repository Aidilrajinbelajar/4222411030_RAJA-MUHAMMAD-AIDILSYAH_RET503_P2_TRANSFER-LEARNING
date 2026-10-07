"""Langkah 3 - Latih ResNet-18 dengan 3 pendekatan: feature | partial | scratch.
Pemakaian : python train.py feature   (atau partial / scratch)
"""
import json, sys, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms as T

MODE = sys.argv[1] if len(sys.argv) > 1 else "feature"
assert MODE in ("feature", "partial", "scratch")
EPOCHS, BS, SIZE = 10, 16, 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build(mode, n_cls):
    if mode == "scratch":
        m = models.resnet18(weights=None)                              # bobot acak
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for p in m.parameters():
            p.requires_grad = False                                    # bekukan backbone
        if mode == "partial":
            for p in m.layer4.parameters():
                p.requires_grad = True                                 # buka blok terakhir
    m.fc = nn.Linear(m.fc.in_features, n_cls)                          # head baru (requires_grad=True)
    return m


def optimizer(m, mode):
    if mode == "feature":
        return torch.optim.Adam(m.fc.parameters(), lr=1e-3)
    if mode == "partial":
        return torch.optim.Adam([{"params": m.layer4.parameters(), "lr": 1e-4},
                                 {"params": m.fc.parameters(), "lr": 1e-3}])
    return torch.optim.Adam(m.parameters(), lr=1e-3)


def set_train(m, mode):
    """Lapisan beku tetap eval() agar statistik BatchNorm ImageNet tidak berubah."""
    m.train()
    if mode == "feature":
        for name, ch in m.named_children():
            if name != "fc": ch.eval()
    elif mode == "partial":
        for name in ("conv1", "bn1", "layer1", "layer2", "layer3"):
            getattr(m, name).eval()


def evaluate(m, loader, dev):
    m.eval(); ok = n = 0; loss = 0.0; crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev); o = m(x)
            loss += crit(o, y).item() * len(y); ok += (o.argmax(1) == y).sum().item(); n += len(y)
    return loss / n, ok / n


def main():
    torch.manual_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_tf = T.Compose([T.RandomResizedCrop(SIZE), T.RandomHorizontalFlip(),
                       T.ColorJitter(0.3, 0.3, 0.3), T.ToTensor(), T.Normalize(MEAN, STD)])
    va_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
    tr = datasets.ImageFolder("dataset/train", tr_tf)
    va = datasets.ImageFolder("dataset/val", va_tf)
    # num_workers=0 paling aman di Windows (menghindari error multiprocessing spawn)
    tl = DataLoader(tr, BS, shuffle=True, num_workers=0)
    vl = DataLoader(va, BS, num_workers=0)

    m = build(MODE, len(tr.classes)).to(dev)
    opt = optimizer(m, MODE)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    print(f"mode={MODE} device={dev} kelas={tr.classes} "
          f"parameter dilatih={sum(p.numel() for p in m.parameters() if p.requires_grad):,}")

    hist, best, ep90, t0 = [], 0.0, None, time.time()
    Path("models").mkdir(exist_ok=True)
    for ep in range(1, EPOCHS + 1):
        set_train(m, MODE); tot = ok = n = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); o = m(x); l = crit(o, y); l.backward(); opt.step()
            tot += l.item() * len(y); ok += (o.argmax(1) == y).sum().item(); n += len(y)
        sch.step()
        vloss, vacc = evaluate(m, vl, dev)
        hist.append(dict(epoch=ep, train_loss=tot / n, train_acc=ok / n, val_loss=vloss, val_acc=vacc))
        if vacc > best:
            best = vacc; torch.save(m.state_dict(), f"models/{MODE}_best.pt")
        if ep90 is None and vacc >= 0.90: ep90 = ep
        print(f"ep {ep:2d} | train acc {ok/n:.3f} | val acc {vacc:.3f} | val loss {vloss:.3f}")
        if ep <= 2 and vacc == 1.0:
            print("  [CURIGA] val 100% dalam <=2 epoch -> periksa data leakage (lihat split.py)")

    out = dict(mode=MODE, best_val_acc=best, train_time_s=time.time() - t0,
               epoch_acc90=ep90, history=hist)
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open(f"results/{MODE}.json", "w"), indent=2)
    print(f"Selesai: best val acc={best:.3f}, waktu={out['train_time_s']:.0f}s, epoch>=90%={ep90}")


if __name__ == "__main__":      # wajib di Windows
    main()
