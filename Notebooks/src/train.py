#============================================================#
#                        src/train.py                        #
#============================================================#
# Author      : Aktham Almomani                              #
# Created     : 2025-10-01                                   #
# Version     : V1.0.0                                       #
#============================================================#


import zipfile, argparse
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def unzip_all(channel: Path):
    for z in channel.glob("*.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(channel)

def pick_root(channel: Path):
    for name in ["train", "val", "validation"]:
        p = channel / name
        if p.exists():
            return p
    return channel

def main(args):
    train_chan = Path("/opt/ml/input/data/train")
    val_chan   = Path("/opt/ml/input/data/validation")
    unzip_all(train_chan); unzip_all(val_chan)
    train_root = pick_root(train_chan); val_root = pick_root(val_chan)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    train_ds = datasets.ImageFolder(train_root, transform=tfm)
    val_ds   = datasets.ImageFolder(val_root,   transform=tfm)

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item(); total += y.numel()
        print(f"epoch {epoch+1}: val_acc={correct/total:.4f}")

    out = Path("/opt/ml/model"); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs",     type=int, default=32)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=256)
    main(ap.parse_args())
