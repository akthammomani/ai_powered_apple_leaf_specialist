#============================================================#
#                     src/evaluate.py                        #
#============================================================#
# Author      : Aktham Almomani                              #
# Created     : 2025-10-01                                   #
# Version     : V1.0.0                                       #
#============================================================#


import os, io, json, tarfile, zipfile
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

MODEL_DIR = Path("/opt/ml/processing/model")      # where the model.tar.gz lands
VAL_DIR   = Path("/opt/ml/processing/val")        # where val.zip lands
OUT_DIR   = Path("/opt/ml/processing/output")     # must match ProcessingOutput(source=...)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def untar_first_in(dirpath: Path):
    """If a *.tar.gz is present in dirpath, extract it there."""
    tars = list(dirpath.glob("*.tar.gz"))
    if not tars:
        return
    with tarfile.open(tars[0]) as t:
        t.extractall(path=dirpath)

def unzip_first_in(dirpath: Path) -> Path:
    """If a *.zip is present, extract to dirpath/<stem> and return that root."""
    zips = list(dirpath.glob("*.zip"))
    if not zips:
        return dirpath
    dest = dirpath / zips[0].stem
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(dest)
    return dest

def build_model(num_classes: int):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def main():
    # unpack model.tar.gz and val.zip:
    untar_first_in(MODEL_DIR)
    val_root = unzip_first_in(VAL_DIR)

    # find model.pt inside the extracted model folder:
    # training saved /opt/ml/model/model.pt -> tarball keeps it at top level:
    ckpt = next((p for p in MODEL_DIR.rglob("model.pt")), None)
    if ckpt is None:
        raise FileNotFoundError("model.pt not found inside model artifact")

    # dataset & transforms (same as training: Resize + ToTensor):
    IMG_SIZE = 256
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # ImageFolder expects class subfolders; our val zip already has /val/<class>/*:
    ds_root = (val_root / "val") if (val_root / "val").exists() else val_root
    val_ds = datasets.ImageFolder(ds_root, transform=tfm)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # build model -> load weights:
    model = build_model(num_classes=len(val_ds.classes))
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # evaluate accuracy:
    correct = total = 0
    with torch.inference_mode():
        for x, y in val_dl:
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.numel()

    val_acc = float(correct / max(1, total))

    # write metrics.json for the ProcessingStep PropertyFile to parse:
    out = {"val_accuracy": val_acc}
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(out, f)
    print("Wrote metrics.json:", out)

if __name__ == "__main__":
    main()
