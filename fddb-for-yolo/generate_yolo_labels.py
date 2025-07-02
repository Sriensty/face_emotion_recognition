import os
from pathlib import Path
from PIL import Image

BASE = Path('/home/sriensty/face_training/data/fbbd_data/fddb/FDDB-folds/')
IMG_OUT = BASE / 'train/images'
LBL_OUT = BASE / 'train/labels'

IMG_OUT.mkdir(parents=True, exist_ok=True)
LBL_OUT.mkdir(parents=True, exist_ok=True)

with open(BASE / 'FDDB-fold-09-rectList.txt') as f:
    for line in f:
        img_path, x1, y1, x2, y2 = line.strip().split(',')
        img_path = Path(img_path)
        dst_img = IMG_OUT / img_path.name
        if not dst_img.exists():
            dst_img.write_bytes(img_path.read_bytes())

        w, h = Image.open(img_path).size
        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        label_path = LBL_OUT / (img_path.stem + '.txt')
        with open(label_path, 'a') as lf:
            lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
print(f"Done!")