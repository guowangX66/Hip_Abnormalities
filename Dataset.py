import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def _default_hip_bboxes(h: int, w: int):
    """
    Tuned left/right hip bboxes in pascal_voc pixel coords (x1,y1,x2,y2).
    Goal: focus on acetabulum + femoral head/neck + trochanter region,
    avoid too much femoral shaft / bottom ruler / central pelvis noise.
    """
    # ---- knobs (you can tune these 3) ----
    w_ratio = 0.36          # box width / image width
    h_ratio = 0.56          # box height / image height
    y_center = 0.53         # vertical center (0~1)
    x_center_left = 0.27
    x_center_right = 0.73
    mid_gap = 0.05          # keep a gap around midline to reduce pubic symphysis/marker noise

    bw = w_ratio * w
    bh = h_ratio * h
    yc = y_center * h

    def box_from_center(xc):
        x1 = xc - bw / 2
        x2 = xc + bw / 2
        y1 = yc - bh / 2
        y2 = yc + bh / 2
        return x1, y1, x2, y2

    lx1, ly1, lx2, ly2 = box_from_center(x_center_left * w)
    rx1, ry1, rx2, ry2 = box_from_center(x_center_right * w)

    # enforce mid-gap
    mid = 0.5 * w
    gap = mid_gap * w
    lx2 = min(lx2, mid - gap)
    rx1 = max(rx1, mid + gap)

    # clip + ensure valid
    def clip(x1, y1, x2, y2):
        x1 = float(max(0.0, min(w - 1.0, x1)))
        x2 = float(max(0.0, min(w - 1.0, x2)))
        y1 = float(max(0.0, min(h - 1.0, y1)))
        y2 = float(max(0.0, min(h - 1.0, y2)))
        if x2 <= x1 + 1:
            x2 = min(w - 1.0, x1 + 2.0)
        if y2 <= y1 + 1:
            y2 = min(h - 1.0, y1 + 2.0)
        return (x1, y1, x2, y2)

    return [clip(lx1, ly1, lx2, ly2), clip(rx1, ry1, rx2, ry2)]

import random

def jitter_box(box, W, H, max_shift=0.03, max_scale=0.06):
    x1,y1,x2,y2 = box
    bw = x2-x1; bh = y2-y1
    cx = (x1+x2)/2; cy = (y1+y2)/2

    # shift
    dx = (random.uniform(-max_shift, max_shift)) * W
    dy = (random.uniform(-max_shift, max_shift)) * H

    # scale
    s = 1.0 + random.uniform(-max_scale, max_scale)
    bw2 = bw * s; bh2 = bh * s

    nx1 = cx + dx - bw2/2
    nx2 = cx + dx + bw2/2
    ny1 = cy + dy - bh2/2
    ny2 = cy + dy + bh2/2

    # clamp
    nx1 = max(0, min(W-1, nx1)); nx2 = max(0, min(W-1, nx2))
    ny1 = max(0, min(H-1, ny1)); ny2 = max(0, min(H-1, ny2))
    if nx2 <= nx1+1: nx2 = min(W-1, nx1+2)
    if ny2 <= ny1+1: ny2 = min(H-1, ny1+2)
    return [nx1, ny1, nx2, ny2]

class MultiLabelDataset_Dynamic(Dataset):
    def __init__(self, txt_file, img_dir, transform=None, label_info: Dict[str, int] = None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []
        self.label_info = label_info or {}
        self.labels = {name: [] for name in self.label_info.keys()}

        with open(txt_file, 'r') as file:
            for lineno, line in enumerate(file, start=1):
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    raise ValueError(f"[Line {lineno}] Format error: expected 'imgname \\t labels', got {line}")

                img_name = parts[0]
                label_str = parts[1].strip()
                # label format: continuous 0/1 string, e.g. 0010000010000000
                label_values = [int(x) for x in label_str]

                if len(label_values) != sum(self.label_info.values()):
                    raise ValueError(
                        f"[Line {lineno}] Label length mismatch: expected {sum(self.label_info.values())}, got {len(label_values)}"
                    )

                start = 0
                for name, length in self.label_info.items():
                    one_hot = label_values[start:start + length]
                    if sum(one_hot) != 1:
                        raise ValueError(f"[Line {lineno}] Label for '{name}' is not one-hot: {one_hot}")
                    class_index = one_hot.index(1)
                    self.labels[name].append(class_index)
                    start += length

                self.img_labels.append(img_name)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Read as RGB (gray replicated to 3-ch is fine too)
        image_np = np.array(Image.open(img_path).convert('RGB'))
        h, w = image_np.shape[:2]

        # default (left,right) boxes in original coords
        bboxes = _default_hip_bboxes(h, w)
        bbox_labels = [0, 1]

        if self.transform is not None:
            # If transform supports bboxes, it will transform boxes consistently.
            try:
                augmented = self.transform(image=image_np, bboxes=bboxes, bbox_labels=bbox_labels)
                image = augmented['image']
                bboxes = augmented.get('bboxes', bboxes)
                # Fallback if bboxes got dropped by Albumentations
                if (bboxes is None) or (len(bboxes) != 2):
                    # use current image size after transform if possible
                    if isinstance(image, torch.Tensor):
                        hh, ww = int(image.shape[1]), int(image.shape[2])
                    else:
                        hh, ww = image.shape[:2]
                    bboxes = _default_hip_bboxes(hh, ww)
            except TypeError:
                augmented = self.transform(image=image_np)
                image = augmented['image']
        else:
            image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        label_dict = {
            name: torch.tensor(self.labels[name][idx], dtype=torch.long)
            for name in self.labels.keys()
        }

        roi_boxes = torch.tensor(bboxes, dtype=torch.float32)  # (2,4)

        return {'img': image, 'labels': label_dict, 'roi_boxes': roi_boxes}
