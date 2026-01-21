# tools/analyze_val.py
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import roc_curve, auc

# ---- make sure import parent medical/model.py, not tools/model.py
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from model import model as model_factory  # noqa

VALID_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

# ----------------- config -----------------
CFG = {
    "val_txt": "data/medician_data/labels/val.txt",                 # 你的 val.txt
    "val_img_dir": "data/medician_data/imgs",                       # val images dir
    "train_txt_for_priors": "data/medician_data/labels/train.txt",  # 用于计算先验(可选)
    "ckpt_path": "checkpoints/epoch088.pth",
    "model_name": "resnet50",
    "class_info": {"label_left_hip": 8, "label_right_hip": 8},
    "img_size": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "./results/val",
    # ensemble
    "tta_n": 4,
    "roi_jitter_n": 5,
    # logit adjust
    "do_logit_adjust": True,
    "tau_list": [0.5, 1.0, 1.5, 2.0],     # sweep tau
    "select_by": "macro_f1",              # "macro_f1" or "balanced_acc"
}

# ----------------- label parsing -----------------
def parse_txt_labels(txt_path):
    entries = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            img_name = parts[0].strip()
            stem = os.path.splitext(img_name)[0]
            label_str = parts[1].strip()
            onehot = [int(ch) for ch in label_str]
            entries.append((stem, onehot))
    return entries

def decode(onehot, task):
    offset = 0 if task == "label_left_hip" else 8
    return int(np.argmax(onehot[offset:offset+8]))

def find_img(img_dir, stem):
    for ext in VALID_EXTS:
        p = os.path.join(img_dir, f"{stem}{ext}")
        if os.path.exists(p):
            return p
    return None

# ----------------- ROI (match medical/dataset.py tuned ROI) -----------------
def default_hip_rois(img_size: int) -> torch.Tensor:
    H = W = img_size
    w_ratio = 0.36
    h_ratio = 0.56
    y_center = 0.53
    x_center_left = 0.27
    x_center_right = 0.73
    mid_gap = 0.05

    bw = w_ratio * W
    bh = h_ratio * H
    yc = y_center * H

    def box_from_center(xc):
        x1 = xc - bw / 2
        x2 = xc + bw / 2
        y1 = yc - bh / 2
        y2 = yc + bh / 2
        return x1, y1, x2, y2

    lx1, ly1, lx2, ly2 = box_from_center(x_center_left * W)
    rx1, ry1, rx2, ry2 = box_from_center(x_center_right * W)

    mid = 0.5 * W
    gap = mid_gap * W
    lx2 = min(lx2, mid - gap)
    rx1 = max(rx1, mid + gap)

    def clip(x1, y1, x2, y2):
        x1 = float(max(0.0, min(W - 1.0, x1)))
        x2 = float(max(0.0, min(W - 1.0, x2)))
        y1 = float(max(0.0, min(H - 1.0, y1)))
        y2 = float(max(0.0, min(H - 1.0, y2)))
        if x2 <= x1 + 1:
            x2 = min(W - 1.0, x1 + 2.0)
        if y2 <= y1 + 1:
            y2 = min(H - 1.0, y1 + 2.0)
        return [x1, y1, x2, y2]

    boxes = [clip(lx1, ly1, lx2, ly2), clip(rx1, ry1, rx2, ry2)]
    return torch.tensor(boxes, dtype=torch.float32).unsqueeze(0)  # (1,2,4)

# ----------------- priors + logit adjust -----------------
def compute_priors_from_train_txt(train_txt: str, n_cls=8, eps=1e-12):
    left = np.zeros(n_cls, dtype=np.float64)
    right = np.zeros(n_cls, dtype=np.float64)
    with open(train_txt, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            s = parts[1].strip()
            if len(s) < n_cls * 2:
                continue
            y = np.array([int(ch) for ch in s[:n_cls*2]], dtype=np.int64)
            l = int(np.argmax(y[:n_cls]))
            r = int(np.argmax(y[n_cls:n_cls*2]))
            left[l] += 1
            right[r] += 1
    left = (left + eps) / (left.sum() + n_cls*eps)
    right = (right + eps) / (right.sum() + n_cls*eps)
    return left.astype(np.float32), right.astype(np.float32)

def apply_logit_adjust_np(logits_np: np.ndarray, prior_np: np.ndarray, tau: float):
    return logits_np + tau * np.log(np.clip(prior_np, 1e-12, 1.0))

# ----------------- TTA + ROI jitter ensemble -----------------
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def intensity_tta_4(x_norm: torch.Tensor):
    mean = _MEAN.to(x_norm.device, x_norm.dtype)
    std  = _STD.to(x_norm.device, x_norm.dtype)
    x = (x_norm * std + mean).clamp(0, 1)

    outs = []
    outs.append((x - mean) / std)  # identity
    x2 = x.clamp_min(1e-6) ** 0.90
    outs.append((x2 - mean) / std)
    x3 = x.clamp_min(1e-6) ** 1.10
    outs.append((x3 - mean) / std)
    x4 = ((x - 0.5) * 1.08 + 0.5 + 0.02).clamp(0, 1)
    outs.append((x4 - mean) / std)
    return outs

def jitter_rois_batch(roi_boxes: torch.Tensor, img_h: int, img_w: int, n=5, max_shift=0.03, max_scale=0.06):
    device = roi_boxes.device
    dtype = roi_boxes.dtype
    B = roi_boxes.shape[0]
    rois = roi_boxes.unsqueeze(1).repeat(1, n, 1, 1)  # (B,n,2,4)

    x1 = rois[..., 0]; y1 = rois[..., 1]; x2 = rois[..., 2]; y2 = rois[..., 3]
    bw = (x2 - x1).clamp_min(2.0)
    bh = (y2 - y1).clamp_min(2.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    dx = (torch.rand(B, n, 2, device=device, dtype=dtype) * 2 - 1) * (max_shift * img_w)
    dy = (torch.rand(B, n, 2, device=device, dtype=dtype) * 2 - 1) * (max_shift * img_h)
    s  = 1.0 + (torch.rand(B, n, 2, device=device, dtype=dtype) * 2 - 1) * max_scale

    bw2 = bw * s; bh2 = bh * s
    nx1 = cx + dx - bw2/2; nx2 = cx + dx + bw2/2
    ny1 = cy + dy - bh2/2; ny2 = cy + dy + bh2/2

    nx1 = nx1.clamp(0, img_w - 1); nx2 = nx2.clamp(0, img_w - 1)
    ny1 = ny1.clamp(0, img_h - 1); ny2 = ny2.clamp(0, img_h - 1)
    nx2 = torch.max(nx2, nx1 + 2.0)
    ny2 = torch.max(ny2, ny1 + 2.0)

    out = torch.stack([nx1, ny1, nx2, ny2], dim=-1)  # (B,n,2,4)
    return out.view(B*n, 2, 4)

@torch.no_grad()
def forward_tta_roiensemble_logits(net, img, roi_boxes, tta_n=4, jitter_n=5):
    B, _, H, W = img.shape
    tta_imgs = intensity_tta_4(img) if tta_n == 4 else [img]

    logits_sum = None
    for x in tta_imgs:
        if roi_boxes is None or jitter_n <= 1:
            out = net(x, roi_boxes=roi_boxes)
        else:
            roi_rep = jitter_rois_batch(roi_boxes, img_h=H, img_w=W, n=jitter_n)
            x_rep = x.repeat_interleave(jitter_n, dim=0)
            out_rep = net(x_rep, roi_boxes=roi_rep)
            out = {k: v.view(B, jitter_n, -1).mean(dim=1) for k, v in out_rep.items()}

        if logits_sum is None:
            logits_sum = {k: v.clone() for k, v in out.items()}
        else:
            for k in logits_sum:
                logits_sum[k] += out[k]

    return {k: v / len(tta_imgs) for k, v in logits_sum.items()}

# ----------------- ROC plot -----------------
def plot_roc(y_true, y_prob, n_cls, save_path, title):
    fpr, tpr, auc_all = {}, {}, {}
    for i in range(n_cls):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        auc_all[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_cls)]))
    mean_tpr = np.mean([np.interp(all_fpr, fpr[i], tpr[i]) for i in range(n_cls)], axis=0)
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure()
    for i in range(n_cls):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} AUC={auc_all[i]:.2f}")
    plt.plot(all_fpr, mean_tpr, "k--", label=f"Macro-AUC={macro_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# ----------------- checkpoint loader -----------------
def load_ckpt_flexible(net, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    missing, unexpected = net.load_state_dict(ckpt, strict=False)
    print(f"[ckpt] loaded: {ckpt_path}")
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

# ----------------- main eval -----------------
def main():
    os.makedirs(CFG["save_dir"], exist_ok=True)
    device = torch.device(CFG["device"])

    # model
    net = model_factory(CFG["model_name"], (1, 3, CFG["img_size"], CFG["img_size"]), CFG["class_info"], pretrained_path=None, class_weights=None)
    load_ckpt_flexible(net, CFG["ckpt_path"])
    net.to(device).eval()

    # data
    entries = parse_txt_labels(CFG["val_txt"])

    tfm = transforms.Compose([
        transforms.Resize((CFG["img_size"], CFG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    base_roi = default_hip_rois(CFG["img_size"]).to(device)  # (1,2,4)

    # priors
    prior_left = prior_right = None
    if CFG["do_logit_adjust"]:
        prior_left, prior_right = compute_priors_from_train_txt(CFG["train_txt_for_priors"])

    # collect logits/gt
    logits_all = {t: [] for t in CFG["class_info"]}
    gt_all = {t: [] for t in CFG["class_info"]}
    stems_all = []

    for stem, onehot in tqdm(entries, desc="[val] collect logits"):
        p = find_img(CFG["val_img_dir"], stem)
        if p is None:
            continue
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)  # (1,3,H,W)
        out = forward_tta_roiensemble_logits(net, x, base_roi, tta_n=CFG["tta_n"], jitter_n=CFG["roi_jitter_n"])
        for task in CFG["class_info"]:
            logits_all[task].append(out[task].detach().cpu().numpy())  # (1,8)
            gt_all[task].append(decode(onehot, task))
        stems_all.append(stem)

    for task in CFG["class_info"]:
        logits = np.vstack(logits_all[task])  # (N,8)
        gt = np.array(gt_all[task], dtype=np.int64)

        # tau sweep (optional)
        if CFG["do_logit_adjust"]:
            prior = prior_left if task == "label_left_hip" else prior_right

            best_tau = None
            best_score = -1e9
            best_probs = None
            best_pred = None

            for tau in CFG["tau_list"]:
                adj = apply_logit_adjust_np(logits, prior, float(tau))
                probs = torch.softmax(torch.from_numpy(adj), dim=1).numpy()
                pred = probs.argmax(axis=1)

                # select score
                from sklearn.metrics import balanced_accuracy_score, f1_score
                bacc = balanced_accuracy_score(gt, pred)
                mf1 = f1_score(gt, pred, average="macro", zero_division=0)
                score = mf1 if CFG["select_by"] == "macro_f1" else bacc

                if score > best_score:
                    best_score = score
                    best_tau = float(tau)
                    best_probs = probs
                    best_pred = pred

            print(f"[{task}] best_tau={best_tau:.2f} best_{CFG['select_by']}={best_score:.4f}")
            probs = best_probs
            pred = best_pred
            tau_used = best_tau
        else:
            probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            pred = probs.argmax(axis=1)
            tau_used = None

        # save excel
        rows = []
        for i, stem in enumerate(stems_all):
            r = {"Sequence": stem, "gt": int(gt[i]), "pred": int(pred[i])}
            for c in range(8):
                r[f"prob_{c}"] = float(probs[i, c])
            rows.append(r)
        df = pd.DataFrame(rows)
        out_xlsx = os.path.join(CFG["save_dir"], f"{task}_results_tau{tau_used}.xlsx")
        df.to_excel(out_xlsx, index=False)

        # roc
        out_png = os.path.join(CFG["save_dir"], f"{task}_roc_tau{tau_used}.png")
        plot_roc(gt, probs, 8, out_png, title=f"{task} ROC (tau={tau_used})")

    print("[done] saved to:", CFG["save_dir"])

if __name__ == "__main__":
    main()
