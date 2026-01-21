# tools/analyze_gradcam.py
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

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from model import model as model_factory  # noqa

label_map = {'正常': 0,'粗隆间骨折': 1,'置换术后': 2,'股骨头坏死': 3,'股骨颈骨折': 4,'骨关节炎': 5,'螺钉内固定': 6,'DDH': 7}
inv_label_map = {v: k for k, v in label_map.items()}
VALID_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

CFG = {
    "test_xlsx": "data/medician_data/test/mylabel.xlsx",
    "test_img_dir": "data/medician_data/test/imgs",
    "train_txt_for_priors": "data/medician_data/labels/train.txt",
    "ckpt_path": "checkpoints/epoch088.pth",
    "model_name": "resnet50",
    "class_info": {"label_left_hip": 8, "label_right_hip": 8},
    "img_size": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "./results/gradcam",
    # logit adjust only for choosing pred class
    "do_logit_adjust": True,
    "tau_left": 0.5,
    "tau_right": 0.5,
    # how many per class to save (avoid huge output)
    "max_per_class_per_task": 1000,
}

def parse_excel(excel_path):
    df = pd.read_excel(excel_path)
    entries = []
    for _, row in df.iterrows():
        stem = str(int(row["Sequence"]))
        left = label_map.get(row["左髋"], -1)
        right = label_map.get(row["右髋"], -1)
        if left >= 0 and right >= 0:
            entries.append((stem, left, right))
    return entries

def find_img(img_dir, stem):
    for ext in VALID_EXTS:
        p = os.path.join(img_dir, f"{stem}{ext}")
        if os.path.exists(p):
            return p
    return None

def default_hip_rois(img_size: int) -> torch.Tensor:
    H = W = img_size
    w_ratio = 0.36; h_ratio = 0.56; y_center = 0.53
    x_center_left = 0.27; x_center_right = 0.73; mid_gap = 0.05
    bw = w_ratio * W; bh = h_ratio * H; yc = y_center * H

    def box_from_center(xc):
        x1 = xc - bw/2; x2 = xc + bw/2
        y1 = yc - bh/2; y2 = yc + bh/2
        return x1,y1,x2,y2

    lx1,ly1,lx2,ly2 = box_from_center(x_center_left * W)
    rx1,ry1,rx2,ry2 = box_from_center(x_center_right * W)
    mid = 0.5 * W; gap = mid_gap * W
    lx2 = min(lx2, mid - gap); rx1 = max(rx1, mid + gap)

    def clip(x1,y1,x2,y2):
        x1 = float(max(0.0, min(W-1.0, x1))); x2 = float(max(0.0, min(W-1.0, x2)))
        y1 = float(max(0.0, min(H-1.0, y1))); y2 = float(max(0.0, min(H-1.0, y2)))
        if x2 <= x1+1: x2 = min(W-1.0, x1+2.0)
        if y2 <= y1+1: y2 = min(H-1.0, y1+2.0)
        return [x1,y1,x2,y2]
    boxes = [clip(lx1,ly1,lx2,ly2), clip(rx1,ry1,rx2,ry2)]
    return torch.tensor(boxes, dtype=torch.float32).unsqueeze(0)

def compute_priors_from_train_txt(train_txt: str, n_cls=8, eps=1e-12):
    left = np.zeros(n_cls, dtype=np.float64)
    right = np.zeros(n_cls, dtype=np.float64)
    with open(train_txt, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2: continue
            s = parts[1].strip()
            if len(s) < n_cls*2: continue
            y = np.array([int(ch) for ch in s[:n_cls*2]], dtype=np.int64)
            left[int(np.argmax(y[:n_cls]))] += 1
            right[int(np.argmax(y[n_cls:n_cls*2]))] += 1
    left = (left + eps) / (left.sum() + n_cls*eps)
    right = (right + eps) / (right.sum() + n_cls*eps)
    return left.astype(np.float32), right.astype(np.float32)

def logit_adjust_torch(logits: torch.Tensor, prior_np: np.ndarray, tau: float, eps=1e-12):
    prior = torch.as_tensor(prior_np, device=logits.device, dtype=logits.dtype)
    return logits + tau * torch.log(prior.clamp_min(eps))

def load_ckpt_flexible(net, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
        elif "model" in ckpt: ckpt = ckpt["model"]
    ckpt = {k.replace("module.",""): v for k,v in ckpt.items()}
    net.load_state_dict(ckpt, strict=False)

def default_target_layer(net):
    # For torchvision resnet backbone = [conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4]
    # layer4 is backbone[-1], pick last bottleneck block backbone[-1][-1]
    try:
        return net.backbone[-1][-1]
    except Exception:
        return net.backbone[-1]

class RoiTaskWrapper(torch.nn.Module):
    def __init__(self, net, task: str, roi_boxes: torch.Tensor):
        super().__init__()
        self.net = net
        self.task = task
        self.register_buffer("roi", roi_boxes)

    def forward(self, x):
        B = x.shape[0]
        roi = self.roi.repeat(B, 1, 1)  # (B,2,4)
        out = self.net(x, roi_boxes=roi)[self.task]
        return out

def main():
    os.makedirs(CFG["output_dir"], exist_ok=True)
    device = torch.device(CFG["device"])

    net = model_factory(CFG["model_name"], (1,3,CFG["img_size"],CFG["img_size"]), CFG["class_info"], pretrained_path=None, class_weights=None)
    load_ckpt_flexible(net, CFG["ckpt_path"])
    net.to(device).eval()

    prior_left = prior_right = None
    if CFG["do_logit_adjust"]:
        prior_left, prior_right = compute_priors_from_train_txt(CFG["train_txt_for_priors"])

    tfm = transforms.Compose([
        transforms.Resize((CFG["img_size"], CFG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    roi = default_hip_rois(CFG["img_size"]).to(device)
    entries = parse_excel(CFG["test_xlsx"])

    for task in CFG["class_info"]:
        save_count = {c: 0 for c in range(8)}
        wrapped = RoiTaskWrapper(net, task, roi).to(device).eval()
        target_layer = default_target_layer(net)
        cam = GradCAM(model=wrapped, target_layers=[target_layer])

        for stem, yL, yR in tqdm(entries, desc=f"[{task}] gradcam"):
            gt = yL if task == "label_left_hip" else yR
            if save_count[gt] >= CFG["max_per_class_per_task"]:
                continue

            img_path = find_img(CFG["test_img_dir"], stem)
            if img_path is None:
                continue

            img = Image.open(img_path).convert("RGB")
            raw = np.array(img.resize((CFG["img_size"], CFG["img_size"]))).astype(np.float32) / 255.0
            x = tfm(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = wrapped(x)  # (1,8)
                if CFG["do_logit_adjust"]:
                    prior = prior_left if task == "label_left_hip" else prior_right
                    tau = CFG["tau_left"] if task == "label_left_hip" else CFG["tau_right"]
                    logits = logit_adjust_torch(logits, prior, float(tau))
                prob = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                pred = int(prob.argmax())
                correct = (pred == int(gt))

            # cam target = predicted class (or you can set to gt class)
            targets = [ClassifierOutputTarget(pred)]
            cam_map = cam(input_tensor=x, targets=targets)[0]  # (H,W) in resized space

            vis = show_cam_on_image(raw, cam_map, use_rgb=True)

            subdir = "correct" if correct else "wrong"
            cls_name = inv_label_map[int(gt)]
            out_dir = os.path.join(CFG["output_dir"], task, f"class_{cls_name}", subdir)
            os.makedirs(out_dir, exist_ok=True)

            base = f"{stem}_gt{gt}_pred{pred}"
            plt.imsave(os.path.join(out_dir, base + "_raw.jpg"), raw)
            plt.imsave(os.path.join(out_dir, base + "_cam.jpg"), vis)

            save_count[int(gt)] += 1

    print("[done] saved:", CFG["output_dir"])

if __name__ == "__main__":
    main()
