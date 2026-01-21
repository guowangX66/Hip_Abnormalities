import os
import warnings
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# -----------------------------
# Distributed helpers
# -----------------------------
def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def is_main_process():
    return (not is_dist()) or (get_rank() == 0)

def reduce_tensor(tensor, average=True):
    if not is_dist():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= get_world_size()
    return rt

def all_gather_object(obj):
    """Gather arbitrary python objects from all ranks."""
    if not is_dist():
        return [obj]
    out = [None for _ in range(get_world_size())]
    dist.all_gather_object(out, obj)
    return out

def broadcast_object(obj):
    """Broadcast python object from rank0 to all ranks."""
    if not is_dist():
        return obj
    obj_list = [obj] if is_main_process() else [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def checkpoint_load(model, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if is_main_process():
        print(f"Restoring checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(checkpoint_path))[0].split('-')[1])
    return epoch


# -----------------------------
# Priors + Logit adjustment
# -----------------------------
def compute_priors_from_train_txt(train_txt: str, n_cls=8, eps=1e-12):
    """
    Parse lines like: xxx.jpg \\t 0010000010000000
    Return priors for left/right tasks, shape (8,) each.
    """
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
            y = np.array([int(ch) for ch in s[:n_cls * 2]], dtype=np.int64)  # len=16
            l = int(np.argmax(y[:n_cls]))
            r = int(np.argmax(y[n_cls:n_cls * 2]))
            left[l] += 1
            right[r] += 1

    left = (left + eps) / (left.sum() + n_cls * eps)
    right = (right + eps) / (right.sum() + n_cls * eps)
    return left.astype(np.float32), right.astype(np.float32)

def apply_logit_adjust_np(logits_np: np.ndarray, prior_np: np.ndarray, tau: float):
    return logits_np + tau * np.log(np.clip(prior_np, 1e-12, 1.0))


# -----------------------------
# Safe intensity TTA (N=4)
# -----------------------------
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def intensity_tta_4(x_norm: torch.Tensor):
    """
    x_norm: (B,3,H,W) normalized by ImageNet mean/std.
    Return list of 4 normalized tensors.
    """
    mean = _MEAN.to(x_norm.device, x_norm.dtype)
    std  = _STD.to(x_norm.device, x_norm.dtype)

    x = (x_norm * std + mean).clamp(0, 1)

    outs = []
    # 1) identity
    outs.append((x - mean) / std)
    # 2) gamma 0.90
    x2 = x.clamp_min(1e-6) ** 0.90
    outs.append((x2 - mean) / std)
    # 3) gamma 1.10
    x3 = x.clamp_min(1e-6) ** 1.10
    outs.append((x3 - mean) / std)
    # 4) mild brightness+contrast
    x4 = ((x - 0.5) * 1.08 + 0.5 + 0.02).clamp(0, 1)
    outs.append((x4 - mean) / std)

    return outs


# -----------------------------
# ROI jitter ensemble (N=5)
# -----------------------------
def jitter_rois_batch(roi_boxes: torch.Tensor, img_h: int, img_w: int,
                      n: int = 5, max_shift: float = 0.03, max_scale: float = 0.06):
    """
    roi_boxes: (B,2,4) xyxy in pixels
    return: (B*n,2,4)
    """
    device = roi_boxes.device
    dtype = roi_boxes.dtype
    B = roi_boxes.shape[0]

    rois = roi_boxes.unsqueeze(1).repeat(1, n, 1, 1)  # (B,n,2,4)
    x1 = rois[..., 0]
    y1 = rois[..., 1]
    x2 = rois[..., 2]
    y2 = rois[..., 3]

    bw = (x2 - x1).clamp_min(2.0)
    bh = (y2 - y1).clamp_min(2.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # random shift per box
    dx = (torch.rand(B, n, 2, device=device, dtype=dtype) * 2 - 1) * (max_shift * img_w)
    dy = (torch.rand(B, n, 2, device=device, dtype=dtype) * 2 - 1) * (max_shift * img_h)

    # random scale per box
    s = 1.0 + (torch.rand(B, n, 2, device=device, dtype=dtype) * 2 - 1) * max_scale

    bw2 = bw * s
    bh2 = bh * s

    nx1 = cx + dx - bw2 / 2.0
    nx2 = cx + dx + bw2 / 2.0
    ny1 = cy + dy - bh2 / 2.0
    ny2 = cy + dy + bh2 / 2.0

    # clamp
    nx1 = nx1.clamp(0, img_w - 1)
    nx2 = nx2.clamp(0, img_w - 1)
    ny1 = ny1.clamp(0, img_h - 1)
    ny2 = ny2.clamp(0, img_h - 1)

    # ensure valid
    nx2 = torch.max(nx2, nx1 + 2.0)
    ny2 = torch.max(ny2, ny1 + 2.0)

    out = torch.stack([nx1, ny1, nx2, ny2], dim=-1)  # (B,n,2,4)
    return out.view(B * n, 2, 4)


@torch.no_grad()
def forward_tta_roiensemble_logits(model, img, roi_boxes, tta_n=4, jitter_n=5):
    """
    img: (B,3,H,W) normalized tensor
    roi_boxes: (B,2,4) or None
    return: dict task -> logits_avg (B,C)
    """
    B, _, H, W = img.shape
    if tta_n == 4:
        tta_imgs = intensity_tta_4(img)
    else:
        tta_imgs = [img]

    logits_sum = None

    for x in tta_imgs:
        if roi_boxes is None or jitter_n <= 1:
            out = model(x, roi_boxes=roi_boxes)
        else:
            roi_rep = jitter_rois_batch(roi_boxes, img_h=H, img_w=W, n=jitter_n)
            x_rep = x.repeat_interleave(jitter_n, dim=0)  # (B*jitter_n,3,H,W)
            out_rep = model(x_rep, roi_boxes=roi_rep)
            out = {}
            for k, v in out_rep.items():  # v: (B*jitter_n, C)
                C = v.shape[1]
                out[k] = v.view(B, jitter_n, C).mean(dim=1)

        if logits_sum is None:
            logits_sum = {k: v.clone() for k, v in out.items()}
        else:
            for k in logits_sum.keys():
                logits_sum[k] += out[k]

    tta_count = len(tta_imgs)
    logits_avg = {k: v / tta_count for k, v in logits_sum.items()}
    return logits_avg


# -----------------------------
# Metrics helpers (dataset-level)
# -----------------------------
def specificity_from_cm(cm: np.ndarray):
    tn_fp = cm.sum(axis=0) - np.diag(cm)
    fn_tp = cm.sum(axis=1) - np.diag(cm)
    tn = cm.sum() - (tn_fp + fn_tp + np.diag(cm))
    spec = np.mean(tn / (tn + tn_fp + 1e-8))
    return float(spec)

def compute_metrics_from_logits(logits: np.ndarray, gt: np.ndarray):
    pred = logits.argmax(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc = balanced_accuracy_score(gt, pred)
        prec = precision_score(gt, pred, average="macro", zero_division=0)
        rec = recall_score(gt, pred, average="macro", zero_division=0)
        f1 = f1_score(gt, pred, average="macro", zero_division=0)
        cm = confusion_matrix(gt, pred, labels=np.arange(logits.shape[1]))
        spec = specificity_from_cm(cm)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(spec)
    }

def compute_auc_from_logits(logits: np.ndarray, gt: np.ndarray):
    try:
        if len(np.unique(gt)) < 2:
            return 0.0
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        gt_onehot = np.eye(probs.shape[1])[gt]
        return float(roc_auc_score(gt_onehot, probs, average="macro", multi_class="ovr"))
    except Exception:
        return 0.0

def sweep_tau_for_task(logits: np.ndarray, gt: np.ndarray, prior: np.ndarray, tau_list):
    best = None
    best_tau = None
    sweep = {}

    for tau in tau_list:
        adj = apply_logit_adjust_np(logits, prior, tau)
        m = compute_metrics_from_logits(adj, gt)
        m["auc"] = compute_auc_from_logits(adj, gt)
        sweep[float(tau)] = m

        # choose by balanced accuracy (your code called it accuracy but actually balanced_acc)
        score = m["accuracy"]
        if (best is None) or (score > best):
            best = score
            best_tau = float(tau)

    best_metrics = sweep[best_tau].copy()
    best_metrics["best_tau"] = best_tau
    best_metrics["tau_sweep"] = sweep
    return best_metrics


# -----------------------------
# Main validate (TTA + ROI jitter + tau sweep)
# -----------------------------
@torch.no_grad()
def validate_Dynamic(
    model,
    dataloader,
    logger,
    iteration,
    device,
    checkpoint=None,
    label_names=None,
    train_txt_for_priors=None,        # <-- REQUIRED for logit adjustment
    tau_list=(0.5, 1.0, 1.5, 2.0),    # <-- sweep values
    tta_n=4,
    roi_jitter_n=5
):
    if checkpoint is not None:
        if is_main_process():
            print(f"Loading checkpoint: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    if not label_names:
        raise ValueError("label_names must be provided.")
    if train_txt_for_priors is None:
        raise ValueError("train_txt_for_priors is required for logit adjustment.")

    # priors (compute on all ranks; cheap)
    prior_left, prior_right = compute_priors_from_train_txt(train_txt_for_priors, n_cls=8)

    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    n_batches = torch.tensor(0, device=device)

    # Collect logits+gt (will gather to rank0)
    local_logits = {name: [] for name in label_names}
    local_gts = {name: [] for name in label_names}

    for batch in dataloader:
        img = batch['img'].float().to(device)
        roi_boxes = batch.get('roi_boxes', None)
        if roi_boxes is not None:
            roi_boxes = roi_boxes.to(device)

        target_labels = {t: batch['labels'][t].to(device) for t in batch['labels']}

        # forward with ensemble
        output = forward_tta_roiensemble_logits(
            model, img, roi_boxes, tta_n=tta_n, jitter_n=roi_jitter_n
        )

        # loss from logits (still valid input type)
        if hasattr(model, 'module'):
            val_loss, _ = model.module.get_loss(output, target_labels)
        else:
            val_loss, _ = model.get_loss(output, target_labels)

        total_loss += val_loss.detach()
        n_batches += 1

        for name in label_names:
            local_logits[name].append(output[name].detach().cpu().numpy())
            local_gts[name].append(target_labels[name].detach().cpu().numpy())

    # reduce loss across ranks
    total_loss = reduce_tensor(total_loss, average=False)
    n_batches = reduce_tensor(n_batches, average=False)
    avg_loss = (total_loss.item() / max(1.0, n_batches.item()))

    # gather logits/gts to main process
    gathered_logits = {name: [] for name in label_names}
    gathered_gts = {name: [] for name in label_names}

    for name in label_names:
        obj = {
            "logits": np.vstack(local_logits[name]) if len(local_logits[name]) else np.zeros((0, 8), dtype=np.float32),
            "gt": np.concatenate(local_gts[name]) if len(local_gts[name]) else np.zeros((0,), dtype=np.int64)
        }
        all_obj = all_gather_object(obj)
        if is_main_process():
            all_logits = [o["logits"] for o in all_obj if o["logits"].shape[0] > 0]
            all_gt = [o["gt"] for o in all_obj if o["gt"].shape[0] > 0]
            gathered_logits[name] = np.vstack(all_logits) if len(all_logits) else np.zeros((0, 8), dtype=np.float32)
            gathered_gts[name] = np.concatenate(all_gt) if len(all_gt) else np.zeros((0,), dtype=np.int64)

    # compute tau sweep + best metrics on main, then broadcast
    score_accumulators = {}
    if is_main_process():
        for name in label_names:
            logits = gathered_logits[name]
            gt = gathered_gts[name]
            if logits.shape[0] == 0:
                score_accumulators[name] = {
                    "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "specificity": 0.0, "auc": 0.0, "best_tau": float(tau_list[0]),
                    "tau_sweep": {}
                }
                continue

            prior = prior_left if name == "label_left_hip" else prior_right
            best_metrics = sweep_tau_for_task(logits, gt, prior, tau_list=tau_list)
            score_accumulators[name] = best_metrics

        # logging
        if logger is not None:
            for name in label_names:
                m = score_accumulators[name]
                logger.add_scalar(f'val_loss', avg_loss, iteration)
                logger.add_scalar(f'val_accuracy_{name}', m["accuracy"], iteration)
                logger.add_scalar(f'val_precision_{name}', m["precision"], iteration)
                logger.add_scalar(f'val_recall_{name}', m["recall"], iteration)
                logger.add_scalar(f'val_f1_{name}', m["f1"], iteration)
                logger.add_scalar(f'val_specificity_{name}', m["specificity"], iteration)
                logger.add_scalar(f'val_auc_{name}', m["auc"], iteration)
                logger.add_scalar(f'val_best_tau_{name}', m["best_tau"], iteration)

        # print sweep summary
        print("\n[validate] tau sweep results (optimized by balanced-acc):")
        for name in label_names:
            print(f"  {name}: best_tau={score_accumulators[name]['best_tau']:.2f} "
                  f"acc={score_accumulators[name]['accuracy']:.4f} f1={score_accumulators[name]['f1']:.4f} auc={score_accumulators[name]['auc']:.4f}")

    score_accumulators = broadcast_object(score_accumulators)

    model.train()
    return avg_loss, score_accumulators


# Keep for compatibility if other code calls it
def calculate_metrics_Dynamic(output, targets, label_names):
    """
    Legacy per-batch metric calculator (kept for compatibility).
    NOTE: validate_Dynamic now computes dataset-level metrics after gather.
    """
    metrics = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in label_names:
            pred_logits = output[name].detach().cpu().numpy()
            gt = targets[name].cpu().numpy()
            pred_labels = np.argmax(pred_logits, axis=1)

            metrics[f'accuracy_{name}'] = balanced_accuracy_score(gt, pred_labels)
            metrics[f'precision_{name}'] = precision_score(gt, pred_labels, average='macro', zero_division=0)
            metrics[f'recall_{name}'] = recall_score(gt, pred_labels, average='macro', zero_division=0)
            metrics[f'f1_{name}'] = f1_score(gt, pred_labels, average='macro', zero_division=0)

            cm = confusion_matrix(gt, pred_labels, labels=np.arange(pred_logits.shape[1]))
            tn_fp = cm.sum(axis=0) - np.diag(cm)
            fn_tp = cm.sum(axis=1) - np.diag(cm)
            tn = cm.sum() - (tn_fp + fn_tp + np.diag(cm))
            specificity = np.mean(tn / (tn + tn_fp + 1e-8))
            metrics[f'specificity_{name}'] = specificity

    return metrics
