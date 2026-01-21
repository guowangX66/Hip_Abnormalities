# train.py (improve_v1)
import os
import random
import time
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import MultiLabelDataset_Dynamic
from model import model
from test import validate_Dynamic, calculate_metrics_Dynamic
from utils.ema import EMA
from utils.lr_scheduler import CosineAnnealingWarmupLR


class ClassBalancedFocalLoss(torch.nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999, gamma=1.0, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
        self.class_balanced_weights = self._compute_cb_weights(samples_per_cls)

    def _compute_cb_weights(self, samples_per_cls):
        weights = {}
        for task, freq in samples_per_cls.items():
            freq = torch.tensor(freq).float()
            effective_num = 1.0 - torch.pow(self.beta, freq)
            cb_weight = (1.0 - self.beta) / (effective_num + 1e-8)
            cb_weight = cb_weight / cb_weight.sum()
            weights[task] = cb_weight
        return weights

    def forward(self, input, target, task_name=None):
        num_classes = input.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(input)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        log_prob = F.log_softmax(input, dim=1)
        pt = torch.exp(log_prob)
        focal_term = (1 - pt) ** self.gamma

        weight = self.class_balanced_weights.get(task_name).to(input.device) if task_name else None
        loss = -true_dist * focal_term * log_prob
        if weight is not None:
            loss = loss * weight.unsqueeze(0)

        loss = loss.sum(dim=1)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def setup_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')


def make_accuracy_dict(class_info):
    return {
        f"{m}_{task}": 0
        for task in class_info
        for m in ["accuracy", "precision", "recall", "specificity", "f1"]
    }


def compute_class_weights(txt_file, class_info):
    """Compute per-task class weights and sample counts from label strings.

    Label format: continuous 0/1 string, e.g. 0010000010000000
    """
    task_names = list(class_info)
    num_classes_list = list(class_info.values())
    counts = {task: Counter() for task in task_names}

    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            label_str = parts[1].strip()
            labels = [int(ch) for ch in label_str]
            if len(labels) != sum(num_classes_list):
                continue
            offset = 0
            for i, task in enumerate(task_names):
                cls_range = num_classes_list[i]
                task_onehot = labels[offset:offset + cls_range]
                idx = task_onehot.index(1) if 1 in task_onehot else -1
                if idx != -1:
                    counts[task][idx] += 1
                offset += cls_range

    class_weights = {}
    samples_per_cls = {}
    for task in task_names:
        num_classes = class_info[task]
        freqs = [counts[task][i] for i in range(num_classes)]
        freqs_tensor = torch.tensor(freqs).float()
        # inverse-frequency -> softmax (stable weights for CE if needed)
        weights = (1.0 / (freqs_tensor + 1e-6)).softmax(0)
        class_weights[task] = weights
        samples_per_cls[task] = freqs

    return class_weights, samples_per_cls


if __name__ == '__main__':
    setup_seed()
    local_rank = setup_ddp()
    device = torch.device('cuda', local_rank)
    is_main = local_rank == 0

    # ------------------------
    # Config
    # ------------------------
    train_txt = 'data/medician_data/labels/train.txt'
    val_txt = 'data/medician_data/labels/val.txt'
    img_dir = 'data/medician_data/imgs'
    model_name = 'resnet50'
    pretrained_path = '/home/lvyinghao/Projects/learn/myclass/weights/resnet50-19c8e357.pth'
    class_info = {'label_left_hip': 8, 'label_right_hip': 8}

    IMG_SIZE = 512
    epochs = 100
    batch_size = 32  # 512^2 is heavier; adjust to your GPU
    num_workers = 8
    lr = 1e-4

    class_weights, samples_per_cls = compute_class_weights(train_txt, class_info)

    # ------------------------
    # Augmentations (geometry-safe)
    # IMPORTANT: No HorizontalFlip for left/right anatomical labels.
    # ------------------------
    train_transform = A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Affine(scale=(0.95, 1.05),rotate=(-10, 10),translate_percent=0.05,shear=(-3, 3),p=0.6,),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc',label_fields=['bbox_labels'],min_visibility=0.3,),
    )

    val_transform = A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc',label_fields=['bbox_labels'],min_visibility=0.3,),
    )

    train_set = MultiLabelDataset_Dynamic(train_txt, img_dir, train_transform, label_info=class_info)
    val_set = MultiLabelDataset_Dynamic(val_txt, img_dir, val_transform, label_info=class_info)

    # DDP-safe sampling
    train_sampler = DistributedSampler(train_set, shuffle=True)
    val_sampler = DistributedSampler(val_set, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)

    net = model(model_name, (1, 3, IMG_SIZE, IMG_SIZE), class_info, pretrained_path=pretrained_path, class_weights=class_weights).to(device)
    net = DDP(net, device_ids=[local_rank], find_unused_parameters=False)

    ema = EMA(net, decay=0.9995)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = CosineAnnealingWarmupLR(optimizer, warmup_epochs=10, max_epochs=epochs)

    criterion = ClassBalancedFocalLoss(samples_per_cls=samples_per_cls, beta=0.9999, gamma=1.0, smoothing=0.1)

    logdir = os.path.join('./logs', get_cur_time())
    ckptdir = os.path.join('./checkpoints', get_cur_time())
    if is_main:
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(ckptdir, exist_ok=True)
        logger = SummaryWriter(logdir)
    else:
        logger = None

    best_val_score = -1
    best_model_path = None

    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        net.train()

        total_loss = 0.0
        metrics = make_accuracy_dict(class_info)
        start = time.time()

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            img = batch['img'].to(device, non_blocking=True)
            roi_boxes = batch.get('roi_boxes', None)
            if roi_boxes is not None:
                roi_boxes = roi_boxes.to(device, non_blocking=True)

            labels = {k: v.to(device, non_blocking=True) for k, v in batch['labels'].items()}

            preds = net(img, roi_boxes=roi_boxes)
            loss = sum(criterion(preds[t], labels[t], task_name=t) for t in preds)

            total_loss += float(loss.item())
            loss.backward()
            optimizer.step()
            ema.update(net)

            batch_metrics = calculate_metrics_Dynamic(preds, labels, list(class_info))
            for k in metrics:
                metrics[k] += batch_metrics[k]

        avg_loss = total_loss / len(train_loader)
        avg_metrics = {k: v / len(train_loader) for k, v in metrics.items()}
        # IMPORTANT (DDP correctness):
        # Apply EMA weights on ALL ranks before validation, otherwise all_gather will mix different weights.
        ema.apply_shadow(net)
        if dist.is_initialized():
            dist.barrier()

        val_loss, val_metrics = validate_Dynamic(
            net, val_loader, logger, epoch, device,
            label_names=list(class_info),
            train_txt_for_priors=train_txt,
            tau_list=(0.5, 1.0, 1.5, 2.0),
            tta_n=4,
            roi_jitter_n=5
        )

        if dist.is_initialized():
            dist.barrier()

        scheduler.step()

        if is_main:
            logger.add_scalar("train_loss", avg_loss, epoch)
            for k, v in avg_metrics.items():
                logger.add_scalar(f"train_{k}", v, epoch)

            print(f"[Epoch {epoch:03d}] Train Loss: {avg_loss:.4f}")
            for task in class_info:
                print(
                    f"  Train [{task}] --> "
                    f"Acc: {avg_metrics[f'accuracy_{task}']:.4f} | "
                    f"Prec: {avg_metrics[f'precision_{task}']:.4f} | "
                    f"Rec: {avg_metrics[f'recall_{task}']:.4f} | "
                    f"Spec: {avg_metrics[f'specificity_{task}']:.4f} | "
                    f"F1: {avg_metrics[f'f1_{task}']:.4f}"
                )

            print(f"[Epoch {epoch:03d}] Val Loss: {val_loss:.4f}")
            for task in class_info:
                print(
                    f"  Val   [{task}] --> "
                    f"Acc: {val_metrics[task]['accuracy']:.4f} | "
                    f"Prec: {val_metrics[task]['precision']:.4f} | "
                    f"Rec: {val_metrics[task]['recall']:.4f} | "
                    f"Spec: {val_metrics[task]['specificity']:.4f} | "
                    f"F1: {val_metrics[task]['f1']:.4f} | "
                    f"AUC: {val_metrics[task]['auc']:.4f}"
                )

            print(f"Time: {time.time() - start:.2f}s\n")

        current_score = sum(val_metrics[task]['f1'] for task in class_info)
        if is_main and current_score > best_val_score:
            best_val_score = current_score
            best_model_path = os.path.join(ckptdir, f"best_model-epoch{epoch:03d}.pth")
            torch.save(net.module.state_dict(), best_model_path)
            print(f"New best model saved at Epoch {epoch} | Score: {best_val_score:.4f}")
        # Sync + restore EMA weights back to training weights (DDP-safe)
        if dist.is_initialized():
            dist.barrier()
        ema.restore(net)
        if dist.is_initialized():
            dist.barrier()


    if is_main:
        print(f"\nTraining complete! Best model saved at:\n{best_model_path}")

    cleanup_ddp()
