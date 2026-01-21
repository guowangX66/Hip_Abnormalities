import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models
from torchvision.ops import roi_align

# -----------------------
# CBAM (Channel + Spatial)
# -----------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, reduction: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        max_pool = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        attn = torch.sigmoid(attn).view(x.size(0), x.size(1), 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(x_cat))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, in_planes: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# -----------------------
# GeM Pooling
# -----------------------

class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(1.0 / self.p)


def build_classifiers(in_features: int, class_info: Dict[str, int]) -> nn.ModuleDict:
    return nn.ModuleDict({
        name: nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, n_cls)
        ) for name, n_cls in class_info.items()
    })


# -----------------------
# Multi-task loss wrapper
# -----------------------

class MultiTaskLossModule_Dynamic(nn.Module):
    def __init__(self, class_info: Dict[str, int], class_weights=None):
        super().__init__()
        self.tasks = list(class_info.keys())
        self.class_weights = class_weights or {}

    def _calculate_task_loss(self, task: str, output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights.get(task, None)
        if weight is not None:
            weight = weight.to(output.device)
        return F.cross_entropy(output, label, weight=weight)

    def get_loss(self, net_output: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]):
        total_loss = 0
        individual_losses = {}
        for task in self.tasks:
            loss = self._calculate_task_loss(task, net_output[task], ground_truth[task])
            total_loss += loss
            individual_losses[task] = loss
        return total_loss, individual_losses


# -----------------------
# Base multi-task head
# -----------------------

class BaseMultiTaskModel(MultiTaskLossModule_Dynamic):
    def __init__(
        self,
        img_bchw,
        class_info: Dict[str, int],
        class_weights=None,
        in_features: Optional[int] = None,
        use_cbam: bool = False,
    ):
        super().__init__(class_info, class_weights)
        assert in_features is not None, "in_features must be provided"

        self.pool = GeM()
        self.use_cbam = use_cbam
        self.cbam = CBAM(in_features) if use_cbam else None

        # Per-task BN is more stable for left/right ROI features
        self.bns = nn.ModuleDict({t: nn.BatchNorm1d(in_features) for t in class_info.keys()})
        self.dropout = nn.Dropout(0.5)
        self.classifiers = build_classifiers(in_features, class_info)

    def _forward_one(self, task: str, x_vec: torch.Tensor) -> torch.Tensor:
        x_vec = self.bns[task](x_vec)
        x_vec = self.dropout(x_vec)
        return self.classifiers[task](x_vec)

    def forward_head_global(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        if feat.ndim == 4:
            x = self.pool(feat).flatten(1)
        elif feat.ndim == 2:
            x = feat
        else:
            raise ValueError(f"Unsupported feature shape: {feat.shape}")

        return {task: self._forward_one(task, x) for task in self.classifiers.keys()}


# -----------------------
# TorchVision CNN backbone (ResNet/VGG/MobileNet) + ROIAlign pooling
# -----------------------

class TorchVisionBackbone(BaseMultiTaskModel):
    def __init__(
        self,
        model_fn,
        img_bchw,
        class_info: Dict[str, int],
        pretrained_path=None,
        class_weights=None,
    ):
        load_local = pretrained_path and os.path.exists(pretrained_path)
        weights_arg = None if load_local else "DEFAULT"
        try:
            # For ResNet: keep deeper features but higher spatial resolution (stride 16)
            model = model_fn(weights=weights_arg, replace_stride_with_dilation=[False, False, True])
        except TypeError:
            model = model_fn(weights=weights_arg)


        if load_local:
            print(f"Loading pretrained weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            state_dict = state_dict.get('model', state_dict)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)

        backbone = nn.Sequential(*list(model.children())[:-2])
        in_features = model.fc.in_features
        use_cbam = True

        super().__init__(img_bchw, class_info, class_weights, in_features=in_features, use_cbam=use_cbam)
        self.backbone = backbone

        # task -> ROI index mapping (default left=0, right=1)
        self.task_to_roi = {}
        if 'label_left_hip' in class_info and 'label_right_hip' in class_info:
            self.task_to_roi = {'label_left_hip': 0, 'label_right_hip': 1}

    def _roi_pool(self, feat: torch.Tensor, boxes_xyxy: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        """ROIAlign pooling on feature map.

        boxes_xyxy: (B,4) in input image pixel coordinates.
        returns: (B,C)
        """
        b, c, h_f, w_f = feat.shape
        # spatial_scale maps input coords -> feature coords
        spatial_scale_h = h_f / float(img_h)
        # assume square-ish scaling; use height scale
        spatial_scale = spatial_scale_h

        # rois: (B,5) with batch indices
        idx = torch.arange(b, device=feat.device, dtype=boxes_xyxy.dtype).unsqueeze(1)
        rois = torch.cat([idx, boxes_xyxy], dim=1)
        pooled = roi_align(feat, rois, output_size=(7, 7), spatial_scale=spatial_scale, aligned=True)
        # use the same GeM pool to aggregate ROI patch -> vector
        pooled_vec = self.pool(pooled).flatten(1)
        return pooled_vec


    def forward(self, x: torch.Tensor, roi_boxes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward.

        roi_boxes (optional): (B,2,4) in input image pixel coords (x1,y1,x2,y2)
        - roi_boxes[:,0] for left hip
        - roi_boxes[:,1] for right hip
        """
        feat = self.backbone(x)
        if self.cbam is not None and feat.ndim == 4:
            feat = self.cbam(feat)

        # global feature
        global_vec = self.pool(feat).flatten(1)

        if roi_boxes is None:
            return self.forward_head_global(feat)

        img_h, img_w = int(x.shape[2]), int(x.shape[3])
        out = {}

        for task in self.classifiers.keys():
            roi_idx = self.task_to_roi.get(task, 0)
            boxes = roi_boxes[:, roi_idx, :].to(feat.dtype)
            roi_vec = self._roi_pool(feat, boxes, img_h=img_h, img_w=img_w)

            # simple fusion: global + ROI
            x_vec = global_vec + roi_vec
            out[task] = self._forward_one(task, x_vec)

        return out


# -----------------------
# TIMM backbone (ConvNeXt/Swin/ViT/etc.)
# -----------------------

class TimmBackbone(BaseMultiTaskModel):
    def __init__(self, model_name, img_bchw, class_info, pretrained=True, class_weights=None, pretrained_path=None):
        load_local = pretrained_path and os.path.exists(pretrained_path)
        model = timm.create_model(model_name, pretrained=not load_local, in_chans=img_bchw[1])

        if load_local:
            print(f"Loading pretrained weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            state_dict = state_dict.get('model', state_dict)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)

        in_features = model.get_classifier().in_features
        model.reset_classifier(0)

        use_cbam = False
        super().__init__(img_bchw, class_info, class_weights, in_features=in_features, use_cbam=use_cbam)
        self.backbone = model

    def forward(self, x: torch.Tensor, roi_boxes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # NOTE: ROI pooling is implemented only for CNN feature maps. For transformers,
        # use global representation.
        feat = self.backbone.forward_features(x)
        if isinstance(feat, tuple):
            feat = feat[0]

        if feat.ndim == 4:
            vec = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        elif feat.ndim == 3:
            vec = feat[:, 0]
        else:
            raise ValueError(f"Unsupported feature shape: {feat.shape}")

        return {task: self._forward_one(task, vec) for task in self.classifiers.keys()}


# -----------------------
# Factory
# -----------------------


def model(name, img_bchw, class_info, pretrained_path=None, class_weights=None):
    name = name.lower()
    if name == 'resnet50':
        return TorchVisionBackbone(models.resnet50, img_bchw, class_info, pretrained_path, class_weights)
    elif name == 'resnet101':
        return TorchVisionBackbone(models.resnet101, img_bchw, class_info, pretrained_path, class_weights)
    elif name == 'wideresnet50':
        return TorchVisionBackbone(models.wide_resnet50_2, img_bchw, class_info, pretrained_path, class_weights)
    elif name in ['densenet121', 'densenet-121', 'dense121']:
        return TorchVisionBackbone(models.densenet121, img_bchw, class_info, pretrained_path, class_weights,".features.conv0")
    elif name == 'vgg11':
        return TorchVisionBackbone(models.vgg11_bn, img_bchw, class_info, pretrained_path, class_weights)
    elif name == 'mobilenetv2':
        return TorchVisionBackbone(models.mobilenet_v2, img_bchw, class_info, pretrained_path, class_weights)
    elif name == 'mobilenetv3':
        return TorchVisionBackbone(models.mobilenet_v3_large, img_bchw, class_info, pretrained_path, class_weights)

    elif name.startswith('efficientnet'):
        return TimmBackbone(name, img_bchw, class_info, pretrained=pretrained_path is None, class_weights=class_weights, pretrained_path=pretrained_path)
    elif name.startswith('convnext'):
        return TimmBackbone(name, img_bchw, class_info, pretrained=pretrained_path is None, class_weights=class_weights, pretrained_path=pretrained_path)
    elif name.startswith('swin'):
        return TimmBackbone(name, img_bchw, class_info, pretrained=pretrained_path is None, class_weights=class_weights, pretrained_path=pretrained_path)
    elif name.startswith('vit'):
        return TimmBackbone(name, img_bchw, class_info, pretrained=pretrained_path is None, class_weights=class_weights, pretrained_path=pretrained_path)

    raise ValueError(f"Unknown model name: {name}")
