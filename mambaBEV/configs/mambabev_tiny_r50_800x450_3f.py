"""mambaBEV Tiny (R50, 800x450, 3 history frames).

Paper-aligned Tiny setting for mambaBEV.
Based on single-frame DETR3D (ResNet-50, no grid mask).
"""

# ========================
# 1) Base inheritance
# ========================
# IMPORTANT:
# Use a single-frame DETR3D R50 config WITHOUT grid mask or heavy augmentation.
# Adjust this path if your repo places it elsewhere.
_base_ = [
    "../../mmdetection3d/projects/DETR3D/configs/detr3d_r50.py",
]

# ========================
# 2) custom_imports (required)
# ========================
custom_imports = dict(
    imports=["mambabev"],
    allow_failed_imports=False,
)

# Explicitly trigger registration
import mambabev  # noqa: E402
mambabev.register_all_modules()  # noqa: E402

# ========================
# 3) Dataset / class names
# ========================
# nuScenes standard 10 classes
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# ========================
# 4) Tiny-specific settings (paper-aligned)
# ========================
img_scale = (800, 450)

bev_h = 50
bev_w = 50
bev_resolution = 2.048

history_frames = 3
encoder_layers = 3

embed_dims = 256

# Put BEV shape into metainfo so detector can read it safely
metainfo = dict(
    classes=class_names,
    bev_h=bev_h,
    bev_w=bev_w,
    bev_resolution=bev_resolution,
    history_frames=history_frames,
)

# ========================
# 5) Data pipeline (NO extra augmentation)
# ========================
# Deterministic resize only (paper: no data augmentation)
train_pipeline = [
    dict(
        type="RandomResize3D",
        scale=img_scale,
        ratio_range=(1.0, 1.0),
        keep_ratio=True,
    ),
]

test_pipeline = train_pipeline

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
        metainfo=metainfo,
    )
)

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        metainfo=metainfo,
    )
)

test_dataloader = val_dataloader

# ========================
# 6) Model (core)
# ========================
model = dict(
    type="MambaBEVDetector",

    # Explicit BEV shape (robust fallback)
    bev_h=bev_h,
    bev_w=bev_w,

    temporal_mamba=dict(
        type="TemporalMamba",
        embed_dim=embed_dims,
        mamba_dim=embed_dims,
        dropout=0.9,  # paper-specified residual dropout
    ),

    # Backbone: ResNet-50, no DCN
    img_backbone=dict(
        _delete_=True,
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
    ),

    # Transformer: explicitly set encoder layers (Tiny = 3)
    transformer=dict(
        encoder=dict(
            num_layers=encoder_layers,
        )
    ),

    # Head: Mamba-based DETR3D
    pts_bbox_head=dict(
        type="MambaDETR3DHead",
        num_query=900,          # paper
        embed_dims=embed_dims,
        mamba_dim=embed_dims,
        d_state=128,
        d_conv=4,
        expand=2,
    ),
)

# ========================
# 7) Optimizer (strict paper setting)
# ========================
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=2e-4,
        weight_decay=0.01,
    ),
    # common practice: backbone lr multiplier
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
)

# ========================
# 8) Training / evaluation
# ========================
# Keep epoch schedule, hooks, evaluators from base config
# nuScenes metrics: mAP / NDS

# ========================
# 9) Work directory
# ========================
work_dir = "./work_dirs/mambabev_tiny_r50_800x450_3f"
