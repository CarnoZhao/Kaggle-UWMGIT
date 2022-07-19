import numpy as np
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    RandSpatialCropd,
    EnsureTyped,
    CastToTyped,
    NormalizeIntensityd,
    RandFlipd,
    CenterSpatialCropd,
    ScaleIntensityRanged,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCoarseDropoutd,
    Rand2DElasticd,
    Lambdad,
    Resized,
    AddChanneld,
    RandGaussianNoised,
    RandGridDistortiond,
    RepeatChanneld,
    Transposed,
    OneOf,
    EnsureChannelFirstd,
    RandLambdad,
    Spacingd,
    FgBgToIndicesd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    ToDeviced,
    SpatialPadd,

)

from default_config import basic_cfg

cfg = basic_cfg

# train
cfg.train = True
cfg.eval = False
cfg.start_eval_epoch = 5  # when use large lr, can set a large num
cfg.run_org_eval = False
cfg.run_tta_val = False
cfg.load_best_weights = False
cfg.amp = False
cfg.val_amp = False
cfg.num_workers = 8
cfg.model_type = "segres20"

# device
cfg.gpu = 0
cfg.device = "cuda:%d" % cfg.gpu

# lr
# warmup_restart, cosine
cfg.lr_mode = "warmup_restart"
cfg.lr = 5e-4
cfg.min_lr = 2e-4
cfg.weight_decay = 1e-6
cfg.epochs = 1000
cfg.restart_epoch = 100  # only for warmup_restart
cfg.eval_epochs = 10

cfg.finetune_lb = -1

# dataset
cfg.img_size = (160, 160, 80)
cfg.spacing = (1.5, 1.5, 1.5)
cfg.batch_size = 4
cfg.val_batch_size = 1
cfg.train_cache_rate = 0.0
cfg.val_cache_rate = 0.0
cfg.gpu_cache = False
cfg.val_gpu_cache = False

# val
cfg.roi_size = (224, 224, 80)
cfg.sw_batch_size = 4

# model

# loss
cfg.w_dice = 1.0

cfg.output_dir = "./output/segres20_all"
        
# transforms
cfg.train_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        # Spacingd(keys=["image", "mask"], pixdim=cfg.spacing, mode=("bilinear", "nearest")),
        RandSpatialCropd(
            keys=("image", "mask"),
            roi_size=cfg.img_size,
            random_size=False,
        ),
       
        Lambdad(keys="image", func=lambda x: x / x.max()),

        RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[1]),
        # RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[2]),
        RandAffined(
            keys=("image", "mask"),
            prob=0.5,
            rotate_range=np.pi / 12,
            translate_range=(cfg.img_size[0]*0.0625, cfg.img_size[1]*0.0625),
            scale_range=(0.1, 0.1),
            mode="nearest",
            padding_mode="reflection",
        ),
        OneOf(
            [
                RandGridDistortiond(keys=("image", "mask"), prob=0.5, distort_limit=(-0.05, 0.05), mode="nearest", padding_mode="reflection"),
                RandCoarseDropoutd(
                    keys=("image", "mask"),
                    holes=5,
                    max_holes=8,
                    spatial_size=(1, 1, 1),
                    max_spatial_size=(12, 12, 12),
                    fill_value=0.0,
                    prob=0.5,
                ),
            ]
        ),
        RandScaleIntensityd(keys="image", factors=(-0.2, 0.2), prob=0.5),
        RandShiftIntensityd(keys="image", offsets=(-0.1, 0.1), prob=0.5),
        EnsureTyped(keys=("image", "mask"), dtype=torch.float32),
    ]
)

cfg.val_transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        # Spacingd(keys=["image", "mask"], pixdim=cfg.spacing, mode=("bilinear", "nearest")),
        Lambdad(keys="image", func=lambda x: x / x.max()),
       
        EnsureTyped(keys=("image", "mask"), dtype=torch.float32),
        # ToDeviced(keys=["image", "mask"], device="cuda:0"),
    ]
)

cfg.org_val_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        # Spacingd(keys="image", pixdim=cfg.spacing, mode="bilinear"),
        Lambdad(keys="image", func=lambda x: x / x.max()),
        # SpatialPadd(keys="image", spatial_size=cfg.img_size),
        EnsureTyped(keys="image", dtype=torch.float32),
    ]
)


