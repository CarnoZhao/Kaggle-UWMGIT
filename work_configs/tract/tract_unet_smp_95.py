num_classes = 3

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
loss = [
    dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    # dict(type='LovaszLoss', per_image=True, loss_type='binary', loss_weight=1.0),
    # dict(type='SMPDiceLoss', mode='multilabel', loss_weight=1.0),
]
model = dict(
    type='SMPUnet',
    backbone=dict(
        type='tu-convnext_base_in22k',
        depth=4,
        pretrained="imagenet"
    ),
    decode_head=dict(
        num_classes=num_classes,
        align_corners=False,
        # attention_type="scse",
        loss_decode=loss
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole", multi_label=True))

# dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/tract/'
classes = ['large_bowel', 'small_bowel', 'stomach']
palette = [[0,0,0], [128,128,128], [255,255,255]]
img_norm_cfg = dict(mean=[114.495, 114.495, 114.495], std=[57.62, 57.62, 57.62], to_rgb=False)
size = 512
# crop_size = (256, 256)
albu_train_transforms = [
    # dict(type='Affine', rotate=45, translate_percent=10, scale=0.05),
    # dict(type='IAAPiecewiseAffine', p=0.5),
    # dict(type='OpticalDistortion', p=0.5),
    dict(type='GridDistortion', p=0.5),
    dict(type='RandomBrightnessContrast', p=0.5),
]
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='unchanged', max_value='q0.95', back_to_uint8=True),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(size, size), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate90', prob=0.5),
    dict(type='Albu', transforms=albu_train_transforms),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
valid_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='unchanged', max_value='q0.95', back_to_uint8=True),
    dict(type='LoadAnnotations'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(size, size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='unchanged', max_value='q0.95', back_to_uint8=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(size, size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        multi_label=True,
        data_root=data_root,
        img_dir='mmseg_train/2.5d_images',
        ann_dir='mmseg_train/2.5d_labels',
        img_suffix=".png",
        seg_map_suffix='.png',
        split="mmseg_train/splits/fold_0.txt",
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        multi_label=True,
        data_root=data_root,
        img_dir='mmseg_train/2.5d_images',
        ann_dir='mmseg_train/2.5d_labels',
        img_suffix=".png",
        seg_map_suffix='.png',
        split="mmseg_train/splits/holdout_0.txt",
        classes=classes,
        palette=palette,
        pipeline=valid_pipeline),
    test=dict(
        type=dataset_type,
        multi_label=True,
        data_root=data_root,
        test_mode=True,
        img_dir='test/images',
        ann_dir='test/labels',
        img_suffix=".jpg",
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='CustomizedTextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None # "./work_dirs/tamper/convx_t_8x/epoch_96.pth"
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

total_iters = 20
# optimizer
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# learning policy
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# runtime settings
find_unused_parameters=True
runner = dict(type='IterBasedRunner', max_iters=total_iters * 1000)
checkpoint_config = dict(by_epoch=False, interval=total_iters * 1000, save_optimizer=False)
evaluation = dict(by_epoch=False, interval=5000, metric=['imDice', 'mDice'], pre_eval=True)
fp16 = dict()

work_dir = f'./work_dirs/tract/ut_cvb_512_grd_95_f0'

