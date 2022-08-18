num_classes = 3

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
loss = [
    dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    # dict(type='FocalLoss', use_sigmoid=True, multi_label=True, alpha=0.5, gamma=2., loss_weight=1.0),
    # dict(type='LovaszLoss', per_image=True, loss_type='binary', loss_weight=1.0),
    # dict(type='SMPDiceLoss', mode='multilabel', loss_weight=1.0),
]
model = dict(
    type='EncoderDecoder',
    pretrained="./weights/convnext_xlarge_22k_1k_224_ema.pth",
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3], 
        dims=[256, 512, 1024, 2048], 
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=loss,
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=loss,
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole", multi_label=True))

# dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/tract/'
classes = ['large_bowel', 'small_bowel', 'stomach']
palette = [[0,0,0], [128,128,128], [255,255,255]]
img_norm_cfg = dict(mean=[0,0,0], std=[1,1,1], to_rgb=True)
size = 384
# crop_size = (256, 256)
albu_train_transforms = [
    # dict(type='Affine', rotate=45, translate_percent=10, scale=0.05),
    # dict(type='IAAPiecewiseAffine', p=0.5),
    # dict(type='OpticalDistortion', p=0.5),
    # dict(type='OneOf', transforms=[
    #     dict(type='Blur'),
    #     dict(type='GaussNoise'),
    #     dict(type='JpegCompression')
    # ]),
    # dict(type='HorizontalFlip'),
    # dict(type='VerticalFlip'),
    # dict(type='RandomRotate90'),
    # dict(type='ShiftScaleRotate', shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
    dict(type='GridDistortion', p=0.5),
    dict(type='RandomBrightnessContrast', p=0.5),
    # dict(type='CoarseDropout', max_holes=8, max_height=size//20, max_width=size//20, min_holes=5, fill_value=0, mask_fill_value=255, p=0.5),
]
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='unchanged', max_value='max'),
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
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='unchanged', max_value='max'),
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
holdout = 2
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        multi_label=True,
        data_root=data_root,
        img_dir='mmseg_train/2.5d_images',
        ann_dir='mmseg_train/2.5d_labels',
        img_suffix=".png",
        seg_map_suffix='.png',
        split=f"mmseg_train/splits/fold_{holdout}.txt",
        # keep_empty_prob=0.1,
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
        split=f"mmseg_train/splits/holdout_{holdout}.txt",
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
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

total_iters = 30
# optimizer
optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', type='AdamW', 
                 lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.8,
                                'decay_type': 'stage_wise',
                                'num_layers': 12})
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# learning policy
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=total_iters * 1000)
checkpoint_config = dict(by_epoch=False, interval=total_iters * 1000, save_optimizer=False)
evaluation = dict(by_epoch=False, interval=5000, metric=['imDice', 'mDice'], pre_eval=True)
fp16 = dict()

work_dir = f'./work_dirs/tract/upt_cvxl_384_grd_2.5d_f{holdout}'

