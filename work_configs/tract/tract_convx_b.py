num_classes = 3

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained="./weights/convnext_base_22k_1k_384.pth",
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3], 
        dims=[128, 256, 512, 1024], 
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            dict(type='DiceLoss', multi_label=True, loss_weight=0.5)],
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            dict(type='DiceLoss', multi_label=True, loss_weight=0.5)],
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole", multi_label=True))

# dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/tract/'
classes = ['large_bowel', 'small_bowel', 'stomach']
palette = [[0,0,0], [128,128,128], [255,255,255]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
size = 384
# crop_size = (256, 256)
albu_train_transforms = [
    dict(type='RandomBrightnessContrast', p=0.5),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(size, size), keep_ratio=True),
    dict(type='RandomFlip', prob=0.0, direction='horizontal'),
    # dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # dict(type='RandomRotate90', prob=0.5),
    dict(type='Albu', transforms=albu_train_transforms),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(size, size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
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
        img_dir='mmseg_train/images',
        ann_dir='mmseg_train/labels',
        img_suffix=".jpg",
        seg_map_suffix='.png',
        split="mmseg_train/splits/fold_0.txt",
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        multi_label=True,
        data_root=data_root,
        img_dir='mmseg_train/images',
        ann_dir='mmseg_train/labels',
        img_suffix=".jpg",
        seg_map_suffix='.png',
        split="mmseg_train/splits/holdout_0.txt",
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

total_iters = 15
# optimizer
optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', type='AdamW', 
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.8,
                                'decay_type': 'stage_wise',
                                'num_layers': 12})
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# learning policy
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=total_iters * 1000)
checkpoint_config = dict(by_epoch=False, interval=total_iters * 1000, save_optimizer=False)
evaluation = dict(by_epoch=False, interval=5000, metric=['mDice'], pre_eval=True)
fp16 = dict(loss_scale=512.0)

work_dir = f'./work_dirs/tract/convx_b_16x2_15k_dcp_aug0_f0'

