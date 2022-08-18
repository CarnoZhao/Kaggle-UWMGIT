num_classes = 3
norm_cfg = dict(type='SyncBN', requires_grad=True)
loss = [dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)]
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.3,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole', multi_label=True))
dataset_type = 'CustomDataset'
data_root = 'data/tract/'
classes = ['large_bowel', 'small_bowel', 'stomach']
palette = [[0, 0, 0], [128, 128, 128], [255, 255, 255]]
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
size = 448
albu_train_transforms = [
    dict(type='GridDistortion', p=0.5),
    dict(type='RandomBrightnessContrast', p=0.5)
]
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        color_type='unchanged',
        max_value='max'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(448, 448), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate90', prob=0.5),
    dict(
        type='Albu',
        transforms=[
            dict(type='GridDistortion', p=0.5),
            dict(type='RandomBrightnessContrast', p=0.5)
        ]),
    dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
    dict(type='Pad', size=(448, 448), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        color_type='unchanged',
        max_value='max'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(448, 448),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
            dict(type='Pad', size=(448, 448), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
holdout = 0
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='CustomDataset',
        multi_label=True,
        data_root='data/tract/',
        img_dir='mmseg_train/images',
        ann_dir='mmseg_train/labels',
        img_suffix='.png',
        seg_map_suffix='.png',
        split='mmseg_train/splits_notail/fold_all.txt',
        classes=['large_bowel', 'small_bowel', 'stomach'],
        palette=[[0, 0, 0], [128, 128, 128], [255, 255, 255]],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                to_float32=True,
                color_type='unchanged',
                max_value='max'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(448, 448), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(type='RandomRotate90', prob=0.5),
            dict(
                type='Albu',
                transforms=[
                    dict(type='GridDistortion', p=0.5),
                    dict(type='RandomBrightnessContrast', p=0.5)
                ]),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
            dict(type='Pad', size=(448, 448), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CustomDataset',
        multi_label=True,
        data_root='data/tract/',
        img_dir='mmseg_train/images',
        ann_dir='mmseg_train/labels',
        img_suffix='.png',
        seg_map_suffix='.png',
        split='mmseg_train/splits_notail/holdout_0.txt',
        classes=['large_bowel', 'small_bowel', 'stomach'],
        palette=[[0, 0, 0], [128, 128, 128], [255, 255, 255]],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                to_float32=True,
                color_type='unchanged',
                max_value='max'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(448, 448),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(448, 448),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CustomDataset',
        multi_label=True,
        data_root='data/tract/',
        test_mode=True,
        img_dir='test/images',
        ann_dir='test/labels',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=['large_bowel', 'small_bowel', 'stomach'],
        palette=[[0, 0, 0], [128, 128, 128], [255, 255, 255]],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                to_float32=True,
                color_type='unchanged',
                max_value='max'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(448, 448),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(448, 448),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='CustomizedTextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './weights/upernet_convnext_small_1k_512x512.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
total_iters = 20
optimizer = dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
find_unused_parameters = True
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=20000, save_optimizer=False)
evaluation = dict(
    by_epoch=False, interval=20000, metric=['imDice', 'mDice'], pre_eval=True)
fp16 = dict()
opencv_num_threads = 0
mp_start_method = 'fork'
# work_dir = './work_dirs/tract/upt_cvs_448_grd_20k_opt_all'
work_dir = './work_dirs/tract/seg_7'
gpu_ids = range(0, 2)
