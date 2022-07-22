num_classes = 3
norm_cfg = dict(type='SyncBN', requires_grad=True)
loss = [dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)]
model = dict(
    type='SMPUnet',
    backbone=dict(type='timm-efficientnet-b5', pretrained='noisy-student'),
    decode_head=dict(
        num_classes=3,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            dict(type='SMPDiceLoss', mode='multilabel', loss_weight=4.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole', multi_label=True))
dataset_type = 'CustomDataset'
data_root = '../mmseg_train/'
classes = ['large_bowel', 'small_bowel', 'stomach']
palette = [[0, 0, 0], [128, 128, 128], [255, 255, 255]]
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
size = 448
resize = 640
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        color_type='unchanged',
        max_value='max'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='Albu',
        transforms=[
            dict(type='RandomBrightnessContrast', p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='ElasticTransform',
                        alpha=120,
                        sigma=6.0,
                        alpha_affine=3.5999999999999996,
                        p=1),
                    dict(type='GridDistortion', p=1),
                    dict(
                        type='OpticalDistortion',
                        distort_limit=2,
                        shift_limit=0.5,
                        p=1)
                ],
                p=0.3),
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=10,
                interpolation=1,
                p=0.5),
            dict(type='Resize', height=640, width=640, always_apply=True, p=1),
            dict(type='RandomCrop', height=448, width=448, p=1)
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
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
            dict(type='Pad', size=(640, 640), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=8,
    train=dict(
        type='CustomDataset',
        multi_label=True,
        data_root='../mmseg_train/',
        img_dir='images',
        ann_dir='labels',
        img_suffix='.png',
        seg_map_suffix='.png',
        split='splits_all_mask8/fold_mask_all.txt',
        classes=['large_bowel', 'small_bowel', 'stomach'],
        palette=[[0, 0, 0], [128, 128, 128], [255, 255, 255]],
        pipeline=[
            dict(
                type='LoadImageFromFile',
                to_float32=True,
                color_type='unchanged',
                max_value='max'),
            dict(type='LoadAnnotations'),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='Albu',
                transforms=[
                    dict(type='RandomBrightnessContrast', p=0.5),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='ElasticTransform',
                                alpha=120,
                                sigma=6.0,
                                alpha_affine=3.5999999999999996,
                                p=1),
                            dict(type='GridDistortion', p=1),
                            dict(
                                type='OpticalDistortion',
                                distort_limit=2,
                                shift_limit=0.5,
                                p=1)
                        ],
                        p=0.3),
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.1,
                        scale_limit=0.2,
                        rotate_limit=10,
                        interpolation=1,
                        p=0.5),
                    dict(
                        type='Resize',
                        height=640,
                        width=640,
                        always_apply=True,
                        p=1),
                    dict(type='RandomCrop', height=448, width=448, p=1)
                ]),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
            dict(type='Pad', size=(448, 448), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CustomDataset',
        multi_label=True,
        data_root='../mmseg_train/',
        img_dir='images',
        ann_dir='labels',
        img_suffix='.png',
        seg_map_suffix='.png',
        split='splits_all_mask8/holdout_1.txt',
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
                img_scale=(640, 640),
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
                        size=(640, 640),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CustomDataset',
        multi_label=True,
        data_root='../mmseg_train/',
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
                img_scale=(640, 640),
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
                        size=(640, 640),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=100,
    hooks=[dict(type='CustomizedTextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
total_iters = 10
optimizer = dict(
    type='AdamW', lr=0.0003, betas=(0.9, 0.999), weight_decay=0.005)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0,
    by_epoch=False)
find_unused_parameters = True
runner = dict(type='IterBasedRunner', max_iters=24900)
checkpoint_config = dict(by_epoch=False, interval=415, save_optimizer=False)
evaluation = dict(
    by_epoch=False, interval=830, metric=['imDice', 'mDice'], pre_eval=True)
fp16 = dict()
work_dir = './work_dirs/new/baseline_640_b5_bs20_mask_dice'
gpu_ids = range(0, 2)
