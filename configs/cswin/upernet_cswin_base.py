_base_ = [
    '../_base_/models/upernet_cswin.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        type='CSWin',
        embed_dim=96,
        depth=[2,4,32,2],
        num_heads=[4,8,16,32],
        split_size=[1,2,7,7],
        drop_path_rate=0.6,
        use_chk=False,
    ),
    decode_head=dict(
        in_channels=[96,192,384,768],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=150
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data=dict(samples_per_gpu=2)
