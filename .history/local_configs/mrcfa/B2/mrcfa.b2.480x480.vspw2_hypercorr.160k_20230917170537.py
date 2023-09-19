_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/vspw_repeat2.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder_clips',
    pretrained='/home/lixinhao/original/mit_b2.pth',
    backbone=dict(
        type='mit_b2',
        style='pytorch'),
    decode_head=dict(
        # type='SegFormerHead_clips2_resize_1_8_hypercorrelation2_topk_ensemble4',      ## SegFormerHead_clips2_resize_1_8_hypercorrelation3
        # type='MLPHead',
        # num_infer = 5,
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=124,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        num_clips=4,
        hypercorre=True,
        backbone='b2'),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.000006 , betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                #  warmup_ratio=1e-8,
                 power=1.0, min_lr=0.0, by_epoch=False)

data = dict(samples_per_gpu=1,workers_per_gpu=4)
# evaluation = dict(interval=4000, metric='mIoU')
evaluation = dict(interval=160000, metric='mIoU')