_base_ = 'faster-rcnn_r50_fpn_1x_SAR-AIRcraft-1.0.py'

train_cfg = dict(
    type='EpochBasedTrainLoop',  # 训练循环的类型，请参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=30,  # 最大训练轮次
    val_interval=1)  # 验证间隔。每个 epoch 验证一次
val_cfg = dict(type='ValLoop')  # 验证循环的类型
test_cfg = dict(type='TestLoop')  # 测试循环的类型

param_scheduler = [
    dict(
        type='MultiStepLR',  # 在训练过程中使用 multi step 学习率策略
        milestones=[20, 28],  # 在哪几个 epoch 进行学习率衰减
        )  # 学习率衰减系数
]
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend',
                    init_kwargs=dict(
                    project='mmdetection-SAR-AIRcraft1.0',
                    name='{{fileBasenameNoExtension}}')
                )
                ]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')