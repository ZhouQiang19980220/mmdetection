_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'


model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=7)))

data_root = 'data/SAR-AIRcraft-1.0/'
metainfo = {
    # 这里必须写实际的类别
    'classes': ('A220', 'A320/321', 'A330', 'ARJ21', 'Boeing737', 'Boeing787', 'other'),
    # 'palette': [
    #     (220, 20, 60),
    # ]
}



train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        # 数据集根目录
        data_root=data_root,
        metainfo=metainfo,
        # 注释文件
        ann_file='COCOAnnotations/train.json',
        # 图片在data_root下的相对路径
        data_prefix=dict(img='JPEGImages/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='COCOAnnotations/val.json',
        data_prefix=dict(img='JPEGImages/')))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'COCOAnnotations/val.json', 
                     format_only = False)
test_evaluator = val_evaluator

auto_scale_lr = dict(enable=True)

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend',
                    init_kwargs=dict(
                    project='mmdetection-SAR-AIRcraft1.0',
                    name='{{fileBasenameNoExtension}}')
                )
                ]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook', max_keep_ckpts=2),
)