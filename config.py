from functools import partial
import torch.nn as nn
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling import FCOS
from detectron2.modeling import ResNet


from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone import BasicStem, FPN, ResNet
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
)


from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

dataloader = OmegaConf.create()




dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="FLIR_THERMAL_train_data"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.RandomApply)(
                tfm_or_aug=L(T.AugmentationList)(
                    augs=[
                        L(T.ResizeShortestEdge)(
                            short_edge_length=[400, 500, 600], sample_style="choice"
                        ),
                        L(T.RandomCrop)(crop_type="absolute_range", crop_size=(384, 600)),
                    ]
                ),
                prob=0.5,
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            L(T.RandomFlip)(horizontal=True),
        ],
        image_format="RGB",
        use_instance_mask=True,
    ),
    total_batch_size=4,
    num_workers=2,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="FLIR_THERMAL_val_data", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=2,
)

#######
#FCOS
#######
from detectron2.model_zoo.configs.common.optim import AdamW as optimizer
from detectron2.model_zoo.configs.common.coco_schedule import lr_multiplier_1x as lr_multiplier
from detectron2.model_zoo.configs.common.data.coco import dataloader
from detectron2.model_zoo.configs.common.models.fcos import model
from detectron2.model_zoo.configs.common.train import train
dataloader.train.mapper.use_instance_mask = False
optimizer.lr = 0.01
model.backbone.bottom_up.freeze_at = -1
#train.init_checkpoint = "detectron2://ImageNetPretranied/MSRA/R-50.pkl"




# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = ""
train.max_iter = 270000
train.output_dir = '/content/drive/MyDrive/FCOS_DEBUG'
train.checkpointer=dict(period=5000, max_to_keep=100) # options for PeriodicCheckpointer
train.eval_period=5000


lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[270000-60000, 270000-20000,270000],
    ),
    warmup_length=5000 / train.max_iter,
    warmup_factor=0.001,
)


dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir= train.output_dir,
)


optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.overrides = {
    "pos_embed": {"weight_decay": 0.0},
    "rel_pos_h": {"weight_decay": 0.0},
    "rel_pos_w": {"weight_decay": 0.0},
}
optimizer.lr = 0.0001
print(model)
