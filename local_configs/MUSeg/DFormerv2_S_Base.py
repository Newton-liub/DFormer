"""Shared MUSeg DFormerv2-S base configuration.

This module carries only the fields that every retained MUSeg config family inherits
unchanged: the dataset name and dataset-independent formats, the label/modal geometry,
the frozen normalization statistics, the HAM decoder choice and the evaluation geometry.

Deliberately **not** here, because they depend on the machine, the frozen split or the
experiment identity, and are declared explicitly by each config instead
(``DFormerv2_S_QuickB0.py``, ``DFormerv2_S_MMFR_A2_Common_v3.py``,
``DFormerv2_S_4090.py``):

* data location: ``dataset_path``, ``rgb_root_folder``, ``gt_root_folder``, ``x_root_folder``
* split sources and counts: ``train_source``, ``eval_source``, ``num_train_imgs``,
  ``num_eval_imgs``
* input identity: ``channel_order``, ``normalization_identity``, ``pretrained_model``

The retired ``DFormerv2_S_MVE.py`` mixed both kinds of fields under a historical MVE
name; Git history keeps that revision.
"""

from .._base_ import *

C.dataset_name = "MUSeg_DFormer"
C.rgb_format = ".jpg"
C.gt_format = ".png"
C.gt_transform = True
C.x_format = ".png"
C.x_is_single_channel = True
C.x_modal = ["d"]
C.num_classes = 15
C.class_names = [
    "person",
    "cable",
    "tube",
    "indicator",
    "metal fixture",
    "container",
    "tools & materials",
    "door",
    "electrical equipment",
    "electronic equipment",
    "mining equipment",
    "anchoring equipment",
    "support equipment",
    "rescue equipment",
    "rail area",
]

C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

C.backbone = "DFormerv2_S"
C.decoder = "ham"
C.decoder_embed_dim = 512
C.aux_rate = 0.0
C.drop_path_rate = 0.25
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.pad = False

C.eval_scale_array = [1.0]
C.eval_flip = False
C.eval_crop_size = [480, 640]
C.eval_stride_rate = 2 / 3
