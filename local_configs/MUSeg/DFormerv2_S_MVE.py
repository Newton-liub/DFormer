from .._base_ import *


C.dataset_name = "MUSeg_DFormer"
C.dataset_path = osp.join(C.root_dir, "MUSeg_DFormer")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".jpg"
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.gt_format = ".png"
C.gt_transform = True
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = True
C.x_modal = ["d"]
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True
C.num_train_imgs = 1595
C.num_eval_imgs = 1576
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
C.pretrained_model = None
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