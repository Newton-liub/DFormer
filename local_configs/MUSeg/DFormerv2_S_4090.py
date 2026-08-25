"""Single-RTX-4090 MUSeg training configuration."""

import os
import os.path as osp

from .DFormerv2_S_MVE import C


_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
_DEFAULT_DATA_ROOT = osp.join(osp.dirname(_PROJECT_ROOT), "dataset")
_DATA_ROOT = osp.abspath(os.environ.get("DFORMER_DATA_ROOT", _DEFAULT_DATA_ROOT))
_OUTPUT_ROOT = osp.abspath(os.environ.get("DFORMER_OUTPUT_ROOT", osp.join(_PROJECT_ROOT, "outputs")))

C.root_dir = _DATA_ROOT
C.dataset_path = osp.join(C.root_dir, "MUSeg_DFormer")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.split_root = osp.join(_PROJECT_ROOT, "data", "splits", "MUSeg", "dev-v1")
C.experiment_phase = "development"
C.run_id = "museg-dformerv2-s-development-dev-v1"
C.train_source = osp.join(C.split_root, "train-dev.txt")
C.val_source = osp.join(C.split_root, "val-dev.txt")
C.test_source = osp.join(C.split_root, "official-test.txt")
C.eval_source = C.val_source  # legacy evaluator compatibility; training uses val_source
C.expected_split_sha256 = {
    "train": "a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470",
    "val": "1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83",
    "test": "12d9834215fcbfe696ad88321539c224850ff6fb66a01f48a02b1df478f48a4b",
}
C.expected_split_samples = {"train": 1277, "val": 318, "test": 1576}
C.num_train_imgs = 1277
C.num_eval_imgs = 318

C.pretrained_model = osp.abspath(
    os.environ.get(
        "DFORMER_PRETRAINED",
        osp.join(osp.dirname(_PROJECT_ROOT), "pretrained", "DFormerv2_Small_pretrained.pth"),
    )
)
C.optimizer = "AdamW"
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.val_batch_size = 1
C.nepochs = 20
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 8
C.train_scale_array = [1.0]
C.warm_up_epoch = 2

C.eval_start_epoch = 5
C.eval_interval = 5
C.save_interval = 5
C.checkpoint_start_epoch = C.nepochs  # legacy-only compatibility
C.checkpoint_step = C.save_interval  # legacy-only compatibility
C.save_epoch_checkpoints = True
C.save_latest_checkpoint = True

C.log_dir = osp.join(_OUTPUT_ROOT, "museg_dformerv2_s_4090", "development", "dev-v1")
C.tb_dir = osp.join(C.log_dir, "tb")
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.join(C.log_dir, "checkpoint")
C.log_file = osp.join(C.log_dir, "train.log")
C.link_log_file = osp.join(C.log_dir, "log_last.log")
C.val_log_file = osp.join(C.log_dir, "val.log")
C.link_val_log_file = osp.join(C.log_dir, "val_last.log")
