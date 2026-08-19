import os
import os.path as osp

from .DFormerv2_S_MVE import C


# This configuration is intentionally absolute so cloud jobs are independent of cwd.
C.root_dir = "/root/rivermind-data/dataset"
C.dataset_path = osp.join(C.root_dir, "MUSeg_DFormer")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")

C.pretrained_model = "/root/rivermind-data/pretrained/DFormerv2_Small_pretrained.pth"
C.batch_size = 2
C.nepochs = 20
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 2
C.train_scale_array = [1.0]
C.warm_up_epoch = 2

C.checkpoint_start_epoch = C.nepochs
C.checkpoint_step = 5
C.save_epoch_checkpoints = True
C.save_latest_checkpoint = True

C.log_dir = "/root/rivermind-data/mve_outputs/museg_20epoch"
C.tb_dir = osp.join(C.log_dir, "tb")
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.join(C.log_dir, "checkpoint")
C.log_file = osp.join(C.log_dir, "train.log")
C.link_log_file = osp.join(C.log_dir, "log_last.log")
C.val_log_file = osp.join(C.log_dir, "val.log")
C.link_val_log_file = osp.join(C.log_dir, "val_last.log")

os.makedirs(C.log_dir, exist_ok=True)
os.makedirs(C.checkpoint_dir, exist_ok=True)