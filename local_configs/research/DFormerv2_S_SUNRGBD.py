"""First-round engineering preset; importing this config never starts training."""
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from local_configs.SUNRGBD.DFormerv2_S import C as official_config

C = deepcopy(official_config)
config = C
C.backbone = "DFormerv2_S"
C.pretrained_model = "checkpoints/pretrained/DFormerv2_Small_pretrained.pth"
C.experiment_name = os.environ.get("DFORMER_EXPERIMENT_NAME", "dformerv2-s-sunrgbd-baseline")
C.dataset_path = str(Path(os.environ.get("DFORMER_DATASET_ROOT", "datasets")) / "SUNRGBD")
C.rgb_root_folder = str(Path(C.dataset_path) / "RGB")
C.gt_root_folder = str(Path(C.dataset_path) / "labels")
C.x_root_folder = str(Path(C.dataset_path) / "Depth")
C.train_source = str(Path(C.dataset_path) / "train.txt")
# Author protocol uses test.txt. Validation policy must be fixed before experiments.
C.eval_source = str(Path(C.dataset_path) / "test.txt")
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
C.log_dir = str(Path("outputs") / C.experiment_name / run_id)
C.log_dir_link = C.log_dir
C.checkpoint_dir = C.log_dir
C.tb_dir = str(Path(C.log_dir) / "tb")
C.log_file = str(Path(C.log_dir) / "train.log")
C.link_log_file = str(Path(C.log_dir) / "train-last.log")
C.val_log_file = str(Path(C.log_dir) / "validation.log")
C.link_val_log_file = str(Path(C.log_dir) / "validation-last.log")
C.swanlab_enabled = True
C.swanlab_project = "dformer-research"
C.swanlab_mode = "online"
C.swanlab_log_dir = str(Path(C.log_dir) / "swanlab")
