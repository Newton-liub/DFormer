import os
import os.path as osp
import time
import argparse
from cv2 import log

import torch
import torch.distributed as dist

from .logger import get_logger
from utils.pyt_utils import extant_file, link_file, ensure_dir
from utils.training_checkpoint import (
    CheckpointProtocol,
    atomic_save_checkpoint,
    create_training_checkpoint,
    inspect_checkpoint_directory,
    load_training_checkpoint,
    restore_training_state,
)

logger = get_logger()


class State(object):
    def __init__(self):
        self.epoch = 1
        self.iteration = 0
        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.checkpoint_protocol = None
        self.global_optimizer_step = 0
        self.best_val_miou = None
        self.best_val_epoch = None

    def register(self, **kwargs):
        allowed = {
            "epoch",
            "iteration",
            "dataloader",
            "model",
            "optimizer",
            "scaler",
            "checkpoint_protocol",
            "global_optimizer_step",
            "best_val_miou",
            "best_val_epoch",
            "resume_parent_run_id",
        }
        for key, value in kwargs.items():
            if key not in allowed:
                raise KeyError(f"unsupported engine state field: {key}")
            setattr(self, key, value)


class Engine(object):
    def __init__(self, custom_parser=None):
        logger.info("PyTorch Version {}".format(torch.__version__))
        self.state = State()
        self.devices = None
        self.distributed = False

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()

        self.continue_state_object = self.args.continue_fpath
        if "WORLD_SIZE" in os.environ:
            self.distributed = int(os.environ["WORLD_SIZE"]) > 1
        print(self.distributed)

        if self.distributed:
            # self.local_rank = self.args.local_rank
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(self.local_rank)
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            # os.environ['MASTER_PORT'] = self.args.port
            torch.distributed.init_process_group(backend="nccl")
            print(self.local_rank)
            self.devices = [0, 1]  # [i for i in range(self.world_size)]
        else:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.devices = [0, 1]  # parse_devices(self.args.devices)

        self.checkpoint_state = []

    def inject_default_parser(self):
        p = self.parser
        p.add_argument("-d", "--devices", default="", help="set data parallel training")
        p.add_argument(
            "-c",
            "--continue",
            type=extant_file,
            metavar="FILE",
            dest="continue_fpath",
            help="continue from one certain checkpoint",
        )
        p.add_argument("--local_rank", default=0, type=int, help="process rank on node")
        p.add_argument(
            "-p",
            "--port",
            type=str,
            default="16005",
            dest="port",
            help="port for init_process_group",
        )

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def update_iteration(self, epoch, iteration):
        self.state.epoch = epoch
        self.state.iteration = iteration

    def save_checkpoint(self, path):
        logger.info("Saving checkpoint to file {}".format(path))
        if not isinstance(self.state.checkpoint_protocol, CheckpointProtocol):
            raise RuntimeError("checkpoint protocol must be registered before saving")
        checkpoint = create_training_checkpoint(
            model=self.state.model,
            optimizer=self.state.optimizer,
            scaler=self.state.scaler,
            completed_epoch=self.state.epoch,
            global_optimizer_step=self.state.global_optimizer_step,
            best_val_miou=self.state.best_val_miou,
            best_val_epoch=self.state.best_val_epoch,
            protocol=self.state.checkpoint_protocol,
        )
        started_at = time.time()
        atomic_save_checkpoint(checkpoint, path)
        logger.info("Saved checkpoint to %s in %.2fs", path, time.time() - started_at)

    def link_tb(self, source, target):
        ensure_dir(source)
        ensure_dir(target)
        link_file(source, target)

    def save_and_link_checkpoint(self, checkpoint_dir, log_dir, log_dir_link, infor="", metric=None):
        if metric is None:
            raise ValueError("metric is required")
        ensure_dir(checkpoint_dir)
        if not osp.exists(log_dir_link):
            link_file(log_dir, log_dir_link)
        checkpoint = osp.join(checkpoint_dir, f"epoch-{self.state.epoch}{infor}.pth")
        self.save_checkpoint(checkpoint)

    def restore_checkpoint(self):
        if not isinstance(self.state.checkpoint_protocol, CheckpointProtocol):
            raise RuntimeError("checkpoint protocol must be registered before restoring")
        resume_path = osp.abspath(self.continue_state_object)
        if osp.basename(resume_path) == "latest.pth":
            checkpoint, _ = inspect_checkpoint_directory(
                osp.dirname(resume_path),
                expected_protocol=self.state.checkpoint_protocol,
                expected_checkpoint_run_id=getattr(self.state, "resume_parent_run_id", None),
            )
        else:
            checkpoint = load_training_checkpoint(
                resume_path,
                expected_protocol=self.state.checkpoint_protocol,
                expected_checkpoint_run_id=getattr(self.state, "resume_parent_run_id", None),
            )
        started_at = time.time()
        resume = restore_training_state(
            checkpoint,
            model=self.state.model,
            optimizer=self.state.optimizer,
            scaler=self.state.scaler,
            restore_rng=True,
        )
        self.state.epoch = resume.next_epoch
        self.state.iteration = 0
        self.state.global_optimizer_step = resume.global_optimizer_step
        self.state.best_val_miou = resume.best_val_miou
        self.state.best_val_epoch = resume.best_val_epoch
        logger.info(
            "Restored epoch-boundary checkpoint %s in %.2fs; next epoch is %d",
            resume_path,
            time.time() - started_at,
            resume.next_epoch,
        )
        return resume

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            logger.warning("A exception occurred during Engine initialization, give up running process")
            return False
