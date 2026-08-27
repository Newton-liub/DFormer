import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C

C.seed = 12345
C.channel_order = "BGR"
C.normalization_identity = "imagenet-rgb-statistics-in-array-order-v1"

# remoteip = os.popen('pwd').read()
C.root_dir = "../dataset"
C.abs_dir = osp.realpath(".")
