"""Direction-A frozen diagnostic config: SUN RGB-D, DFormerv2-S + HAM, 37 classes.

A deep copy of the ODG research config with the study protocol frozen for the
direction-A frozen-weight diagnostic:

* ``geometry_mode = 'original'`` -- the author's geometry path only.  Direction A
  is an *attention control* installed on the built model by
  ``research/evaluate_influence.py``; it is not a new ``geometry_mode`` and it
  never touches ``research/geometry.py`` or the ODG observation distribution.
* ``C.pad = True`` -- dev evaluation runs the whole padded 531x730 image, exactly
  the entry the historical clean/hole evaluations used (not a 480x480 window).
* online logging is off: the diagnostic writes its own CSV/JSON records.

Importing this module never trains and never reaches the official test split.
"""
from copy import deepcopy

from local_configs.research.ODG_SUNRGBD import C as _odg

C = deepcopy(_odg)
config = C

C.geometry_mode = "original"
C.pad = True

"""Frozen diagnostic protocol: fixed dev split, single scale, no flip."""
C.num_workers = 2
C.val_msf = False
C.val_flip = False
# The diagnostic drives its own paired conditions; the training-time artificial
# hole hook stays off.
C.artificial_hole_ratio = 0.0

"""No online logging for the diagnostic run."""
C.swanlab_enabled = False
C.swanlab_mode = "disabled"

C.experiment_name = "influence-sun-diagnostic"
