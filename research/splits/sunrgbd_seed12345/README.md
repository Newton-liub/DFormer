# SUN RGB-D development split (seed 12345)

Derived from the official `SUNRGBD/train.txt` (5285 lines).  The
official `train.txt` and `test.txt` are left unchanged.

- `dev.txt`      : 528 lines (fixed development/validation set)
- `train-dev.txt`: 4757 lines (development training set)

Algorithm: image-level partition, no scene grouping (the local SUN
RGB-D package ships no scene/sequence grouping file).  Indices are
shuffled with `random.Random(12345)` in Python's Mersenne Twister;
the selected indices are re-sorted so both files keep the original
`train.txt` line order.  Lines keep the author two-column format
`RGB/<name>.jpg labels/<name>.png`.  `dev.txt` and `train-dev.txt`
are disjoint and their union is the official train list.
