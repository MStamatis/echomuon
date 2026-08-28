"""Verify the --val-frac selection split is class-balanced after the shuffle fix.

Before the fix the slice was contiguous, and Tiny-ImageNet is decoded in wnid
order, so the val slice held only classes the train slice never showed (acc=0).
"""
import ast

import numpy as np

src = open("/lab/src/train.py").read()
ast.parse(src)
assert "permutation(len(train_y))" in src, "PATCH MISSING"
print("train.py: syntax OK, shuffle patch present\n")

for ds in ["cifar10", "cifar100", "tinyimagenet", "tinyimagenetn20"]:
    y = np.load("/lab/data/%s/train_y.npy" % ds)
    ncls = int(y.max()) + 1
    perm = np.random.default_rng(1234).permutation(len(y))  # exactly what the patch does
    ys = y[perm]
    cut = int(len(ys) * 0.9)
    tr, va = ys[:cut], ys[cut:]
    unseen = len(set(va.tolist()) - set(tr.tolist()))
    cnt = np.bincount(va, minlength=ncls)
    print("%-16s train %3d/%d classes | val %3d/%d | unseen-in-train %d | "
          "val per-class min/med/max %d/%d/%d"
          % (ds, len(set(tr.tolist())), ncls, len(set(va.tolist())), ncls, unseen,
             cnt.min(), int(np.median(cnt)), cnt.max()))
    assert unseen == 0, "%s STILL BROKEN" % ds
print("\nALL SPLITS CLASS-COMPLETE")
