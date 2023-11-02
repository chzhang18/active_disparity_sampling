#!/usr/bin/env bash
set -x
DATAPATH="/data/zhangchenghao/KITTI2012/"
python test_active_sampling.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti12_train.txt --testlist ./filenames/kitti12_val.txt \
    --epochs 300 --lrepochs "200:10" \
    --model gwcnet-gc --logdir ./checkpoints/kitti12/active_sampling \
    --batch_size 2 \
    --test_batch_size 1