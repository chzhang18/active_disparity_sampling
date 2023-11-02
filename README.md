# Active Disparity Sampling for Stereo Matching with Adjoint Network

## Abstract

The sparse signals provided by external sources have been leveraged as guidance for improving dense disparity estimation. However, previous methods assume depth measurements to be randomly sampled, which restricts performance improvements due to under-sampling in challenging regions and over-sampling in well-estimated areas. In this work, we introduce an Active Disparity Sampling problem that selects suitable sampling patterns to enhance the utility of depth measurements given arbitrary sampling budgets. We achieve this goal by learning an Adjoint Network for a deep stereo model to measure its pixel-wise disparity quality. Specifically, we design a hard-soft prior supervision mechanism to provide hierarchical supervision for learning the quality map. A Bayesian optimized disparity sampling policy is further proposed to sample depth measurements with the guidance of the disparity quality. Extensive experiments on standard datasets with various stereo models demonstrate that our method is suited and effective with different stereo architectures and outperforms existing fixed and adaptive sampling methods under different sampling rates. Remarkably, the proposed method makes substantial improvements when generalized to heterogeneous unseen domains.

## Requirements
```
python 3.7
PyTorch >= 1.1
torchvision >= 0.3
matplotlib
tensorboard
tensorboardX
scikit-image
opencv
```

## Data Preparation
Download [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
### Training
Run the script `./scripts/kitti12_active_train.sh` to train the Adjoint Network on the KITTI 2012 dataset. 
Please update `DATAPATH` in the bash file as your training data path. The corresponding checkpoints are in the `./checkpoints/kitti12/` folder. 

### Evaluation
Run the script `./scripts/kitti12_active_test.sh` to evaluation the proposed active disparity sampling method on the KITTI 2012 dataset. Please update `DATAPATH` in the bash file as your testing data path. The corresponding checkpoints are in the `./checkpoints/kitti12/` folder. 

# Citation
If you find this code useful in your research, please cite:

```
@article{zhang2023active,
  title={Active Disparity Sampling for Stereo Matching with Adjoint Network},
  author={Zhang, Chenghao and Meng, Gaofeng and Tian, Kun and Ni, Bolin and Xiang, Shiming},
  year={2023}
}
```
## Acknowledgements
This repository makes liberal use of code from[ \[GwcNet\]](https://github.com/xy-guo/GwcNet),[ \[deep-adaptive-LiDAR\]](https://github.com/alexanderbergman7/deep-adaptive-LiDAR)
