# CER
The implementation code for the paper "Camera-aware Embedding Refinement for Unsupervised Person Re-identification."

The code for my paper has been uploaded to GitHub. Detailed instructions for deployment and proper attribution to any referenced code will be provided later. Thank you for your patience and understanding.

A new unsupervised person re-identification method is coming...

## 1. Introduction

Compared to other methods using 4gpus, our method uses a minimum of two Nvidia GTX 1080Ti for training.

## 2. Installation
- Python 3.7.10
- Pytorch 1.13.1
- Torchvision 0.14.1
- Faiss-gpu 1.6.4
- Please refer to `setup.py` for the other packages with the corresponding versions.

## 3. Preparation
1. Run `git clone https://github.com/Yimin-Liu/CER.git`
2. Prepare Datasets

---

Download the datasets Market-1501,MSMT17,DukeMTMC-reID from this [link](https://drive.google.com/file/d/19oWiYGjTgouFMK_psZvH8ysDGQ1KUbk-/view?usp=sharing) and unzip them under the directory like:

    CER/examples/data
    ├── market1501
    │   └── Market-1501-v15.09.15
    │       └── bounding_box_train
    │       └── bounding_box_test
    │       └── query
    └── msmt17
    │   └── MSMT17_V1
    │       └── bounding_box_train
    │       └── bounding_box_test
    │       └── query

Prepare ImageNet Pre-trained Models for IBN-Net

When training with the backbone of [IBN-ResNet](https://arxiv.org/abs/1807.09441), you need to download the ImageNet-pretrained model from this [link](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) and save it under the path of `examples/pretrained/`.

```
CER/examples
└── pretrained
    └── resnet50_ibn_a.pth.tar
```
## 4. Train
You can directly run `CER_*.sh ` file for the transferring training process.

```
sh CER_train_market1501.sh  ### for Market1501
sh CER_train_msmt17.sh  ### for MSMT17
```
## 5. Test

You can simply run `test_*.sh ` file for the transferring testing process.

```
sh CER_test_market1501.sh  ### for Market1501
sh CER_test_msmt17.sh  ### for MSMT17
```

## 6. References
[1] Our code is conducted based on [ClusterContrast](https://github.com/alibaba/cluster-contrast-reid) and [ISE](https://github.com/zhangxinyu-xyz/ISE-ReID.git).

<a name='8'></a>
