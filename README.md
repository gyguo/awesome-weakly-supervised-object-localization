# Awesome Weakly-supervised Object Localization

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)![GitHub stars](https://img.shields.io/github/stars/gyguo/awesome-weakly-supervised-object-localization?color=yellow)  ![GitHub forks](https://img.shields.io/github/forks/gyguo/awesome-weakly-supervised-object-localization?color=green&label=Fork)

# Table of Contents
- [1. Paper List](#1-paper-list)
- [2. Performance](#2-performance)
  * [2.1. Top1/5 results on CUB-200-2011](#21-results-on-cub-200-2011)
    + [Transformer](#transformer)
    + [VGG](#vgg)
    + [InceptionV3](#inceptionv3)
    + [Others](#others)
  * [2.2. Top1/5 results on ImageNet](#22-results-on-imagenet)
    + [Transformer](#transformer-1)
    + [VGG](#vgg-1)
    + [InceptionV3](#inceptionv3-1)
    + [Others](#others-1)
- [3. Dataset](#3-dataset)
  * [CUB-200-2011](#cub-200-2011)
  * [ImageNet](#imagenet)
- [4. Awesome-list of Weakly-supervised Learning from Our Team](#4-awesome-list-of-weakly-supervised-learning-from-our-team)

------

**<u>Contact gyguo95@gmail.com if any paper is missed!</u>**

------



# 1. Paper List
## 2024
- ***2024IJCAI*** A Consistency and Integration Model with Adaptive Thresholds for Weakly Supervised Object Localization
- ***2024CVPR*** CAM Back Again: Large Kernel CNNs from a Weakly Supervised Object Localization Perspective
- ***2024TPAMI*** Boosting Weakly Supervised Object Localization and Segmentation With Domain Adaption
- ***2024PR*** Discovering an inference recipe for weakly-supervised object localization
- ***2024TNNLS*** Adaptive Zone Learning for Weakly Supervised Object Localization
- ***2024PR*** Semantic-Constraint Matching for transformer-based weakly supervised object localization


## 2023
- ***2023TPAMI*** Evaluation for Weakly Supervised Object Localization: Protocol, Metrics, and Datasets
- ***2023ACM MM*** LocLoc: Low-level Cues and Local-area Guides for Weakly Supervised Object Localization
- **WEND:** ***2023ACM MM*** Rethinking the Localization in Weakly Supervised Object Localization
- **GenPromp:** ***2023ICCV*** Generative Prompt Model for Weakly Supervised Object Localization
- ***2023PR***  Weakly supervised foreground learning for weakly supervised localization and detection

## 2022
- ***2022CVPR*** C2AM: Contrastive Learning of Class-Agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation
- ***2022ECCV*** Bagging regional classification activation maps for weakly supervised object localization
- **CREAM:** ***2022CVPR*** CREAM:  Weakly Supervised Object Localization via Class RE-Activation Mapping
- **DA-WSOL:** ***2022CVPR*** Weakly Supervised Object Localization as Domain Adaption
- **AlignMix:** ***2022CVPR*** AlignMix: Improving representation by interpolating aligned features
- **ViTOL:** ***2022CVPRW*** ViTOL: Vision Transformer for Weakly Supervised Object Localization
- ***2022TNNLS*** Diverse Complementary Part Mining for Weakly Supervised Object Localization
- ***2022TNNLS*** Generalized Weakly Supervised Object Localization
- ***2022PR*** Gradient-based refined class activation map for weakly supervised object localization
- ***2022TMM*** Dual-Gradients Localization Framework With Skip-Layer Connections for Weakly Supervised Object Localization
- ***2022ICMR*** FreqCAM: Frequent Class Activation Map for Weakly Supervised Object Localization
- **SCM:2022ECCV** Weakly Supervised Object Localization via Transformer with Implicit Spatial Calibration
- ***2022arxiv*** Learning Consistency from High-quality Pseudo-labels for Weakly Supervised Object Localization

## 2021

- **SLT-Net: 2021CVPR**: Strengthen Learning Tolerance for Weakly Supervised Object Localization
- **TS-CAM:** ***2021ICCV*** TS-CAM: Token Semantic Coupled Attention Map for Weakly Supervised Object Localization
- ***2021TIP*** Multi-Scale Low-Discriminative Feature Reactivation for Weakly Supervised Object Localization
- ***2021TIP*** LayerCAM: Exploring Hierarchical Class Activation Maps for Localization
- ***2021PR*** Region-based dropout with attention prior for weakly supervised object localization
- ***2021arxiv*** Background-aware Classification Activation Map for Weakly Supervised Object Localization
- ***2021arxiv*** MinMaxCAM Improving object coverage for CAM-based Weakly Supervised Object Localization
- ***2021arxiv*** Weakly Supervised Foreground Learning for Weakly Supervised Localization and Detection

## 2020

- **PSOL:** ***2020CVPR*** Rethinking the Route Towards Weakly Supervised Object Localization
- ***2020CVPR*** Evaluating Weakly Supervised Object Localization Methods Right
- **MEIL:** ***2020CVPR*** Erasing Integrated Learning  A Simple yet Effective Approach for Weakly Supervised Object Localization
- **GC-Net:** ***2020ECCV*** Geometry Constrained Weakly Supervised Object Localization
- ***I2C:*** ***2020ECCV*** Inter-Image Communication for Weakly Supervised Localization
- ***2020ECCV*** Pairwise Similarity Knowledge Transfer for Weakly Supervised Object Localization
- ***2020ICPR*** Dual-attention Guided Dropblock Module for Weakly Supervised Object Localization
- ***2020arxiv*** Rethinking Localization Map Towards Accurate Object Perception with Self-Enhancement Maps

## 2019

- **ADL:** ***2019CVPR*** Attention-based Dropout Layer for Weakly Supervised Object Localization
- **DANet:** ***2019ICCV*** DANet: Divergent Activation for Weakly Supervised Object Localization
- **CutMix:** ***2019ICCV*** CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
- ***2019ICLR*** Marginalized average attentional network for weakly-supervised learning
- ***2019arxiv*** Dual-attention Focused Module for Weakly Supervised Object Localization
- ***2019arxiv*** Weakly Supervised Localization Using Background Images
- ***2019arxiv*** Weakly Supervised Object Localization with Inter-Intra Regulated CAMs

## 2018

- **ACoL:** ***2018CVPR*** Adversarial Complementary Learning for Weakly Supervised Object Localization
- **SPG:** ***2018ECCV*** Self-produced Guidance for Weakly-supervised Object Localization

## 2017

- **Grad-CAM:** ***2017ICCV*** Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
- **HaS:** ***2017ICCV*** Hide-and-Seek Forcing a Network to be Meticulous for Weakly-supervised Object and Action Localization

## 2016

- **CAM:** ***2016CVPR*** Learning Deep Features for Discriminative Localization


# 2. Performance
**Performance will no be updated anymore**
- **Bac. C:** backbone for classification 
- **Bac. L:** backbone for localization, **does not exist** for methods use a single network for classification and localization.
- **Top-1/Top-5 CLS:** is correct if the Top-1/Top-5 predict categories contain the correct label.
- **GT-known Loc** is correct when the intersection over union (IoU) between the ground-truth and the prediction is larger than 0.5 and does not consider whether the predicted category is correct.
- **Top-1/Top-5 Loc** is correct when Top-1/Top-5 CLS and GT-Known LOC are both correct.
- **"-"** indicates not exist.  **"?"** indicates the corresponding item is not mentioned in the paper.

## 2.1. Results on CUB-200-2011

### Transformer

| Method   | Pub.      | Bac.C           | Bac.L    | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:-------- |:---------:|:---------------:|:--------:|:-----------:|:--------:|:-----------:|
| GenPromp | 2023CVPR  | EfficientNet-B7 | -        | 87.0/96.1   | 98.0     | -/-         |
| WEND     | 2023ACMMM | EfficientNet-B7 | ResNet50 | 83.77/93.84 | 95.78    | -/-         |
| SCM      | 2022ECCV  | Deit-S          | -        | 76.4/91.6   | 96.6     | 78.5/94.5   |
| TS-CAM   | 2021ICCV  | Deit-S          | -        | 71.3/83.8   | 87.7     | -/-         |

### VGG

| Method  | Pub.     | Bac.C | Bac.L | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------- |:--------:|:-----:|:-----:|:-----------:|:--------:|:-----------:|
| CREAM   | 2022CVPR | VGG16 | -     | 70.4/85.7   | 91.0     | -/-         |
| SLT-Net | 2021CVPR | VGG16 | VGG16 | 67.8/-      | 87.6     | 76.6/-      |
| PSOL    | 2020CVPR | VGG16 | VGG16 | 66.3/84.1   | -        | -/-         |
| GC-Net  | 2020ECCV | VGG16 | VGG16 | 63.2/75.5   | 81.1     | 76.8/92.3   |
| MEIL    | 2020CVPR | VGG16 | -     | 57.5/-      | 73.8     | 74.8/-      |
| DANet   | 2019ICCV | VGG16 | -     | 52.5/62.0   | 67.7     | 75.4/92.3   |
| CutMix  | 2019ICCV | VGG16 | -     | 52.5/-      | -        | -           |
| ADL     | 2019CVPR | VGG16 | -     | 52.4/-      | 75.4     | 65.3/-      |
| CAM     | 2016CVPR | VGG16 | -     | 44.2/52.2   | 56.0     | 76.6/92.5   |
| SPG     | 2018ECCV | VGG16 | -     | 48.9/57.9   | 58.9     | 75.5/92.1   |

### InceptionV3

| Method  | Pub.     | Bac.C       | Bac.L       | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------- |:--------:|:-----------:|:-----------:|:-----------:|:--------:|:-----------:|
| CREAM   | 2022CVPR | InceptionV3 | -           | 71.8/86.4   | 90.4     | -/-         |
| SLT-Net | 2021CVPR | InceptionV3 | VGG16       | 66.1/-      | 86.5     | 76.4/-      |
| PSOL    | 2020CVPR | InceptionV3 | InceptionV3 | 65.5/83.4   | -        | -/-         |
| I2C     | 2020ECCV | InceptionV3 |             | 56.0/68.3   | 72.6     | -/-         |
| DANet   | 2019ICCV | InceptionV3 | -           | 49.5/60.5   | 67.0     | 71.2/90.6   |
| ADL     | 2019CVPR | InceptionV3 | -           | 53.0/-      | -        | 74.6/-      |
| SPG     | 2018ECCV | InceptionV3 | -           | 46.6/57.7   | -        | -           |

### Others

| Method  | Pub.     | Bac.C         | Bac.L | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------- |:--------:|:-------------:|:-----:|:-----------:|:--------:|:-----------:|
|         |          | **ResNet50**  |       |             |          |             |
| DA-WSOL | 2022CVPR | ResNet50      | -     | 66.8/-      | 82.3     | -/-         |
| CutMix  | 2019ICCV | ResNet50      | -     | 54.81/-     | -        | -/-         |
|         |          | **GoogleNet** |       |             |          |             |
| CAM     | 2016CVPR | GoogleNet     | -     | 41.1/50.7   | -        | 73.8/91.5   |

## 2.2. Results on ImageNet

### Transformer

| Method   | Pub.      | Bac.C           | Bac.L | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:-------- |:---------:|:---------------:|:-----:|:-----------:|:--------:|:-----------:|
| GenPromp | 2023ICCV  | EfficientNet-B7 | -     | 65.2/73.4   | 75.0     | -/-         |
| ViTOL    | 2022CVPRW | DeiT-B          | -     | 58.6/-      | 72.5     | 77.1/-      |
| SCM      | 2022ECCN  | Deit-S          | -     | 56.1/66/4   | 68.8     | 76.7/93.0   |
| TS-CAM   | 2021ICCV  | Deit-S          | -     | 53.4/64.3   | 67.6     | -/-         |

### VGG

| Method  | Pub.     | Bac.C | Bac.L       | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------- |:--------:|:-----:|:-----------:|:-----------:|:--------:|:-----------:|
| CREAM   | 2022CVPR | VGG16 | -           | 52.4/64.2   | 68.3     | -/-         |
| SLT-Net | 2021CVPR | VGG16 | InceptionV3 | 51.2/62.4   | 67.2     | 72.4/-      |
| PSOL    | 2020CVPR | VGG16 | VGG16       | 50.9/60.9   | 64.0     | -/-         |
| I2C     | 2020ECCV | VGG16 | -           | 47.4/58.5   | 63.9     | 69.4/89.3   |
| MEIL    | 2020CVPR | VGG16 | -           | 46.8/-      | -        | 70.3/-      |
| ADL     | 2019CVPR | VGG16 | -           | 44.9/-      | -        | 69.5/-      |
| CAM     | 2016CVPR | VGG16 | -           | 42.8/54.9   | 59.0     | 68.8/88.6   |

### InceptionV3

| Method  | Pub.     | Bac.C       | Bac.L       | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------- |:--------:|:-----------:|:-----------:|:-----------:|:--------:|:-----------:|
| CREAM   | 2022CVPR | InceptionV3 | -           | 56.1/66.2   | 69.0     | -/-         |
| SLT-Net | 2021CVPR | InceptionV3 | InceptionV3 | 55.7/65.4   | 67.6     | 78.1/-      |
| PSOL    | 2020CVPR | InceptionV3 | InceptionV3 | 54.8/63.3   | 65.2     | -/-         |
| I2C     | 2020ECCV | InceptionV3 | -           | 53.1/64.1   | 68.5     | 73.3/91.6   |
| GC-Net  | 2020ECCV | InceptionV3 | InceptionV3 | 49.1/58.1   | -        | 77.4/93.6   |
| MEIL    | 2020CVPR | InceptionV3 | -           | 49.5/-      | -        | 73.3/-      |
| ADL     | 2019CVPR | InceptionV3 | -           | 48.7/-      | -        | 72.8/-      |
| SPG     | 2018ECCV | InceptionV3 | -           | 48.6/60.0   | 64.7     |             |
| CAM     | 2016CVPR | InceptionV3 | -           | 46.3/58.2   | 62.7     | 73.3/91.8   |

### Others
| Method  | Pub.     | Bac.C         | Bac.L | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------- |:--------:|:-------------:|:-----:|:-----------:|:--------:|:-----------:|
|         |          | **ResNet50**  |       |             |          |             |
| DA-WSOL | 2022CVPR | ResNet50      | -     | 54.9/-      | 70.2     | -/-         |
| CutMix  | 2019ICCV | ResNet50      | -     | 47.25/-     | -        | 78.6/94.1   |
|         |          | **GoogleNet** |       |             |          |             |
| CAM     | 2016CVPR | GoogleNet     | -     | 41.1/50.7   | -        | 73.8/91.5   |


# 3. Dataset

## CUB-200-2011

```
@article{wah2011caltech,
 title={The caltech-ucsd birds-200-2011 dataset},
 author={Wah, Catherine and Branson, Steve and Welinder, Peter and Perona, Pietro and Belongie, Serge},
 year={2011},
 publisher={California Institute of Technology}
}
```
## ImageNet

```
@inproceedings{deng2009imagenet,
  title={Imagenet: A large-scale hierarchical image database},
  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  booktitle={2009 IEEE conference on computer vision and pattern recognition},
  pages={248--255},
  year={2009},
  organization={Ieee}
}
```

---
# 4. Awesome-list of Weakly-supervised Learning from Our Team
- [Awesome Weakly-supervised Semantic Segmentation](https://github.com/gyguo/awesome-weakly-supervised-semantic-segmentation)
- [Awesome Weakly-supervised Action Localization](https://github.com/VividLe/awesome-weakly-supervised-action-localization)
- [Awesome Weakly-supervised Object Localization](https://github.com/gyguo/awesome-weakly-supervised-object-localization)
