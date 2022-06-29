# Awesome Weakly-supervised Object Localization

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)![GitHub stars](https://img.shields.io/github/stars/gyguo/awesome-weakly-supervised-object-localization?color=yellow)  ![GitHub forks](https://img.shields.io/github/forks/gyguo/awesome-weakly-supervised-object-localization?color=green&label=Fork)  ![visitors](https://visitor-badge.glitch.me/badge?page_id=gyguo.awesome-weakly-supervised-object-localization)

# Table of Contents

- [1. Performance](#1-performance)
  * [1.1. Top1/5 results on CUB-200-2011](#11-top1-5-results-on-cub-200-2011)
    + [Transformer](#transformer)
    + [VGG](#vgg)
    + [InceptionV3](#inceptionv3)
    + [Others](#others)
  * [1.2. Top1/5 results on ImageNet](#12-top1-5-results-on-imagenet)
    + [Transformer](#transformer-1)
    + [VGG](#vgg-1)
    + [InceptionV3](#inceptionv3-1)
    + [Others](#others-1)
  * [1.3. MaxBoxAccV2 results on CUB-200-2011](#13-maxboxaccv2-results-on-cub-200-2011)
  * [1.4. MaxBoxAccV2 results on ImageNet](#14-maxboxaccv2-results-on-imagenet)
- [2. Paper List](#2-paper-list)
  * [2022](#2022)
  * [2021](#2021)
  * [2020](#2020)
  * [2019](#2019)
  * [2018](#2018)
  * [2017](#2017)
  * [2016](#2016)
- [3. Dataset](#3-dataset)
  * [CUB-200-2011](#cub-200-2011)
  * [ImageNet](#imagenet)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

------

**<u>Contact gyguo95@gmail.com if any paper is missed!</u>**

------

# 1. Performance

- **Bac. C:** backbone for classification 
- **Bac. L:** backbone for localization, **does not exist** for methods use a single network for classification and localization.
- **Top-1/Top-5 CLS:** is correct if the Top-1/Top-5 predict categories contain the correct label.
- **GT-known Loc** is correct when the intersection over union (IoU) between the ground-truth and the prediction is larger than 0.5 and does not consider whether the predicted category is correct.
- **Top-1/Top-5 Loc** is correct when Top-1/Top-5 CLS and GT-Known LOC are both correct.
- **"-"** indicates not exist.  **"?"** indicates the corresponding item is not mentioned in the paper.

## 1.1. Top1/5 results on CUB-200-2011

### Transformer

| Method | Pub.     | Bac.C  | Bac.L | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------ |:--------:|:------:|:-----:|:-----------:|:--------:|:-----------:|
| TS-CAM | 2021ICCV | Deit-S | -     | 71.3/83.8   | 87.7     | -/-         |
|        |          |        |       |             |          |             |

### VGG

| Method | Pub.     | Bac.C | Bac.L | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------ |:--------:|:-----:|:-----:|:-----------:|:--------:|:-----------:|
| CREAM  | 2022CVPR | VGG16 | -     | 70.4/85.7   | 91.0     |             |
| PSOL   | 2020CVPR | VGG16 | VGG16 | 66.3/84.1   | -        | -/-         |
| DANet  | 2019ICCV | VGG16 | -     | 52.5/62.0   | 67.7     | 75.4/92.3   |
| CAM    | 2016CVPR | VGG16 | -     | 44.2/52.2   | 56.0     | 76.6/92.5   |
| SPG    | 2018ECCV | VGG16 | -     | 48.9/57.9   | 58.9     | 75.5/92.1   |

### InceptionV3

| Method | Pub.     | Bac.C       | Bac.L       | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------ |:--------:|:-----------:|:-----------:|:-----------:|:--------:|:-----------:|
| CREAM  | 2022CVPR | InceptionV3 | -           | 71.8/86.4   | 90.4     | -/-         |
| PSOL   | 2020CVPR | InceptionV3 | InceptionV3 | 65.5/83.4   | -        | -/-         |
| DANet  | 2019ICCV | InceptionV3 | -           | 49.5/60.5   | 67.0     | 71.2/90.6   |
| SPG    | 2018ECCV | InceptionV3 | -           | 46.6/57.7   | -        | -           |

### Others

| Method | Pub.     | Bac.C     | Bac.L | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------ |:--------:|:---------:|:-----:|:-----------:|:--------:|:-----------:|
| CAM    | 2016CVPR | GoogleNet | -     | 41.1/50.7   | -        | 73.8/91.5   |

## 1.2. Top1/5 results on ImageNet

### Transformer

| Method | Pub.      | Bac.C  | Bac.L | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------ |:---------:|:------:|:-----:|:-----------:|:--------:|:-----------:|
| ViTOL  | 2022CVPRW | DeiT-B | -     | 58.6/-      | 72.5     | 77.1/-      |
| TS-CAM | 2021ICCV  | Deit-S | -     | 53.4/64.3   | 67.6     | -/-         |

### VGG

| Method | Pub.     | Bac.C | Bac.L | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------ |:--------:|:-----:|:-----:|:-----------:|:--------:|:-----------:|
| CREAM  | 2022CVPR | VGG16 | -     | 52.4/64.2   | 68.3     | -/-         |
| PSOL   | 2020CVPR | VGG16 | VGG16 | 50.9/60.9   | 64.0     | -/-         |
| CAM    | 2016CVPR | VGG16 | -     | 42.8/54.9   | 59.0     | 68.8/88.6   |

### InceptionV3

| Method | Pub.     | Bac.C       | Bac.L       | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------ |:--------:|:-----------:|:-----------:|:-----------:|:--------:|:-----------:|
| CREAM  | 2022CVPR | InceptionV3 | -           | 56.1/66.2   | 69.0     | -/-         |
| PSOL   | 2020CVPR | InceptionV3 | InceptionV3 | 54.8/63.3   | 65.2     | -/-         |
| SPG    | 2018ECCV | InceptionV3 | -           | 48.6/60.0   | 64.7     |             |
| CAM    | 2016CVPR | InceptionV3 | -           | 46.3/58.2   | 62.7     | 73.3/91.8   |

### Others

| Method | Pub.     | Bac.C     | Bac.L | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------ |:--------:|:---------:|:-----:|:-----------:|:--------:|:-----------:|
| CAM    | 2016CVPR | GoogleNet | -     | 41.1/50.7   | -        | 73.8/91.5   |

## 1.3. MaxBoxAccV2 results on CUB-200-2011

To do

## 1.4. MaxBoxAccV2 results on ImageNet

To do

# 2. Paper List

## 2022

- **CREAM:** *2022CVPR* CREAM:  Weakly Supervised Object Localization via Class RE-Activation Mapping
- **ViTOL:** *2022CVPRW* ViTOL: Vision Transformer for Weakly Supervised Object Localization
- 2022TPAMI Evaluation for Weakly Supervised Object Localization Protocol, Metrics, and Datasets
- 2022TNNLS Diverse Complementary Part Mining for Weakly Supervised Object Localization
- 2022PR Gradient-based refined class activation map for weakly supervised object localization
- 2022arxiv Learning Consistency from High-quality Pseudo-labels for Weakly Supervised Object Localization

## 2021

- **TS-CAM:** *2021ICCV* TS-CAM: Token Semantic Coupled Attention Map for Weakly Supervised Object Localization
- 2021TIP Multi-Scale Low-Discriminative Feature Reactivation for Weakly Supervised Object Localization
- 2021TIP LayerCAM: Exploring Hierarchical Class Activation Maps for Localization
- 2021PR Region-based dropout with attention prior for weakly supervised object localization
- 2021arxiv Background-aware Classification Activation Map for Weakly Supervised Object Localization
- 2021arxiv MinMaxCAM Improving object coverage for CAM-based Weakly Supervised Object Localization
- 2021arxiv Weakly Supervised Foreground Learning for Weakly Supervised Localization and Detection

## 2020

- **PSOL:** *2020CVPR* Rethinking the Route Towards Weakly Supervised Object Localization
- 2020arxiv Rethinking Localization Map Towards Accurate Object Perception with Self-Enhancement Maps

## 2019

- **DANet:** *2019ICCV* DANet: Divergent Activation for Weakly Supervised Object Localization
- 2019arxiv Dual-attention Focused Module for Weakly Supervised Object Localization
- 2019arxiv Weakly Supervised Localization Using Background Images
- 2019arxiv Weakly Supervised Object Localization with Inter-Intra Regulated CAMs

## 2018

- **ACoL:** *2018CVPR* Adversarial Complementary Learning for Weakly Supervised Object Localization
- **SPG:** *2018ECCV* Self-produced Guidance for Weakly-supervised Object Localization

## 2017

- **Grad-CAM:** *2017ICCV* Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
- **HaS:** *2017ICCV* Hide-and-Seek Forcing a Network to be Meticulous for Weakly-supervised Object and Action Localization

## 2016

- **CAM:** *2016CVPR* Learning Deep Features for Discriminative Localization

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
