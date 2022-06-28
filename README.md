# Awesome Weakly-supervised Object Localization

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)![GitHub stars](https://img.shields.io/github/stars/gyguo/awesome-weakly-supervised-object-localization?color=yellow)  ![GitHub forks](https://img.shields.io/github/forks/gyguo/awesome-weakly-supervised-object-localization?color=green&label=Fork)  ![visitors](https://visitor-badge.glitch.me/badge?page_id=gyguo.awesome-weakly-supervised-object-localization)

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

| Method | Pub.     | Bac.C           | Bac.L       | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------ |:--------:|:---------------:|:-----------:|:-----------:|:--------:|:-----------:|
|        |          | **Transformer** |             |             |          |             |
| TS-CAM | 2021ICCV | Deit-S          | -           | 71.3/83.8   | 87.7     | -/-         |
|        |          |                 |             |             |          |             |
|        |          |                 |             |             |          |             |
|        |          | **VGG**         |             |             |          |             |
| PSOL   | 2020CVPR | VGG16           | VGG16       | 66.3/84.1   | -        | -/-         |
| DANet  | 2019ICCV | VGG16           | -           | 52.5/62.0   | 67.7     | 75.4/92.3   |
| CAM    | 2016CVPR | VGG16           | -           | 44.2/52.2   | 56.0     | 76.6/92.5   |
| SPG    | 2018ECCV | VGG16           | -           | 48.9/57.9   | 58.9     | 75.5/92.1   |
|        |          |                 |             |             |          |             |
|        |          | **InceptionV3** |             |             |          |             |
| PSOL   | 2020CVPR | InceptionV3     | InceptionV3 | 65.5/83.4   | -        | -/-         |
| DANet  | 2019ICCV | InceptionV3     | -           | 49.5/60.5   | 67.0     | 71.2/90.6   |
| SPG    | 2018ECCV | InceptionV3     | -           | 46.6/57.7   | -        | -           |
|        |          |                 |             |             |          |             |
|        |          | **GoogleNet**   |             |             |          |             |
| CAM    | 2016CVPR | GoogleNet       | -           | 41.1/50.7   | -        | 73.8/91.5   |
|        |          |                 |             |             |          |             |
|        |          |                 |             |             |          |             |

## 1.2. Top1/5 results on ImageNet

| Method | Pub.      | Bac.C           | Bac.L       | Top-1/5 Loc | GT-Known | Top-1/5 Cls |
|:------ |:---------:|:---------------:|:-----------:|:-----------:|:--------:|:-----------:|
|        |           | **Transformer** |             |             |          |             |
| ViTOL  | 2022CVPRW | DeiT-B          | -           | 58.6/-      | 72.5     | 77.1/-      |
| TS-CAM | 2021ICCV  | Deit-S          | -           | 53.4/64.3   | 67.6     | -/-         |
|        |           |                 |             |             |          |             |
|        |           |                 |             |             |          |             |
|        |           | **VGG16**       |             |             |          |             |
| PSOL   | 2020CVPR  | VGG16           | VGG16       | 50.9/60.9   | 64.0     | -/-         |
|        |           |                 |             |             |          |             |
| CAM    | 2016CVPR  | VGG16           | -           | 42.8/54.9   | 59.0     | 68.8/88.6   |
|        |           |                 |             |             |          |             |
|        |           |                 |             |             |          |             |
|        |           | **InceptionV3** |             |             |          |             |
| PSOL   | 2020CVPR  | InceptionV3     | InceptionV3 | 54.8/63.3   | 65.2     | -/-         |
| SPG    | 2018ECCV  | InceptionV3     | -           | 48.6/60.0   | 64.7     |             |
| CAM    | 2016CVPR  | InceptionV3     | -           | 46.3/58.2   | 62.7     | 73.3/91.8   |
|        |           |                 |             |             |          |             |
|        |           |                 |             |             |          |             |

## 1.3. MaxBoxAccV2 results on CUB-200-2011

To do

## 1.3. MaxBoxAccV2 results on ImageNet

To do

# 2. Paper List

## 2022

- **ViTOL:** *2022CVPRW* ViTOL: Vision Transformer for Weakly Supervised Object Localization

## 2021

- **TS-CAM:** *2021ICCV* TS-CAM: Token Semantic Coupled Attention Map for Weakly Supervised Object Localization

## 2020

- **PSOL:** *2020CVPR* Rethinking the Route Towards Weakly Supervised Object Localization

## 2019

- **DANet:** *2019ICCV* DANet: Divergent Activation for Weakly Supervised Object Localization

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