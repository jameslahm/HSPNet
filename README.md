# [Hierarchical Prompt Learning Using CLIP for Multi-label Classification with Single Positive Labels](https://dl.acm.org/doi/pdf/10.1145/3581783.3611988)

Official PyTorch Implementation of **HSPNet**, from the following paper:

[Hierarchical Prompt Learning Using CLIP for Multi-label Classification with Single Positive Labels](https://dl.acm.org/doi/pdf/10.1145/3581783.3611988). ACMMM 2023.

> Ao Wang, Hui Chen, Zijia Lin, Zixuan Ding, Pengzhang Liu, Yongjun Bao, Weipeng Yan, and Guiguang Ding

**Abstract**

Collecting full annotations to construct multi-label datasets is difficult and labor-consuming. As an effective solution to relieve the annotation burden, single positive multi-label learning (SPML) draws increasing attention from both academia and industry. It only annotates each image with one positive label, leaving other labels unobserved. Therefore, existing methods strive to explore the cue of unobserved labels to compensate for the insufficiency of label supervision. Though achieving promising performance, they generally consider labels independently, leaving out the inherent hierarchical semantic relationship among labels which reveals that labels can be clustered into groups. In this paper, we propose a hierarchical prompt learning method with a novel Hierarchical Semantic Prompt Network (HSPNet) to harness such hierarchical semantic relationships using a large-scale pretrained vision and language model, i.e., CLIP, for SPML. We first introduce a Hierarchical Conditional Prompt (HCP) strategy to grasp the hierarchical label-group dependency. Then we equip a Hierarchical Graph Convolutional Network (HGCN) to capture the high-order inter-label and inter-group dependencies. Comprehensive experiments and analyses on several benchmark datasets show that our method significantly outperforms the state-of-the-art methods, well demonstrating its superiority and effectiveness.

## Credit to previous work
This repository is built upon the code base of [ASL](https://github.com/Alibaba-MIIL/ASL) and [SPLC](https://github.com/xinyu1205/robust-loss-mlml), thanks very much!

## Performance

| Dataset | mAP | Ckpt | Log |
|:---: | :---: | :---: | :---: |
| COCO | 75.7 | [hspnet+coco.ckpt](https://github.com/jameslahm/HSPNet/releases/download/v1.0/hspnet+coco.ckpt)  | [hspnet+coco.txt](logs/hspnet+coco.txt) |
| VOC | 90.4 | [hspnet+voc.ckpt](https://github.com/jameslahm/HSPNet/releases/download/v1.0/hspnet+voc.ckpt) | [hspnet+voc.txt](logs/hspnet+voc.txt) |
| NUSWIDE | 61.8 | [hspnet+nuswide.ckpt](https://github.com/jameslahm/HSPNet/releases/download/v1.0/hspnet+nuswide.ckpt)  | [hspnet+nuswide.txt](logs/hspnet+nuswide.txt) |
| CUB | 24.3 | [hspnet+cub.ckpt]() | [hspnet+cub.txt](logs/hspnet+cub.txt) |

## Training

### COCO
```python
python train.py -c configs/hspnet+coco.yaml
```

### VOC
```python
python train.py -c configs/hspnet+voc.yaml
```

### NUSWIDE
```python
python train.py -c configs/hspnet+nuswide.yaml
```

### CUB
```python
python train.py -c configs/hspnet+cub.yaml
```

## Inference

> Note: Please place the pretrained checkpoint to checkpoints/hspnet+coco/round1/model-highest.ckpt

#### COCO
```python
python train.py -c configs/hspnet+coco.yaml -t -r 1
```

#### VOC
```python
python train.py -c configs/hspnet+voc.yaml -t -r 1
```

#### NUSWIDE
```python
python train.py -c configs/hspnet+nuswide.yaml -t -r 1
```

#### CUB
```python
python train.py -c configs/hspnet+cub.yaml -t -r 1
```

## Citation
```
@inproceedings{wang2023hierarchical,
  title={Hierarchical prompt learning using clip for multi-label classification with single positive labels},
  author={Wang, Ao and Chen, Hui and Lin, Zijia and Ding, Zixuan and Liu, Pengzhang and Bao, Yongjun and Yan, Weipeng and Ding, Guiguang},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={5594--5604},
  year={2023}
}
```