# Delve into Source and Target Collaboration in Semi-supervised Domain Adaptation for Semantic Segmentation (ICME 2024 Oral)
This repository is released that can reproduce the main results (our proposed DSTC) of the experiment on GTA5 to Cityscapes. Experiments on the SYNTHIA to Cityscapes can be easily implemented by slightly modifying the dataset and setting.

## Paper
![image](https://github.com/EdenHazardan/DSTC-SSDA/blob/master/DSTC.PNG)
Delve into Source and Target Collaboration in Semi-supervised Domain Adaptation for Semantic Segmentation
Yuan Gao, Zilei Wang, Yixin, Zhang.
University of Science and Technology of China.

## Abstract
Abstract—Semi-supervised domain adaptation (SSDA) for semantic segmentation aims to train a model that performs well on target domain by learning from both fully-labeled source domain and partially-labeled target domain data. The key to this task is how to collaborate the labeled data from both domains, as well as the unlabeled data, to benefit the model training complementarily. In this paper, we innovatively achieve this goal from both data combination and model mergence perspectives. To this end, we propose a co-training framework based on siamese networks, where two networks are encouraged to learn from each other by cross-supervision with pseudo labels of unlabeled data. Meanwhile, for the labeled data, we enforce two networks to separately learn the knowledge dominated by source domain and target domain. Specifically, we propose domain- specific initialization and differentiated cross-domain combina- tion of labeled data. Moreover, we propose a target-preferred alignment method to encourage the source-biased network to optimize towards target domain, as the target-biased network is more in line with the task than the source-biased network. We conduct extensive experiments on two challenging benchmarks, and the results demonstrate the effectiveness of our method, which outperforms previous state-of-the-art methods with con- siderable performance improvement.

## Install & Requirements

The code has been tested on pytorch=1.8.0 and python3.8. Please refer to ``requirements.txt`` for detailed information.

### To Install python packages

```
pip install -r requirements.txt
```

## Download Pretrained Weights
For the segmentation model initialization, following DDM, we start with a model pretrained on ImageNet: [Download](http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth)


## Data preparation
You need to download the [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) datasets and [Cityscapes](https://www.cityscapes-dataset.com/) datasets.

Your directory tree should be look like this:
```
./DSTC-SSDA/data
├── cityscapes
|  ├── gtFine
|  |  |—— train
|  |  └── val
|  └── leftImg8bit
│       ├── train
│       └── val
├── GTA5
|  ├── images
|  └── labels 
```

To get the list of 100 annotated samples of target domain:

```
python DSTC-SSDA/data_split.py
```

## Training 
To realize DSI, we need warm-up model on source and target domain, respectively.

```
bash DSTC-SSDA/exp/warm-up/source_only/script/train.sh
bash DSTC-SSDA/exp/warm-up/target_100_only/train.sh
```

Then, we can train the segmentation model with DSTC.
```
bash DSTC-SSDA/exp/co-training/DSTC_100/script/train.sh
```
