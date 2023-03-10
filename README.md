# DSTC-SSDA
This repository is released for double-blind submission, which can reproduce the main results (our proposed DSTC) of the experiment on VIPER to Cityscapes-Seq.  Experiments on the SYNTHIA-Seq to Cityscapes-Seq can be easily implemented by slightly modifying the dataset and setting.

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
To realize CI, we need warm-up model on source and target domain, respectively.

```
bash DSTC-SSDA/exp/warm-up/source_only/script/train.sh
bash DSTC-SSDA/exp/warm-up/target_100_only/train.sh
```

Then, we can train the segmentation model with DSTC.
```
bash DSTC-SSDA/exp/co-training/DSTC_100/script/train.sh
```
