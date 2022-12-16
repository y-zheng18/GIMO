# GIMO: Gaze-Informed Human Motion Prediction in Context (ECCV 2022)
 ![demo](./assets/dataset_overview.jpg)

## Introduction

This is the official repo of our paper [GIMO: Gaze-Informed Human Motion Prediction in Context](https://arxiv.org/abs/2204.09443).

For more information, please visit our [project page](https://geometry.stanford.edu/projects/gimo/).

## Demo

A demo of our dataset:

<img src="./assets/demo.gif" alt="demo" width="480" align="left;">

Demo data can be downloaded from [**here**](https://drive.google.com/file/d/1cTF1zFcYbxAh8GZ5MsDK16C-iGubUgfn/view?usp=sharing)

## Quickstart

To setup the environment, firstly install the packages in requirements.txt:

```
pip install -r requirements.txt
```

Install PointNet++ as described [here](https://github.com/daerduoCarey/o2oafford/tree/main/exps) :

```
git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
cd Pointnet2_PyTorch
# [IMPORTANT] comment these two lines of code:
#   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
# [IMPORTANT] Also, you need to change l196-198 of file `[PATH-TO-VENV]/lib64/python3.8/site-packages/pointnet2_ops/pointnet2_modules.py` to `interpolated_feats = known_feats.repeat(1, 1, unknown.shape[1])`)
pip install -r requirements.txt
pip install -e .
```

Download and install [Vposer](https://github.com/nghorbani/human_body_prior), [SMPL-X](https://github.com/vchoutas/smplx)

Download the [pertained weight](https://drive.google.com/file/d/1P48SaFSrBguUDdY0FwXeFqZtuU7KfEyX/view?usp=sharing) and put it in the checkpoints folder

For a quickstart, run:

```
bash scripts/eval.sh
```

You can download the full dataset and have a test.

## Dataset
### Agreement
1. The GIMO dataset (the "Dataset") is available for **non-commercial** research purposes only. Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, as training data for a commercial product, for commercial ergonomic analysis (e.g. product design, architectural design, etc.), or production of other artifacts for commercial purposes including, for example, web services, movies, television programs, mobile applications, or video games. The dataset may not be used for pornographic purposes or to generate pornographic material whether commercial or not. The Dataset may not be reproduced, modified and/or made available in any form to any third party without our prior written permission.

2. You agree **not to** reproduce, modified, duplicate, copy, sell, trade, resell or exploit any portion of the images and any portion of derived data in any form to any third party without our prior written permission

3. You agree **not to** further copy, publish or distribute any portion of the Dataset. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.

4. Stanford University and Tsinghua University reserve the right to terminate your access to the Dataset at any time.

### Download Instructions 
The dataset is encrypted to prevent unauthorized access.

Please fill the [request form](./assets/GIMO_Dataset_Agreement.pdf) and send it to Yang Zheng (yzheng18@stanford.edu) to request the download link. 

By requesting for the link, you acknowledge that you have read the agreement, understand it, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Dataset.

### Dataset Structure & Visualization
You can refer to [README](./demo_data/README.md) for details.

### Citation
If you find this repo useful for your research, please consider citing:
```
@article{zheng2022gimo,
  title={GIMO: Gaze-Informed Human Motion Prediction in Context},
  author={Zheng, Yang and Yang, Yanchao and Mo, Kaichun and Li, Jiaman and Yu, Tao and Liu, Yebin and Liu, Karen and Guibas, Leonidas J},
  journal={arXiv preprint arXiv:2204.09443},
  year={2022}
}

```
