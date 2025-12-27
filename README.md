# A Weakly Supervised Approach for Large-Scale Agricultural Parcel Extraction from VHR Imagery via Foundation Models and Adaptive Noise Correction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

WSPE is a framework for agricultural parcel extraction that leverages foundation models and adaptive noise correction, enabling accurate field boundary detection without manual annotations.

## Table of Contents

- [Overview](#overview)
- [Workflow](#workflow)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

APEX (Annotation-free Parcel EXtraction) is designed to extract agricultural field boundaries from very high-resolution (VHR) satellite imagery without requiring manual annotations. This repository contains the implementation of our framework that leverages foundation models and adaptive noise correction techniques.

## Workflow

![SS](image/Flowchart.jpg)

## Installation

```bash
# Clone the repository
git clone https://github.com/zhaowenpeng/WSPE.git
cd WSPE
```

## Datasets

The framework has been tested on the following datasets:

- [AI4Boundaries](http://data.europa.eu/89h/0e79ce5d-e4c8-4721-8773-59a4acf2c9c9) - European agricultural parcel dataset
- [FGFD](https://pan.baidu.com/s/1kdGAowJ2Dcqyn-dUQWLHJA?pwd=FGFD) - Fine-grained farmland dataset

## Usage

### APEX Training

```bash
python Train_WeakSupervise.py 
    --train-image-folder "/path/to/train/images" 
    --train-label-folder "/path/to/train/labels" 
    --valid-image-folder "/path/to/valid/images" 
    --valid-label-folder "/path/to/valid/labels" 
    --save-dir "/path/to/save/model" 
    --batch-size 4 
    --learning-rate 1e-4 
    --max-epochs 70 
    --warmup-epochs 7 
    --correct-epochs 14 
    --monitor-metric "val_iou" 
    --mixed-precision false
```

### Fully-Supervised Training

```bash
python Train_FullSupervise.py 
    --train-image-folder "/path/to/train/images" 
    --train-label-folder "/path/to/train/labels" 
    --valid-image-folder "/path/to/valid/images" 
    --valid-label-folder "/path/to/valid/labels" 
    --save-dir "/path/to/save/model" 
    --model "ablation" 
    --batch-size 4 
    --learning-rate 1e-4 
    --max-epochs 100 
    --monitor-metric "val_iou" 
    --mixed-precision false
```

### Inference

```bash
python Test_Infer.py 
    --image-folder "/path/to/test/images" 
    --model-path "/path/to/model" 
    --output-dir "/path/to/output" 
    --batch-size 1
```

## Acknowledgements

This project builds upon several excellent open-source projects:

- [Segment Anything 2](https://github.com/facebookresearch/sam2) - For SAM2
- [PyTorch](https://github.com/pytorch/pytorch) - For the deep learning framework
- [GR-KAN](https://github.com/Adamdad/kat) - For Group-KAN
- [TabPFN](https://github.com/PriorLabs/TabPFN) - For TabPFN
- [DBBANet](https://github.com/Henryjiepanli/DBBANet) - For model comparison

## License

This project is licensed under the MIT License - see the LICENSE file for details.
