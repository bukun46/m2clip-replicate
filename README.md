# M2CLIP Replicate

This is a replicate of the [M2CLIP](https://github.com/sallymmx/m2clip) project, a Multimodal, Multi-Task Adapting Framework for Video Action Recognition. The original project was presented at the AAAI 2024 conference.

## Table of Contents

- [M2CLIP Replicate](#m2clip-replicate)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Dataset Preparation](#dataset-preparation)
  - [Backbone](#backbone)
  - [Training and Evaluation](#training-and-evaluation)
  - [Acknowledgments](#acknowledgments)
  - [Checkpoints](#checkpoints)
  - [Training Conditions and Results](#training-conditions-and-results)

## Installation

To set up the environment, use `conda` with the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate m2clip
```

## Configuration

Some common configurations, such as dataset paths and pretrained backbone paths, are set in `configs/config.py`. You can just set the required fields to your own paths and parameters.

## Dataset Preparation
Follow the dataset preparation instructions in the original [M2CLIP](https://github.com/sallymmx/m2clip) repository. This replicate uses the Kinetics-400 dataset.

## Backbone
This replicate uses the CLIP-ViT-B/12 from the official release. You can download the [CLIP-ViT-B/16](https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/clip.py#L30) 

## Training and Evaluation
For training, run the following command:
```bash
python train.py 
```

For evaluation, run the following command:
```bash
python eval.py
```

## Acknowledgments

This project is based on the original [M2CLIP](https://github.com/sallymmx/m2clip) repository. Special thanks to the authors and contributors of the original project.

## Checkpoints

The checkpoints are saved in the `checkpoints` folder. You can download the checkpoints from the official repository.

## Training Conditions and Results
This replicate uses a single RTX4080 GPU for training. The model is trained with 32 batch size(36 in original project) and 12 epochs. Because of the restricted VRAM on my GPU, the training using video data is sampled with 8 frames, and the spatial & temporal views are set to 1 for both instead of 4 & 3 in the original project, which may lead to the performance drop.

**The final result for this replicate is as follows: Acc@1: 68.67%, Acc@5: 88.05%**




