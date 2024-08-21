# Image-to-Image Translation with cGAN

This repository contains code and resources for performing Image-to-Image Translation using Conditional Generative Adversarial Networks (cGANs). The goal of this project is to translate images from one domain to another (e.g., turning black-and-white images into color, or transforming sketches into realistic images).

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Results](#results)
- [Directory Structure](#directory-structure)
- [Acknowledgments](#acknowledgments)

## Introduction
Image-to-Image Translation is a field in computer vision where the task is to transform an image from one domain into a corresponding image in another domain. This can be achieved by using Conditional Generative Adversarial Networks (cGANs), which are a variant of GANs that condition on some input data to generate the output.

In this project, we implement cGANs using a deep learning framework to perform image translation tasks. The network is trained to learn the mapping from input images to output images using paired or unpaired image datasets.

## Features
- Implement cGANs for image-to-image translation.
- Customizable architecture and training parameters.
- Supports various image translation tasks (e.g., colorization, style transfer).
- Save and load trained models for future use.
- Visualize results during and after training.

## Requirements
- Python 3.x
- TensorFlow or PyTorch
- NumPy
- Matplotlib
- OpenCV
- argparse

To install the required dependencies, you can run:

```bash
pip install -r requirements.txt
```

## Dataset
To train the cGAN, you'll need a dataset of paired images (i.e., input-output pairs) or unpaired images depending on the task.

For paired datasets, the input and output images should be aligned and stored in corresponding directories.

Example datasets:
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [Edges2Shoes](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)
- [Maps](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)

## Usage

### Training
To train the cGAN model on your dataset, run the following command:

```bash
python train.py --dataset_name <DATASET_NAME> --epochs <NUM_EPOCHS> --batch_size <BATCH_SIZE> --learning_rate <LEARNING_RATE>
```

Parameters:
- `dataset_name`: Name of the dataset folder.
- `epochs`: Number of epochs to train.
- `batch_size`: Size of the mini-batches.
- `learning_rate`: Learning rate for the optimizer.

### Testing
After training, you can test the model on unseen images using:

```bash
python test.py --model_path <MODEL_PATH> --input_image <INPUT_IMAGE_PATH> --output_image <OUTPUT_IMAGE_PATH>
```

Parameters:
- `model_path`: Path to the saved model.
- `input_image`: Path to the input image for translation.
- `output_image`: Path to save the translated output image.

## Results
During training, results will be saved in the `output` directory, including:
- Generated images at different epochs.
- Loss curves.
- Model checkpoints.

After training, you can evaluate the model's performance on test data.

## Directory Structure

```plaintext
├── datasets
│   └── <dataset_name>
│       ├── train
│       └── test
├── models
│   ├── generator.h5
│   └── discriminator.h5
├── output
│   ├── images
│   ├── losses
│   └── models
├── train.py
├── test.py
└── README.md
```

## Acknowledgments
This project is inspired by the work of [Pix2Pix](https://phillipi.github.io/pix2pix/) and [CycleGAN](https://junyanz.github.io/CycleGAN/). Special thanks to the open-source community for providing valuable datasets and resources.
