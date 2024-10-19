# CycleGAN Face-Sketch Translation

This repository implements a CycleGAN model for translating between face sketches and real face images using PyTorch. The dataset consists of face sketches paired with corresponding real face images, and the goal is to map sketches to real faces and vice versa. The model architecture includes Residual Blocks in the Generator and uses a PatchGAN Discriminator. 

## Table of Contents
- [CycleGAN Overview](#cyclegan-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Generating Images](#generating-images)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)

## CycleGAN Overview

CycleGAN is a type of Generative Adversarial Network (GAN) that enables image translation between two domains (e.g., sketches and photos) without needing paired examples. It uses two Generators to map images between the domains and two Discriminators to distinguish real and fake images in each domain. It also includes cycle-consistency loss to ensure that images translated from one domain to the other can be translated back to the original domain.

## Project Structure

```
.
├── data/
│   └── train/ (Dataset directory for training)
│       ├── photos/ (Real face images)
│       └── sketches/ (Face sketches)
├── checkpoints/ (Generated models are saved here)
├── main.py (Script for training and generating images)
└── README.md
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/cyclegan-face-sketch.git
    cd cyclegan-face-sketch
    ```

2. Install dependencies:
    ```bash
    pip install torch torchvision Pillow
    ```

3. Ensure you have a CUDA-compatible GPU for faster training.

## Training the Model

1. Prepare your dataset, ensuring it follows this structure:
    ```
    dataset/
    └── train/
        ├── photos/
        └── sketches/
    ```

2. Train the CycleGAN model:
    ```bash
    python main.py
    ```

    - You can specify the dataset directory by modifying the `root_dir` variable in `main.py`.
    - Checkpoints will be saved every 10 epochs.

## Generating Images

Once the model is trained, you can generate images by loading a trained model and passing an input image for transformation.

### Example Usage:

```bash
python main.py --generate --model-path path/to/checkpoint.pth --input-path path/to/input_image.jpg --output-path path/to/output_image.jpg --direction AtoB
```

- `--generate`: Flag to switch to image generation mode.
- `--model-path`: Path to the saved model checkpoint.
- `--input-path`: Path to the input image (photo or sketch).
- `--output-path`: Path where the output image will be saved.
- `--direction`: Specify the direction of transformation (`AtoB` or `BtoA`).

## Dataset

You will need a dataset of paired face sketches and real face images. The data should be divided into two directories:
- `photos/`: Containing the real face images.
- `sketches/`: Containing the corresponding face sketches.

The used dataset is https://www.kaggle.com/datasets/almightyj/person-face-sketches.

## Model Architecture

### Generator
The generator architecture follows the residual block pattern to ensure smoother transformations. The model consists of:
- An initial convolution layer.
- Downsampling layers.
- A series of residual blocks.
- Upsampling layers.
- An output convolution layer.

### Discriminator
The discriminator is based on a PatchGAN architecture, which classifies whether 70x70 image patches are real or fake. This model progressively downsamples the input image using convolution layers with stride 2 and LeakyReLU activations.

## Contributing

If you would like to contribute, feel free to submit a pull request with bug fixes, optimizations, or new features!

## License

This project is licensed under the MIT License.
