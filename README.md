
# Mistral 7B Fine-Tuning for Lesson Plan Data

This repository contains a Jupyter notebook that fine-tunes the Mistral 7B language model using lesson plan data. The notebook demonstrates how to fine-tune the model using LoRA adapters and the Unsloth library for efficient model training, making it ideal for creating personalized instructional materials.

## Requirements

The notebook installs and uses the following packages:
- `unsloth`: for efficient model loading and fine-tuning.
- `xformers`: Flash Attention support for faster model inference.
- `trl`, `peft`, `accelerate`, `bitsandbytes`, and `triton`: for model training and optimization.
- `wandb`: for tracking experiments.

## Key Features

- **LoRA Adapters**: The notebook fine-tunes the model using Low-Rank Adaptation (LoRA) to update a small subset of parameters for efficient training.
- **Memory Efficient**: The model can be loaded in 4-bit quantization, significantly reducing VRAM usage.
- **Flash Attention**: Xformers' Flash Attention support is included for faster inference and scaling to longer sequences.
- **Personalized Instructional Materials**: The notebook focuses on generating lesson plans tailored to various subjects and student needs.

## How to Use

1. Clone this repository to your local machine.
2. Open the notebook `Mistral_7B_Fine_tuning_with_Lesson_Plan_data.ipynb`.
3. Install the necessary packages by running the first cell.
4. Run through the notebook to load and fine-tune the Mistral 7B model.
5. Modify the lesson plan data as needed to fine-tune the model on different subjects or curricula.

## Model Details

- **Model**: `unsloth/mistral-7b-v0.3`
- **Quantization**: 4-bit quantization is used for faster training and inference on limited hardware.
- **LoRA Configuration**: Parameters like `q_proj`, `k_proj`, `v_proj`, and others are updated during fine-tuning to optimize the model for lesson plan generation.

## Fine-Tuning Process

The notebook provides the following fine-tuning steps:
1. Installing the necessary libraries and dependencies.
2. Loading the pre-trained Mistral 7B model with 4-bit quantization.
3. Adding LoRA adapters to selectively fine-tune the model's parameters.
4. Running the fine-tuning process on lesson plan data.
5. Optionally logging the fine-tuning process to Weights & Biases (`wandb`).

## License

This project uses the Apache 2.0 license as per the Mistral model's release license.

## Acknowledgements

- The `unsloth` library for efficient model handling and fine-tuning.
- The `xformers` package for Flash Attention and faster inference.
- The open-source community for developing these incredible tools.
