# Rock–Paper–Scissors CNN

A simple convolutional neural network (CNN) to classify images of rock, paper, and scissors using PyTorch.

## Project Structure

```text
mask_prediction/
├── data/                         # Your images: paper/, rock/, scissors/
├── datasets/                     # (optional) external RPS dataset
├── models/
│   └── model.py                  # CNN definition
├── data/
│   └── datamodule.py             # Transforms, dataset, dataloaders
├── train.py                      # Training script (saves best model)
├── predict.py                    # Inference on single image or folders
└── requirements.txt
```

## Setup

```bash
# Create and activate virtual env (optional)
python -m venv .venv
.venv\Scripts\activate  # on Windows

# Install dependencies
pip install -r requirements.txt
```

## Data

Place your images in:

```text
data/
  paper/
  rock/
  scissors/
```

Each subfolder contains images of that class.

## Training

```bash
python train.py
```

The best model (lowest validation loss) is saved to:

```text
models/model.pt
```

## Inference

Run predictions on a single image or test folders:

```bash
python predict.py
```

By default, `predict.py`:

- Loads `models/model.pt`
- Predicts a sample image (`sample/scissors2.webp`)
- Evaluates accuracy on:

```text
datasets/Rock-Paper-Scissors/test/paper
datasets/Rock-Paper-Scissors/test/rock
datasets/Rock-Paper-Scissors/test/scissors
```