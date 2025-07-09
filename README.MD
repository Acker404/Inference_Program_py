# Smoking Detection Inference Program

This program uses a pre-trained ResNet18 model to detect smoking in images. It provides inference capabilities using both PyTorch and ONNX runtimes.

## Environment Setup

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv Inference_Program_py_env
    ```

2.  **Activate the virtual environment:**
    ```bash
    source Inference_Program_py_env/bin/activate
    ```

## Package Installation

Install the required Python packages using pip:

```bash
pip install torch torchvision onnxruntime pillow numpy
```

## Usage

Run the main program to perform inference on a test image:

```bash
python main.py
```

The program will output the prediction from both the PyTorch and ONNX models.

### Models

*   **PyTorch Model:** `../models/smoke_model.pt`
*   **ONNX Model:** `../models/smoke_model.onnx`

### Output

The program will print the prediction to the console. The output will be an integer representing the predicted class:

*   `0`: non_smoking
*   `1`: smoking