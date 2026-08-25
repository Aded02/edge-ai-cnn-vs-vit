# Edge AI: CNN vs Vision Transformer

> Benchmarking lightweight CNN and Vision Transformer architectures for image classification in resource-constrained edge environments.

## Overview

This project compares two image-classification approaches:

- **CNN baseline:** MobileNetV2
- **Transformer:** a lightweight custom Vision Transformer (ViT)

The study investigates the trade-off between **accuracy, inference latency, throughput (FPS), and model size**, and examines how TensorFlow Lite conversion and post-training quantisation affect deployment efficiency.

The project is based on my final-year Computer Systems Engineering dissertation.

## Research Question

> Can a lightweight Vision Transformer provide a competitive performance/efficiency trade-off against a lightweight CNN for edge-oriented image classification?

## Dataset

**CIFAR-10**

- 60,000 colour images
- 32 × 32 pixels
- 10 classes
- Images normalised to `[0, 1]`
- Labels one-hot encoded

The original study used reduced subsets and short training runs for rapid experimentation.

## Models

### MobileNetV2

MobileNetV2 was used as the CNN baseline because it is designed around efficient computation and is commonly associated with mobile and embedded workloads.

### Lightweight Vision Transformer

The ViT uses:

1. 32 × 32 × 3 input
2. 4 × 4 convolutional patch extraction
3. 64-dimensional patch embeddings
4. Multi-head self-attention
5. Layer normalisation
6. Dense feed-forward layers
7. Global average pooling
8. 10-class softmax output

## Experimental Setup

| Setting | Value |
|---|---|
| Dataset | CIFAR-10 |
| Fast-training subset | 10,000 training / 2,000 test |
| Batch size | 64 |
| Optimiser | Adam |
| Loss | Categorical cross-entropy |
| Fast-training epochs | 3 |
| Image size | 32 × 32 × 3 |
| Framework | TensorFlow / Keras |
| Edge format | TensorFlow Lite |

Both models were evaluated using:

- Classification accuracy
- Inference latency
- Frames per second (FPS)
- TFLite model size

## Results

The following values are the reported results from the dissertation's controlled experiment.

| Metric | CNN | ViT |
|---|---:|---:|
| Accuracy | 10.00% | **34.16%** |
| TFLite latency | 2.4 ms | **2.2 ms** |
| TFLite FPS | 412.40 | **449.81** |
| TFLite model size | 2.5452 MB | **0.1149 MB** |

### Optimisation impact

Reported latency reductions:

- CNN: **97.95%**
- ViT: **97.91%**

Reported FPS increases:

- CNN: **4779.69%**
- ViT: **4683.34%**

## Important Experimental Limitation

The project is **edge-oriented**, but the original experiment did **not** run on physical Raspberry Pi hardware. TensorFlow Lite inference was used to simulate an edge deployment workflow.

Therefore, this repository does **not** claim that the reported latency/FPS values are Raspberry Pi hardware benchmarks.

Physical Raspberry Pi deployment is a planned extension.

## Results Visualisation

### Accuracy

![Accuracy comparison](results/figures/accuracy.png)

### TFLite latency

![Latency comparison](results/figures/latency.png)

### TFLite FPS

![FPS comparison](results/figures/fps.png)

### TFLite model size

![Model size comparison](results/figures/model_size.png)

## Project Structure

```text
edge-ai-cnn-vs-vit/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── src/
│   ├── data.py
│   ├── models.py
│   └── train.py
│
├── optimization/
│   └── tflite_convert.py
│
├── benchmarking/
│   └── benchmark.py
│
├── results/
│   ├── benchmark_results.csv
│   └── figures/
│
├── notebooks/
│   └── README.md
│
└── docs/
    └── PROJECT_NOTES.md
```

## Running the Project

### 1. Clone

```bash
git clone https://github.com/YOUR-USERNAME/edge-ai-cnn-vs-vit.git
cd edge-ai-cnn-vs-vit
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train

```bash
python src/train.py
```

### 5. Convert to TensorFlow Lite

```bash
python optimization/tflite_convert.py
```

### 6. Benchmark

```bash
python benchmarking/benchmark.py
```

## Original Notebook

The dissertation appendix contains the original Google Colab notebook used for the university project. The repository version is organised into reusable Python modules rather than relying on one large notebook.

## What I Learned

- Deep-learning model construction
- CNN architecture selection
- Self-attention and Vision Transformers
- TensorFlow/Keras
- TensorFlow Lite
- Quantisation
- Edge-AI performance evaluation
- Benchmark design
- Interpreting accuracy/latency/throughput trade-offs
- Reproducible technical documentation

## Future Work

- Run both models on physical Raspberry Pi hardware
- Measure CPU utilisation, RAM and power consumption
- Compare FP32, FP16 and INT8 quantisation
- Train for longer and use the complete CIFAR-10 dataset
- Add stronger data augmentation
- Compare additional lightweight CNNs and ViTs
- Explore hardware acceleration
- Investigate CUDA/GPU inference
- Build a real-time camera demonstration

## Author

**Adedamola Bamigboye**

Computer Systems Engineering graduate · AI · Embedded Systems · Computer Architecture · Robotics · R&D
