# Tredence AI Engineering Intern - Case Study Solution

This repository contains the solution for the **AI Engineering Intern** case study at Tredence Analytics 

## Problem: The Self-Pruning Neural Network
The task was to design and implement a neural network that learns to prune its own weights dynamically during the training process. This is achieved through a custom prunable linear layer and a sparsity regularization loss function.

## Project Structure
- `self_pruning_net.py`: Main Python script containing the `PrunableLinear` layer, model definition, training loop, and evaluation logic.
- `REPORT.md`: A detailed report explaining the methodology, sparsity analysis, and experimental results.
- `gate_distribution.png`: Visualization of the final gate values showing the pruning effect.
- `TREDENCE CASE STUDY/`: Directory containing the original JD and case study problem description.
- `requirements.txt`: List of Python dependencies required to run the project.

## Key Features
- **Custom PrunableLinear Layer**: Implements a gated weight mechanism where each weight is multiplied by a learnable sigmoid-transformed "gate" parameter.
- **Dynamic Pruning**: The network uses an L1-regularization term on the gate parameters to encourage sparsity, effectively removing unnecessary connections during training.
- **CIFAR-10 Implementation**: The model is trained and evaluated on the CIFAR-10 dataset.
- **Multi-Lambda Experiments**: Comparison of model performance and sparsity levels across different regularization strengths ($\lambda$).

## How to Run
1. **Clone the repository**:
   ```bash
   git clone https://github.com/RohanSingh-official/tredence-ai-case-study.git
   cd tredence-ai-case-study
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the training script**:
   ```bash
   python self_pruning_net.py
   ```
   *Note: The script is currently configured to use a subset of CIFAR-10 for quick verification. You can modify the data loading section in `self_pruning_net.py` to use the full dataset.*

## Results Summary
For a detailed analysis of the accuracy-vs-sparsity trade-off, please refer to the [REPORT.md](REPORT.md) file.

---
**Author**: Rohan Singh
**Role**: AI Engineering Intern Applicant 
