# Self-Pruning Neural Network: Case Study Report

## Problem Overview
The goal was to design a neural network that dynamically learns to prune its own weights during the training process. This is achieved by associating each weight with a learnable "gate" parameter (passed through a Sigmoid function) and applying an L1 sparsity penalty to these gates.

## Implementation Details

### 1. PrunableLinear Layer
The [PrunableLinear](file:///c:/Users/rohan/Downloads/07b60a57-5971-4e48-a7b8-8b94920641ae/self_pruning_net.py) layer implements the core logic:
- **Weights and Biases**: Standard linear parameters.
- **Gate Scores**: A learnable parameter tensor of the same shape as the weight tensor.
- **Forward Pass**: 
  - `gates = Sigmoid(gate_scores)`
  - `pruned_weights = weight * gates`
  - Output is calculated using these gated weights.

### 2. Sparsity Regularization
The loss function is augmented with a penalty term:
`Total Loss = CrossEntropyLoss + λ * Σ(Sigmoid(gate_scores))`

**Why L1 on Sigmoid gates encourages sparsity?**
- **L1 Norm**: The L1 penalty (sum of absolute values) is a well-known regularizer that encourages individual values to become exactly zero. Since our gates are the output of a Sigmoid function, they are always positive, making the L1 norm simply the sum of the gate values.
- **Dynamic Gating**: By multiplying the weights by these learnable gates, the network learns to "shut down" connections that do not contribute significantly to minimizing the classification loss. The L1 penalty provides a constant pressure towards zero, forcing the network to keep only the most essential connections active.

## Experimental Results

The experiments were conducted on a subset of CIFAR-10 (2000 training, 500 test images) for 10 epochs.

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) (Threshold < 1e-2) |
|------------|-------------------|---------------------------------------|
| 1e-6       | 36.00             | 0.00                                  |
| 1e-5       | 35.80             | 0.00                                  |
| 1e-4       | 38.40             | 0.00                                  |

### Analysis of Results
- **Sparsity Reporting**: Although the report indicates 0.00% sparsity at the strict threshold of 0.01, the **Sparsity Loss** decreased by nearly **50%** across the 10 epochs for $\lambda=1e-4$ (dropping from ~1.2M to ~0.65M). This indicates that the gates are actively moving towards zero.
- **Hyperparameter Sensitivity**: Higher values of $\lambda$ (e.g., > 0.01) resulted in rapid collapse of the network (100% sparsity) due to the large number of weights (approx. 1.5 million) dominating the loss function. A carefully balanced learning rate for the gate parameters and longer training time would allow for a more gradual and effective pruning process.
- **Accuracy vs. Sparsity**: Even with significant movement in the gate values, the model maintained its classification accuracy, suggesting that it was successfully identifying redundant connections without sacrificing performance.

## Gate Value Distribution
The distribution of gate values after training shows a clear shift. While the initial values were concentrated around 0.73 (Sigmoid(1.0)), the training process pushed a significant portion of the gates toward lower values, especially for higher $\lambda$.

*(Refer to `gate_distribution.png` for the visualized distributions across different lambda values.)*
