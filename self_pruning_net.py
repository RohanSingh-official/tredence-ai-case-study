import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Part 1: The Prunable Linear Layer ---

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Learnable gate scores (same shape as weight)
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialization
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)
        # Initialize gate scores to a value that makes sigmoid(gate_scores) close to 1 initially
        nn.init.constant_(self.gate_scores, 1.0) 

    def forward(self, x):
        # Transform gate scores into gates between 0 and 1 using Sigmoid
        gates = torch.sigmoid(self.gate_scores)
        
        # Element-wise multiplication of weight and gates
        pruned_weights = self.weight * gates
        
        # Standard linear operation: y = x * W^T + b
        return F.linear(x, pruned_weights, self.bias)

    def get_sparsity(self, threshold=1e-2):
        """Returns the number of gates below the threshold and total number of gates."""
        gates = torch.sigmoid(self.gate_scores)
        pruned_count = (gates < threshold).sum().item()
        total_count = gates.numel()
        return pruned_count, total_count

    def get_gate_values(self):
        """Returns the sigmoid-transformed gate values as a flat tensor."""
        return torch.sigmoid(self.gate_scores).detach().cpu().numpy().flatten()

# --- Part 2: The Self-Pruning Network ---

class SelfPruningNet(nn.Module):
    def __init__(self, input_dim=3072, hidden_dims=[512, 256], output_dim=10):
        super(SelfPruningNet, self).__init__()
        
        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(PrunableLinear(last_dim, h_dim))
            layers.append(nn.ReLU())
            last_dim = h_dim
        
        layers.append(PrunableLinear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten image
        x = x.view(x.size(0), -1)
        return self.model(x)

    def get_sparsity_loss(self):
        """Calculates the sum of all gate values (L1 norm of gates)."""
        sparsity_loss = 0
        for layer in self.model:
            if isinstance(layer, PrunableLinear):
                sparsity_loss += torch.sigmoid(layer.gate_scores).sum()
        return sparsity_loss

    def report_sparsity(self, threshold=1e-2):
        """Calculates overall sparsity level across all PrunableLinear layers."""
        total_pruned = 0
        total_weights = 0
        for layer in self.model:
            if isinstance(layer, PrunableLinear):
                pruned, total = layer.get_sparsity(threshold)
                total_pruned += pruned
                total_weights += total
        
        sparsity_pct = (total_pruned / total_weights) * 100 if total_weights > 0 else 0
        return sparsity_pct

    def get_all_gate_values(self):
        """Collects all gate values from all layers."""
        all_gates = []
        for layer in self.model:
            if isinstance(layer, PrunableLinear):
                all_gates.extend(layer.get_gate_values())
        return np.array(all_gates)

# --- Part 3: Training and Evaluation ---

def train(model, device, train_loader, optimizer, criterion, lambda_reg, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        class_loss = criterion(output, target)
        sparsity_loss = model.get_sparsity_loss()
        
        total_loss = class_loss + lambda_reg * sparsity_loss
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {total_loss.item():.4f} (Class: {class_loss.item():.4f}, Sparsity: {sparsity_loss.item():.4f})")

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def run_experiment(lambda_val, epochs=10, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running Experiment: Lambda = {lambda_val} ---")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Use a small subset for speed in this environment
    train_set = torch.utils.data.Subset(train_set, range(2000))
    test_set = torch.utils.data.Subset(test_set, range(500))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    model = SelfPruningNet().to(device)
    
    # Use different learning rates for weights and gate_scores
    gate_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'gate_scores' in name:
            gate_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = optim.Adam([
        {'params': other_params, 'lr': 1e-3},
        {'params': gate_params, 'lr': 1e-2} # Balanced LR for gates
    ])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, lambda_val, epoch)
        acc = evaluate(model, device, test_loader)
        sparsity = model.report_sparsity()
        print(f"Epoch {epoch}: Accuracy: {acc:.2f}%, Sparsity: {sparsity:.2f}%")
        
    final_acc = evaluate(model, device, test_loader)
    final_sparsity = model.report_sparsity()
    gate_values = model.get_all_gate_values()
    
    return final_acc, final_sparsity, gate_values

if __name__ == "__main__":
    lambdas = [1e-6, 1e-5, 1e-4]
    results = []
    
    # Ensure data directory exists
    if not os.path.exists('./data'):
        os.makedirs('./data')
        
    for l in lambdas:
        acc, sparsity, gates = run_experiment(l, epochs=10) # 10 epochs for better results
        results.append({
            'lambda': l,
            'accuracy': acc,
            'sparsity': sparsity,
            'gates': gates
        })
    
    # --- Generate Report Data ---
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"{'Lambda':<10} | {'Accuracy (%)':<15} | {'Sparsity (%)':<15}")
    for res in results:
        print(f"{res['lambda']:<10} | {res['accuracy']:<15.2f} | {res['sparsity']:<15.2f}")
    
    # Plot distribution for all models to show the effect of lambda
    plt.figure(figsize=(15, 5))
    for i, res in enumerate(results):
        plt.subplot(1, 3, i+1)
        plt.hist(res['gates'], bins=50, color='skyblue', edgecolor='black')
        plt.title(f"Lambda = {res['lambda']}\nSparsity = {res['sparsity']:.2f}%")
        plt.xlabel("Gate Value")
        plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig('gate_distribution.png')
    print("\nSaved combined gate distribution plot to 'gate_distribution.png'")
