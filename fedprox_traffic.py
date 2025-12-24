"""
FedProx vs FedAvg: Encrypted Network Traffic Classification (Simulation)
========================================================================
A standalone simulation comparing FedAvg and FedProx on Non-IID synthetic network traffic data.
Uses a 1D-CNN for classification of packet sequence features.

Key Features:
- Synthetic Data: Encrypted traffic stats (Packet Size, Inter-arrival Time)
- Non-IID: Dirichlet distribution partitioning (alpha=0.5)
- Model: Lightweight 1D-CNN suitable for edge devices
- Algorithm: FedProx with custom proximal loss term
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import matplotlib.pyplot as plt
import random
import os

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
CONFIG = {
    'n_clients': 10,
    'n_rounds': 25,
    'local_epochs': 5,
    'batch_size': 32,
    'learning_rate': 0.01,
    'mu_fedprox': 0.1,  # Proximal term weight for FedProx
    'alpha_dirichlet': 0.5,  # Degree of Non-IID (lower = more heterogeneous)
    'n_samples': 5000,
    'seq_len': 50,
    'n_features': 2,    # Packet Size, Inter-arrival Time
    'n_classes': 4,     # Streaming, VoIP, Chat, File Transfer
    'seed': 42
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(CONFIG['seed'])

# =============================================================================
# 2. DATA SYNTHESIS & NON-IID PARTITIONING
# =============================================================================
def generate_synthetic_data(n_samples, seq_len, n_features, n_classes):
    """
    Generate synthetic encrypted traffic features.
    Class 0 (Streaming): High packet size, low inter-arrival (bursty)
    Class 1 (VoIP): Low packet size, constant low inter-arrival
    Class 2 (Chat): Low packet size, high/variable inter-arrival
    Class 3 (File Transfer): Max packet size, min inter-arrival
    """
    X = np.zeros((n_samples, n_features, seq_len)) # [N, C, L] for Conv1d
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        label = np.random.randint(0, n_classes)
        y[i] = label
        
        # Base patterns with noise
        if label == 0: # Streaming
            X[i, 0, :] = np.random.normal(0.8, 0.1, seq_len) # Size
            X[i, 1, :] = np.random.normal(0.2, 0.05, seq_len) # Time
        elif label == 1: # VoIP
            X[i, 0, :] = np.random.normal(0.3, 0.05, seq_len)
            X[i, 1, :] = np.random.normal(0.1, 0.01, seq_len)
        elif label == 2: # Chat
            X[i, 0, :] = np.random.normal(0.2, 0.1, seq_len)
            X[i, 1, :] = np.random.exponential(0.5, seq_len) # Bursty time
        elif label == 3: # File Transfer
            X[i, 0, :] = np.random.normal(0.9, 0.05, seq_len)
            X[i, 1, :] = np.random.exponential(0.05, seq_len)
            
    # Normalize
    X = (X - X.mean()) / (X.std() + 1e-8)
    return torch.FloatTensor(X), torch.LongTensor(y)

def partition_data_non_iid(X, y, n_clients, alpha):
    """
    Partition data using Dirichlet distribution for Non-IID setting.
    """
    try:
        n_classes = len(np.unique(y))
        client_data_indices = [[] for _ in range(n_clients)]
        
        # For each class, distribute samples to clients based on Dirichlet
        for c in range(n_classes):
            # Indices of current class
            idx_c = np.where(y == c)[0]
            np.random.shuffle(idx_c)
            
            # Draw proportions from Dirichlet
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            
            # Calculate split points
            proportions = np.array([p for p in proportions])
            proportions = proportions / proportions.sum()
            split_idx = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            
            # Split indices
            splits = np.split(idx_c, split_idx)
            
            for client_id, indices in enumerate(splits):
                client_data_indices[client_id].extend(indices)
                
        return client_data_indices
    except Exception as e:
        print(f"Partitioning error: {e}")
        # Fallback to random partition
        idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        return np.array_split(idxs, n_clients)

# =============================================================================
# 3. MODEL ARCHITECTURE
# =============================================================================
class TrafficCNN(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(TrafficCNN, self).__init__()
        # Lightweight 1D-CNN
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =============================================================================
# 4. FEDPROX ALGORITHM IMPLEMENTATION
# =============================================================================
def train_client_fedprox(model, global_model, train_loader, optimizer, device, mu, epochs):
    """
    Local training loop with FedProx Proximal Term.
    Loss = CE(y_pred, y) + (mu/2) * ||w - w_global||^2
    """
    model.train()
    global_model.eval()
    
    # Store global weights for comparison
    global_weights = list(global_model.parameters())
    
    epoch_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        batch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # 1. Standard Cross Entropy Loss
            ce_loss = criterion(outputs, y_batch)
            
            # 2. FedProx Proximal Term
            prox_term = 0.0
            if mu > 0:
                for w, w_global in zip(model.parameters(), global_weights):
                    prox_term += (w - w_global).norm(2)**2
            
            # Total Loss
            loss = ce_loss + (mu / 2) * prox_term
            
            loss.backward()
            optimizer.step()
            
            batch_loss += loss.item()
        epoch_loss += batch_loss / len(train_loader)
        
    return epoch_loss / epochs

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return 100 * correct / total

# =============================================================================
# 5. SIMULATION LOOP
# =============================================================================
def run_simulation(mode='FedAvg'):
    print(f"\n🚀 Starting Simulation: {mode}")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mu = CONFIG['mu_fedprox'] if mode == 'FedProx' else 0.0
    
    # 1. Prepare Data
    X, y = generate_synthetic_data(CONFIG['n_samples'], CONFIG['seq_len'], CONFIG['n_features'], CONFIG['n_classes'])
    
    # Split Test Set (Held-out)
    test_split = int(0.2 * CONFIG['n_samples'])
    X_test, y_test = X[:test_split], y[:test_split]
    X_train, y_train = X[test_split:], y[test_split:]
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Partition Training Data
    client_indices = partition_data_non_iid(X_train, y_train, CONFIG['n_clients'], CONFIG['alpha_dirichlet'])
    
    # Create Local Loaders
    client_loaders = []
    print("Client Class Distributions:")
    for i, indices in enumerate(client_indices):
        dataset = TensorDataset(X_train[indices], y_train[indices])
        client_loaders.append(DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True))
        
        # Print distribution for debugging
        labels = y_train[indices].numpy()
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"  Client {i}: {dist}")
    
    # 2. Initialize Models
    global_model = TrafficCNN(CONFIG['n_features'], CONFIG['n_classes']).to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(CONFIG['n_clients'])]
    
    # Trace Metrics
    accuracy_history = []
    loss_history = []
    
    # 3. Federated Training Loop
    for round_idx in range(CONFIG['n_rounds']):
        print(f"Round {round_idx+1}/{CONFIG['n_rounds']}...", end=' ')
        
        global_weights = global_model.state_dict()
        local_weights = []
        round_loss = 0
        
        # Train Clients
        for i in range(CONFIG['n_clients']):
            # Synchronize with global
            client_models[i].load_state_dict(global_weights)
            
            # Local Training
            optimizer = optim.SGD(client_models[i].parameters(), lr=CONFIG['learning_rate'], momentum=0.9)
            loss = train_client_fedprox(client_models[i], global_model, client_loaders[i], optimizer, device, mu, CONFIG['local_epochs'])
            
            local_weights.append(client_models[i].state_dict())
            round_loss += loss
        
        # Aggregation (FedAvg)
        avg_weights = copy.deepcopy(global_weights)
        for key in avg_weights.keys():
            weights_stack = torch.stack([w[key] for w in local_weights])
            if weights_stack.is_floating_point():
                avg_weights[key] = weights_stack.mean(0)
            else:
                # Handle LongTensor (e.g., BatchNorm num_batches_tracked)
                avg_weights[key] = weights_stack.float().mean(0).to(weights_stack.dtype)
            
        global_model.load_state_dict(avg_weights)
        
        # Evaluation
        acc = evaluate(global_model, test_loader, device)
        accuracy_history.append(acc)
        loss_history.append(round_loss / CONFIG['n_clients'])
        print(f"Test Acc: {acc:.2f}%, Avg Loss: {round_loss/CONFIG['n_clients']:.4f}")
        
    return accuracy_history, loss_history

# =============================================================================
# 6. EXECUTION & VISUALIZATION
# =============================================================================
if __name__ == "__main__":
    # Run FedAvg
    acc_avg, loss_avg = run_simulation(mode='FedAvg')
    
    # Run FedProx
    acc_prox, loss_prox = run_simulation(mode='FedProx')
    
    # Plot Results
    plt.figure(figsize=(14, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(acc_avg, 'o-', label=f'FedAvg (mu=0)', color='#e53e3e', linewidth=2)
    plt.plot(acc_prox, 's-', label=f'FedProx (mu={CONFIG["mu_fedprox"]})', color='#3182ce', linewidth=2)
    plt.title(f'Test Accuracy vs Rounds (Non-IID, alpha={CONFIG["alpha_dirichlet"]})')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(loss_avg, 'dashed', label='FedAvg Loss', color='#e53e3e')
    plt.plot(loss_prox, 'dashed', label='FedProx Loss', color='#3182ce')
    plt.title('Training Loss Convergence')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Average Local Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save absolute path
    save_path = os.path.join(os.getcwd(), 'fedprox_traffic_results.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ Simulation Complete! Results saved to '{save_path}'")
    # plt.show() # Disabled for headless environment
