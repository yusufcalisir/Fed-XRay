import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import matplotlib.pyplot as plt
import os

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# ==================================================================================================
# 1. TRAFFIC MODEL (Universal Architecture)
# ==================================================================================================
class TrafficModel(nn.Module):
    def __init__(self, input_channels=2, num_classes=4, sequence_length=50):
        super(TrafficModel, self).__init__()
        
        # Backbone (1D-CNN)
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Calculate flattened size
        self.flatten_dim = 64 * (sequence_length // 2 // 2)
        
        # Representation Head (z)
        self.representation_layer = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64) # z dimension
        )
        
        # Classification Head (logits)
        self.classifier_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        z = self.representation_layer(x)
        logits = self.classifier_layer(z)
        return z, logits

# ==================================================================================================
# 2. DATA GENERATION (Severe Non-IID)
# ==================================================================================================
def generate_data(n_clients=10, n_samples=200, alpha=0.1):
    """
    Generates synthetic traffic data and partitions it with Dirichlet(alpha=0.1).
    Alpha=0.1 creates extreme heterogeneity.
    """
    input_channels = 2
    seq_length = 50
    num_classes = 4
    
    # Generate pool
    X_pool = []
    y_pool = []
    
    for c in range(num_classes):
        # Generate 10x required samples to ensure enough per class
        n_class_samples = n_clients * n_samples 
        t = np.linspace(0, 5, seq_length)
        # Unique frequency per class
        wave = np.sin((1.0 + c*2.5) * t)
        wave = np.tile(wave, (n_class_samples, 1))
        
        # Add noise and second channel
        feat1 = wave + np.random.normal(0, 0.4, wave.shape)
        feat2 = np.random.normal(0, 0.8, wave.shape) # Noise channel
        
        X_c = np.stack([feat1, feat2], axis=1) # (N, 2, 50)
        y_c = np.full(n_class_samples, c)
        
        X_pool.append(X_c)
        y_pool.append(y_c)
        
    X_pool = np.concatenate(X_pool)
    y_pool = np.concatenate(y_pool)
    
    # Dirichlet Partitioning
    min_size = 0
    K = num_classes
    N = len(y_pool)
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(K):
            idx_k = np.where(y_pool == k)[0]
            np.random.shuffle(idx_k)
            # Dirichlet split
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    client_datasets = []
    for j in range(n_clients):
        np.random.shuffle(idx_batch[j])
        data = X_pool[idx_batch[j]][:n_samples] # Limit to n_samples
        targets = y_pool[idx_batch[j]][:n_samples]
        
        tensor_x = torch.FloatTensor(data)
        tensor_y = torch.LongTensor(targets)
        client_datasets.append(TensorDataset(tensor_x, tensor_y))
        
    return client_datasets, X_pool, y_pool

# ==================================================================================================
# 3. LOCAL TRAINER (The Core Logic)
# ==================================================================================================
class LocalTrainer:
    def __init__(self, algorithm, device, mu=1.0, temperature=0.5, lr=0.01):
        self.algorithm = algorithm
        self.device = device
        self.mu = mu
        self.temperature = temperature
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def train(self, local_model, global_model, prev_model, train_loader, c_local=None, c_global=None):
        local_model.train()
        optimizer = optim.SGD(local_model.parameters(), lr=self.lr)
        
        epoch_loss = 0
        
        # Frozen models for reference
        if global_model:
            global_model.eval()
            for p in global_model.parameters(): p.requires_grad = False
            
        if prev_model:
            prev_model.eval()
            for p in prev_model.parameters(): p.requires_grad = False
            
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            
            # Forward
            z, logits = local_model(images)
            loss_sup = self.criterion(logits, labels)
            loss = loss_sup
            
            # --- ALGORITHM SPECIFIC LOGIC ---
            
            if self.algorithm == 'FedProx':
                # Proximal Term: mu/2 * ||w - w_t||^2
                proximal_term = 0.0
                for w, w_t in zip(local_model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss += (self.mu / 2) * proximal_term

            elif self.algorithm == 'MOON':
                if global_model and prev_model:
                    with torch.no_grad():
                        z_glob, _ = global_model(images)
                        z_prev, _ = prev_model(images)
                    
                    # Contrastive Loss
                    cos_sim = nn.CosineSimilarity(dim=-1)
                    logits_pos = cos_sim(z, z_glob) / self.temperature
                    logits_neg = cos_sim(z, z_prev) / self.temperature
                    
                    # -log( exp(pos) / (exp(pos) + exp(neg)) )
                    # Stability trick: -log_softmax
                    # But implementing direct user logic for clarity:
                    nominator = torch.exp(logits_pos)
                    denominator = nominator + torch.exp(logits_neg)
                    loss_con = -torch.log(nominator / denominator).mean()
                    
                    loss += (self.mu * loss_con)
            
            elif self.algorithm == 'SCAFFOLD':
                # SCAFFOLD handles updates via optimizer step modification, 
                # but we need to compute gradients first.
                # Standard loss is just CrossEntropy here.
                pass 
                
            # Backward
            loss.backward()
            
            # --- UPDATE RULE ---
            if self.algorithm == 'SCAFFOLD':
                # w_new = w - lr * (g - c_i + c)
                # g is already in p.grad
                for param, c_l, c_g in zip(local_model.parameters(), c_local, c_global):
                    if param.grad is not None:
                        # Modified gradient: g_mod = g - c_l + c_g
                        # We apply it by manually updating the parameter
                        # or modifying grad. Let's manual update for clarity logic.
                        
                        # param.data = param.data - lr * (param.grad.data - c_l + c_g)
                        # To support momentum, we usually modify .grad:
                        param.grad.data += (c_g - c_l)
                
                optimizer.step()
            else:
                # Standard SGD for others
                optimizer.step()
                
            epoch_loss += loss.item()
            
        return epoch_loss / len(train_loader)

# ==================================================================================================
# 4. SIMULATION LOOP
# ==================================================================================================
def run_benchmark():
    # Config
    ALGORITHMS = ['FedAvg', 'FedProx', 'MOON', 'SCAFFOLD']
    ROUNDS = 50
    CLIENTS = 10
    LOCAL_EPOCHS = 3 # Increased slightly to capture drift
    LR = 0.01
    SAMPLE_RATE = 1.0 # 10/10 clients per round
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Benchmark on {device}")
    
    # Generate Data
    print("Generating Non-IID Data (alpha=0.1)...")
    client_datasets, X_test, y_test = generate_data(n_clients=CLIENTS)
    
    # Benchmarking Store
    results = {alg: {'acc': [], 'loss': []} for alg in ALGORITHMS}
    
    for alg in ALGORITHMS:
        print(f"\n>>> Starting {alg} Simulation")
        
        # Init Global Model
        global_model = TrafficModel().to(device)
        global_weights = global_model.state_dict()
        
        # Init Local Trainer
        trainer = LocalTrainer(alg, device, mu=1.0, temperature=0.5, lr=LR)
        
        # Algorithm constraints
        # MOON: Needs prev_models
        client_prev_models = [None] * CLIENTS if alg == 'MOON' else None
        
        # SCAFFOLD: Needs control variates
        # c_global initialized to 0
        # c_local initialized to 0
        if alg == 'SCAFFOLD':
            c_global = [torch.zeros_like(p) for p in global_model.parameters()]
            c_locals = [[torch.zeros_like(p) for p in global_model.parameters()] for _ in range(CLIENTS)]
        
        for r in range(ROUNDS):
            local_weights = []
            local_losses = []
            
            # New c_global delta tracker for SCAFFOLD
            total_delta_c = None 
            
            # Select Clients (All 10)
            client_indices = list(range(CLIENTS))
            
            # Temporary storage to update state AFTER the round
            next_prev_models = [None] * CLIENTS
            
            for cid in client_indices:
                # Setup Local Model
                local_model = TrafficModel().to(device)
                local_model.load_state_dict(copy.deepcopy(global_weights))
                
                loader = DataLoader(client_datasets[cid], batch_size=32, shuffle=True)
                
                # SCAFFOLD Params
                c_l = c_locals[cid] if alg == 'SCAFFOLD' else None
                c_g = c_global if alg == 'SCAFFOLD' else None
                
                # MOON Params
                prev_m = client_prev_models[cid].to(device) if (alg == 'MOON' and client_prev_models[cid]) else None
                
                # Train
                l_loss = 0
                for _ in range(LOCAL_EPOCHS):
                    l_loss += trainer.train(
                        local_model, global_model, prev_m, loader, 
                        c_local=c_l, c_global=c_g
                    )
                local_losses.append(l_loss / LOCAL_EPOCHS)
                
                # Store new weights
                new_w = copy.deepcopy(local_model.state_dict())
                local_weights.append(new_w)
                
                # --- ALGORITHM SPECIFIC STATE UPDATES ---
                if alg == 'MOON':
                    # Save current trained model as prev for next round
                    next_prev_models[cid] = copy.deepcopy(local_model).cpu()
                
                elif alg == 'SCAFFOLD':
                    # Update c_local
                    # c_l_new = c_l - c_g + (w_g - w_l) / (lr * epochs)
                    # For simplicity assuming constant steps K = len(loader) * epochs
                    K = len(loader) * LOCAL_EPOCHS
                    
                    # Calculate drift and update c_local
                    # We need access to model params directly
                    new_c_l = []
                    delta_c_l = [] # c_l_new - c_l
                    
                    with torch.no_grad():
                        for p_g, p_l, cl, cg in zip(global_model.parameters(), local_model.parameters(), c_l, c_g):
                            # (w_global - w_local) / (lr * steps)
                            drift = (p_g - p_l) / (LR * K)
                            # c_new = c_local - c_global + drift
                            cl_new = cl - cg + drift
                            new_c_l.append(cl_new)
                            delta_c_l.append(cl_new - cl)
                            
                    c_locals[cid] = new_c_l # Update client state
                    
                    # Accumulate delta for global update
                    if total_delta_c is None:
                        total_delta_c = [d.clone() for d in delta_c_l]
                    else:
                        for i, d in enumerate(delta_c_l):
                            total_delta_c[i] += d
            
            # --- SERVER AGGREGATION ---
            # FedAvg for weights (Common to all)
            avg_w = copy.deepcopy(global_weights)
            for key in avg_w.keys():
                avg_w[key] = torch.stack([w[key].float() for w in local_weights], 0).mean(0)
            global_model.load_state_dict(avg_w)
            global_weights = avg_w
            
            # Post-Aggregation Checks
            if alg == 'MOON':
                client_prev_models = next_prev_models
                
            elif alg == 'SCAFFOLD':
                # Update c_global
                # c_g_new = c_g + mean(delta_c_local)
                for i, d in enumerate(total_delta_c):
                    c_global[i] += (d / CLIENTS)
            
            # Evaluation
            global_model.eval()
            test_x = torch.FloatTensor(X_test).to(device)
            test_y = torch.LongTensor(y_test).to(device)
            with torch.no_grad():
                _, logits = global_model(test_x)
                _, preds = torch.max(logits, 1)
                acc = (preds == test_y).float().mean().item() * 100
                
            results[alg]['acc'].append(acc)
            results[alg]['loss'].append(sum(local_losses) / len(local_losses))
            
            if r % 10 == 0:
                print(f"Round {r}: Acc {acc:.2f}%")
                
    # ==================================================================================================
    # 5. VISUALIZATION
    # ==================================================================================================
    print("\nSimulations Complete. Plotting Results...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'FedAvg': 'gray', 'FedProx': 'blue', 'MOON': 'purple', 'SCAFFOLD': 'green'}
    styles = {'FedAvg': '--', 'FedProx': '-', 'MOON': '-', 'SCAFFOLD': '-'}
    
    for alg in ALGORITHMS:
        ax1.plot(results[alg]['acc'], label=alg, color=colors[alg], linestyle=styles[alg], linewidth=2)
        ax2.plot(results[alg]['loss'], label=alg, color=colors[alg], linestyle=styles[alg], linewidth=2)
        
    ax1.set_title("Global Accuracy vs. Rounds (Non-IID $\\alpha=0.1$)", fontsize=14)
    ax1.set_xlabel("Rounds")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    
    ax2.set_title("Training Loss Convergence", fontsize=14)
    ax2.set_xlabel("Rounds")
    ax2.set_ylabel("Loss")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("Benchmark Saved: benchmark_results.png")

if __name__ == "__main__":
    run_benchmark()
