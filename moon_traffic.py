import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==================================================================================================
# 1. MODEL ARCHITECTURE (Split Head)
# ==================================================================================================
class SimpleCNN_MOON(nn.Module):
    def __init__(self, input_channels=2, num_classes=4, sequence_length=50):
        super(SimpleCNN_MOON, self).__init__()
        
        # Encoder (Base)
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        
        # Calculate flattened size
        self.flatten_dim = 32 * (sequence_length // 2 // 2) 
        
        # Representation Head (Returns z)
        # We use a projection layer to map flattened features to a lower dimensional z
        self.fc_projection = nn.Linear(self.flatten_dim, 128)
        self.fc_z = nn.Linear(128, 64) # Output z dimension = 64
        
        # Classification Head (Takes z and classification)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Base Encoder
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        
        # Projection to Representation (z)
        x = self.relu(self.fc_projection(x))
        z = self.fc_z(x)
        
        # Classification
        logits = self.classifier(z)
        
        return z, logits

# ==================================================================================================
# 2. MOON LOSS LOGIC
# ==================================================================================================
def train_step_moon(
    model: nn.Module, 
    global_model: nn.Module, 
    prev_model: nn.Module, 
    optimizer: optim.Optimizer, 
    images: torch.Tensor, 
    labels: torch.Tensor, 
    device: torch.device,
    temperature: float = 0.5,
    mu: float = 1.0
) -> Tuple[float, float, float]:
    
    model.train()
    if global_model: global_model.eval()
    if prev_model: prev_model.eval()
    
    # 1. Forward Pass Current Model
    z, logits = model(images)
    
    # 2. Supervised Loss (Cross Entropy)
    loss_sup = F.cross_entropy(logits, labels)
    
    # 3. MOON Contrastive Loss
    loss_con = torch.tensor(0.0).to(device)
    
    if global_model is not None and prev_model is not None:
        with torch.no_grad():
            z_glob, _ = global_model(images)
            z_prev, _ = prev_model(images)
        
        # Cosine Similarity Function
        cos = nn.CosineSimilarity(dim=-1)
        
        # Positive Pair: Current z <-> Global z
        sim_pos = cos(z, z_glob.detach())
        logits_pos = sim_pos / temperature
        
        # Negative Pair: Current z <-> Previous z
        sim_neg = cos(z, z_prev.detach())
        logits_neg = sim_neg / temperature
        
        # Numerator: exp(pos)
        # Denominator: exp(pos) + exp(neg)
        # We use LogSumExp trick or simple construction for stability?
        # User formula: -log( exp(pos) / (exp(pos) + exp(neg)) )
        
        # Create logits for CrossEntropy implementation of Contrastive Loss
        # We act as if we have a classification task with 1 positive class (index 0)
        # logits matrix: [batch_size, 2] -> Col 0: Pos, Col 1: Neg
        
        # Explicit implementation of user formula for clarity
        numerator = torch.exp(logits_pos)
        denominator = numerator + torch.exp(logits_neg)
        
        # loss per sample
        loss_con_samples = -torch.log(numerator / denominator)
        loss_con = loss_con_samples.mean()

    # 4. Total Loss
    loss_total = loss_sup + (mu * loss_con)
    
    # Backprop
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    
    return loss_total.item(), loss_sup.item(), loss_con.item()

# ==================================================================================================
# 3. DATA GENERATION (Synthetic Non-IID)
# ==================================================================================================
def generate_synthetic_data(n_clients=10, n_samples_per_client=200, alpha=0.5):
    """
    Generates synthetic sequence data (Batch, 50, 2) and distributes it non-IID via Dirichlet.
    """
    input_channels = 2
    seq_length = 50
    input_size = input_channels * seq_length
    num_classes = 4
    
    X_all = []
    y_all = []
    
    # Create distinct patterns for classes
    for c in range(num_classes):
        # Generate raw data with some class-specific signal
        n_class_samples = n_clients * n_samples_per_client # sufficient pool
        
        # Signal: Sine waves with different frequencies
        t = np.linspace(0, 10, seq_length)
        freq = 1.0 + (c * 1.5)
        
        wave = np.sin(freq * t)
        wave = np.tile(wave, (n_class_samples, 1))
        
        # Feature 1: Signal + Noise
        feat1 = wave + np.random.normal(0, 0.5, wave.shape)
        # Feature 2: Noise or different signal
        feat2 = np.random.normal(0, 1.0, wave.shape)
        
        # Stack features [Samples, Channels, Seq_Len]
        # X shape: (N, 2, 50)
        X_c = np.stack([feat1, feat2], axis=1)
        y_c = np.full(n_class_samples, c)
        
        X_all.append(X_c)
        y_all.append(y_c)
        
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    # Dirichlet Partitioning
    min_size = 0
    K = num_classes
    N = len(y_all)
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(K):
            idx_k = np.where(y_all == k)[0]
            np.random.shuffle(idx_k)
            # Standard Dirichlet split
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    client_datasets = []
    for j in range(n_clients):
        np.random.shuffle(idx_batch[j])
        client_data = X_all[idx_batch[j]]
        client_targets = y_all[idx_batch[j]]
        
        # Convert to Tensor
        tensor_x = torch.FloatTensor(client_data)
        tensor_y = torch.LongTensor(client_targets)
        
        client_datasets.append(TensorDataset(tensor_x, tensor_y))
        
    return client_datasets

# ==================================================================================================
# 4. FEDERATED SIMULATION LOOP
# ==================================================================================================
def main():
    # Config
    N_CLIENTS = 10
    N_ROUNDS = 20
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 32
    LR = 0.01
    MU = 1.0         # MOON Weight
    TEMP = 0.5       # Temperature
    ALPHA_DIRICHLET = 0.5 # Non-IID skew
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    # 1. Data
    print("Generating synthetic Non-IID data...")
    client_datasets = generate_synthetic_data(N_CLIENTS, alpha=ALPHA_DIRICHLET)
    
    # 2. Initialization
    global_model = SimpleCNN_MOON().to(device)
    
    # Track models for MOON (prev_models per client)
    # At start, prev_model is same as global or None
    client_prev_models = [None for _ in range(N_CLIENTS)]
    
    # History
    history = {'loss': [], 'acc': [], 'moon_loss': []}
    
    print("-" * 60)
    print(f"{'Round':<6} | {'Global Acc':<12} | {'Train Loss':<12} | {'MOON Loss':<12}")
    print("-" * 60)
    
    for round_num in range(1, N_ROUNDS + 1):
        global_weights = global_model.state_dict()
        local_weights = []
        local_losses = []
        local_moon_losses = []
        
        # Select all clients or subset
        selected_clients = range(N_CLIENTS) 
        
        current_client_models_update = [None] * N_CLIENTS
        
        for client_id in selected_clients:
            # Local Training
            local_model = SimpleCNN_MOON().to(device)
            local_model.load_state_dict(copy.deepcopy(global_weights))
            
            optimizer = optim.SGD(local_model.parameters(), lr=LR, momentum=0.9)
            loader = DataLoader(client_datasets[client_id], batch_size=BATCH_SIZE, shuffle=True)
            
            # Get previous model for this client (for MOON loss)
            # If round 1, prev_model might be None, handled in loss function
            prev_model = client_prev_models[client_id]
            if prev_model: prev_model.to(device)
            
            # Global model for this round (frozen)
            global_model_frozen = copy.deepcopy(global_model)
            global_model_frozen.eval()
            for p in global_model_frozen.parameters(): p.requires_grad = False
            
            epoch_loss = 0
            epoch_moon = 0
            
            for epoch in range(LOCAL_EPOCHS):
                batch_loss = 0
                batch_moon = 0
                count = 0
                for imgs, lbls in loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    
                    l_total, l_sup, l_con = train_step_moon(
                        model=local_model,
                        global_model=global_model_frozen, # z_global source
                        prev_model=prev_model,            # z_prev source
                        optimizer=optimizer,
                        images=imgs,
                        labels=lbls,
                        device=device,
                        temperature=TEMP,
                        mu=MU
                    )
                    
                    batch_loss += l_total
                    batch_moon += l_con
                    count += 1
                
                epoch_loss += batch_loss / count
                epoch_moon += batch_moon / count
            
            # Store update
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_losses.append(epoch_loss)
            local_moon_losses.append(epoch_moon)
            
            # Save current model to be next round's prev_model
            # Important: We want the model AFTER training this round
            # to be the "prev_model" for the NEXT round
            current_client_models_update[client_id] = copy.deepcopy(local_model)
            if prev_model: prev_model.to('cpu') # Free GPU
            
        # FedAvg Aggregation
        avg_weights = copy.deepcopy(global_weights)
        for key in avg_weights.keys():
            avg_weights[key] = torch.stack([w[key].float() for w in local_weights], 0).mean(0)
            
        global_model.load_state_dict(avg_weights)
        
        # Update client prev models for next round
        for cid in selected_clients:
            client_prev_models[cid] = current_client_models_update[cid]
        
        # Evaluate Global Model (Simple validation on Client 0 for speed)
        global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # Validate on all data combined (or subset)
            limit = 0
            for ds in client_datasets:
                limit += 1
                if limit > 2: break # Check first 2 clients
                loader = DataLoader(ds, batch_size=32)
                for imgs, lbls in loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    _, outputs = global_model(imgs) # Ignore z
                    _, predicted = torch.max(outputs.data, 1)
                    total += lbls.size(0)
                    correct += (predicted == lbls).sum().item()
        
        acc = 100 * correct / total
        avg_loss = sum(local_losses) / len(local_losses)
        avg_moon = sum(local_moon_losses) / len(local_moon_losses)
        
        history['acc'].append(acc)
        history['loss'].append(avg_loss)
        
        print(f"{round_num:<6} | {acc:<11.2f}% | {avg_loss:<12.4f} | {avg_moon:<12.4f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['acc'])
    plt.title('Global Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Acc %')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Total Loss')
    plt.title('Training Loss')
    plt.xlabel('Round')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('moon_results.png')
    print("Simulation Complete. Results saved to moon_results.png")

if __name__ == "__main__":
    main()
