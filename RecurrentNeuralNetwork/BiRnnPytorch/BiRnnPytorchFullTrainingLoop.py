import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# TASK: Predict the MIDDLE value of a sequence using BiRNN
# ─────────────────────────────────────────────────────────────────────────────

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN     = 20
MIDDLE_IDX  = 10
INPUT_SIZE  = 1
HIDDEN_SIZE = 32
OUTPUT_SIZE = 1
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 0.001

# ── 1. Generate Data ──────────────────────────────────────────────────────────
def make_sine_data(n_points=2000, noise=0.1):
    t = np.linspace(0, 16 * np.pi, n_points)
    y = np.sin(t) + noise * np.random.randn(n_points)
    return y.astype(np.float32)

def make_sequences(data, seq_len, target_idx):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        Y.append(data[i + target_idx])
    
    X = torch.tensor(X).unsqueeze(-1)
    Y = torch.tensor(Y).unsqueeze(-1)
    return X, Y

raw = make_sine_data()
split = int(len(raw) * 0.8)

X_train, Y_train = make_sequences(raw[:split], SEQ_LEN, MIDDLE_IDX)
X_val,   Y_val   = make_sequences(raw[split:],  SEQ_LEN, MIDDLE_IDX)

# ── 2. DataLoader ─────────────────────────────────────────────────────────────
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, Y_train), 
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_val, Y_val), 
    batch_size=BATCH_SIZE
)

# ── 3. BiRNN From Scratch ─────────────────────────────────────────────────────
class BiRNNFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # ── FORWARD WEIGHTS ─────────────────────────────────────────────────
        self.f_Wxh = nn.Linear(input_size, hidden_size, bias=True)
        self.f_Whh = nn.Linear(hidden_size, hidden_size, bias=False)

        # ── BACKWARD WEIGHTS ────────────────────────────────────────────────
        self.b_Wxh = nn.Linear(input_size, hidden_size, bias=True)
        self.b_Whh = nn.Linear(hidden_size, hidden_size, bias=False)

        # ── OUTPUT WEIGHTS ──────────────────────────────────────────────────
        self.Why = nn.Linear(hidden_size * 2, output_size, bias=True)

        # ── Initialize Weights ──────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """CORRECTED: Check bias FIRST, then weights"""
        for name, param in self.named_parameters():
            if 'bias' in name:              # ← BIAS FIRST! (1D tensor)
                nn.init.constant_(param, 0)
            elif 'Whh' in name:             # ← Recurrent weights (2D tensor)
                nn.init.orthogonal_(param)
            elif 'Wxh' in name:             # ← Input weights (2D tensor)
                nn.init.xavier_uniform_(param)

    def forward(self, x_seq, states=None):
        """
        Input x_seq shape: (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, _ = x_seq.shape
        device = x_seq.device

        # 1. Initialize Hidden States
        # Shape: (batch_size, hidden_size)
        if states is None:
            h_fwd = torch.zeros(batch_size, self.hidden_size, device=device)
            h_bwd = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_fwd, h_bwd = states

        # ── 2. Forward Pass (Left → Right) ──────────────────────────────────
        for t in range(seq_len):
            # x_t shape: (batch_size, input_size)
            x_t = x_seq[:, t, :]
            
            # h_fwd shape: (batch_size, hidden_size)
            h_fwd = torch.tanh(self.f_Wxh(x_t) + self.f_Whh(h_fwd))

        # ── 3. Backward Pass (Right → Left) ─────────────────────────────────
        # x_seq_rev shape: (batch_size, seq_len, input_size)
        x_seq_rev = torch.flip(x_seq, dims=[1])
        
        for t in range(seq_len):
            # x_t shape: (batch_size, input_size)
            x_t = x_seq_rev[:, t, :]
            
            # h_bwd shape: (batch_size, hidden_size)
            h_bwd = torch.tanh(self.b_Wxh(x_t) + self.b_Whh(h_bwd))

        # ── 4. Combine Both Directions ──────────────────────────────────────
        # h_combined shape: (batch_size, hidden_size * 2)
        h_combined = torch.cat((h_fwd, h_bwd), dim=1)

        # ── 5. Output ───────────────────────────────────────────────────────
        # out shape: (batch_size, output_size)
        out = self.Why(h_combined)
        
        return out, (h_fwd, h_bwd)

# ── 4. Setup ──────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiRNNFromScratch(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"\nUsing device: {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}\n")

# ── 5. Training Loop ──────────────────────────────────────────────────────────
train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    batch_losses = []

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        preds, _ = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)

    # Validation
    model.eval()
    batch_val_losses = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds, _ = model(x_batch)
            loss = criterion(preds, y_batch)
            batch_val_losses.append(loss.item())

    val_loss = np.mean(batch_val_losses)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

# ── 6. Plot Results ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Loss Plot
axes[0].plot(train_losses, label='Train', color='#7c6fff')
axes[0].plot(val_losses, label='Val', color='#ff6b9d')
axes[0].set_title('Loss Over Epochs')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Prediction Plot
model.eval()
with torch.no_grad():
    x_sample = X_val[:100].to(device)
    y_actual = Y_val[:100].numpy().squeeze()
    preds, _ = model(x_sample)
    y_pred = preds.cpu().numpy().squeeze()

axes[1].plot(y_actual, label='Actual', color='#4ecca3')
axes[1].plot(y_pred, label='Predicted', color='#f5a623', linestyle='--')
axes[1].set_title('Actual vs Predicted')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('birnn_shapes.png', dpi=120)
plt.show()
