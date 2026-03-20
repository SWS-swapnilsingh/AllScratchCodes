import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# TASK: Predict the MIDDLE value of a sequence using BiLSTM
#       Why? BiLSTM sees Past + Future with better memory control.
# ─────────────────────────────────────────────────────────────────────────────

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN     = 20            # Sequence length
MIDDLE_IDX  = 10            # Target index
INPUT_SIZE  = 1             # Input features
HIDDEN_SIZE = 32            # Hidden units
OUTPUT_SIZE = 1             # Output features
BATCH_SIZE  = 32            # Batch size
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
    
    # X shape: (number_of_samples, seq_len, input_size)
    # Y shape: (number_of_samples, 1)
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

# ── 3. BiLSTM From Scratch ────────────────────────────────────────────────────
class BiLSTMFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # ── FORWARD LSTM WEIGHTS (4 gates) ──────────────────────────────────
        # Forget gate: decides what to erase from cell state
        self.f_Wxf = nn.Linear(input_size, hidden_size, bias=True)
        self.f_Whf = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Input gate: decides what new info to write to cell
        self.f_Wxi = nn.Linear(input_size, hidden_size, bias=True)
        self.f_Whi = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Candidate: new content to potentially write
        self.f_Wxg = nn.Linear(input_size, hidden_size, bias=True)
        self.f_Whg = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output gate: decides what part of cell to expose as hidden
        self.f_Wxo = nn.Linear(input_size, hidden_size, bias=True)
        self.f_Who = nn.Linear(hidden_size, hidden_size, bias=False)

        # ── BACKWARD LSTM WEIGHTS (4 gates, separate from forward) ──────────
        # Forget gate
        self.b_Wxf = nn.Linear(input_size, hidden_size, bias=True)
        self.b_Whf = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Input gate
        self.b_Wxi = nn.Linear(input_size, hidden_size, bias=True)
        self.b_Whi = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Candidate
        self.b_Wxg = nn.Linear(input_size, hidden_size, bias=True)
        self.b_Whg = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output gate
        self.b_Wxo = nn.Linear(input_size, hidden_size, bias=True)
        self.b_Who = nn.Linear(hidden_size, hidden_size, bias=False)

        # ── OUTPUT WEIGHTS ──────────────────────────────────────────────────
        # Input is (hidden_size * 2) because we join forward + backward
        # Weight shape: (hidden_size * 2, output_size)
        self.Why = nn.Linear(hidden_size * 2, output_size, bias=True)

        # ── Initialize Weights ──────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        CORRECTED: Check bias FIRST, then weights.
        Bias is 1D, weights are 2D. Xavier/Orthogonal need 2D.
        """
        for name, param in self.named_parameters():
            if 'bias' in name:              # ← BIAS FIRST! (1D tensor)
                nn.init.constant_(param, 0)
            elif 'Wh' in name:              # ← Recurrent weights (2D tensor)
                nn.init.orthogonal_(param)
            elif 'Wx' in name:              # ← Input weights (2D tensor)
                nn.init.xavier_uniform_(param)
        
        # Special trick: Initialize forget gate bias to 1.0
        # This helps the model learn to "remember" initially
        for name, param in self.named_parameters():
            if 'f_Wxf.bias' in name or 'b_Wxf.bias' in name:
                nn.init.constant_(param, 1.0)

    def forward(self, x_seq, states=None):
        """
        Input x_seq shape: (batch_size, seq_len, input_size)
        
        NOTE ON STATES:
        If states=None, we create new zero states. This is correct for 
        independent sequences (like our sine wave batches).
        Learning happens in WEIGHTS, not hidden states.
        """
        batch_size, seq_len, _ = x_seq.shape
        device = x_seq.device

        # 1. Initialize Hidden States and Cell States
        # h shape: (batch_size, hidden_size)
        # c shape: (batch_size, hidden_size)
        if states is None:
            h_fwd = torch.zeros(batch_size, self.hidden_size, device=device)
            c_fwd = torch.zeros(batch_size, self.hidden_size, device=device)
            h_bwd = torch.zeros(batch_size, self.hidden_size, device=device)
            c_bwd = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_fwd, c_fwd, h_bwd, c_bwd = states

        # ── 2. Forward Pass (Left → Right) ──────────────────────────────────
        for t in range(seq_len):
            # x_t shape: (batch_size, input_size)
            x_t = x_seq[:, t, :]
            
            # Forget gate: how much of OLD cell to keep
            # f_t shape: (batch_size, hidden_size)
            f_t = torch.sigmoid(self.f_Wxf(x_t) + self.f_Whf(h_fwd))
            
            # Input gate: which positions in cell to write to
            # i_t shape: (batch_size, hidden_size)
            i_t = torch.sigmoid(self.f_Wxi(x_t) + self.f_Whi(h_fwd))
            
            # Candidate: new content that COULD be written
            # g_t shape: (batch_size, hidden_size)
            g_t = torch.tanh(self.f_Wxg(x_t) + self.f_Whg(h_fwd))
            
            # Update cell state: forget old, write new
            # c_fwd shape: (batch_size, hidden_size)
            c_fwd = f_t * c_fwd + i_t * g_t
            
            # Output gate: which part of cell to expose as hidden
            # o_t shape: (batch_size, hidden_size)
            o_t = torch.sigmoid(self.f_Wxo(x_t) + self.f_Who(h_fwd))
            
            # New hidden state: filtered view of the cell
            # h_fwd shape: (batch_size, hidden_size)
            h_fwd = o_t * torch.tanh(c_fwd)

        # ── 3. Backward Pass (Right → Left) ─────────────────────────────────
        # Flip sequence on dimension 1 (seq_len)
        # x_seq_rev shape: (batch_size, seq_len, input_size)
        x_seq_rev = torch.flip(x_seq, dims=[1])
        
        for t in range(seq_len):
            # x_t shape: (batch_size, input_size)
            x_t = x_seq_rev[:, t, :]
            
            # Forget gate
            # f_t shape: (batch_size, hidden_size)
            f_t = torch.sigmoid(self.b_Wxf(x_t) + self.b_Whf(h_bwd))
            
            # Input gate
            # i_t shape: (batch_size, hidden_size)
            i_t = torch.sigmoid(self.b_Wxi(x_t) + self.b_Whi(h_bwd))
            
            # Candidate
            # g_t shape: (batch_size, hidden_size)
            g_t = torch.tanh(self.b_Wxg(x_t) + self.b_Whg(h_bwd))
            
            # Update cell state
            # c_bwd shape: (batch_size, hidden_size)
            c_bwd = f_t * c_bwd + i_t * g_t
            
            # Output gate
            # o_t shape: (batch_size, hidden_size)
            o_t = torch.sigmoid(self.b_Wxo(x_t) + self.b_Who(h_bwd))
            
            # New hidden state
            # h_bwd shape: (batch_size, hidden_size)
            h_bwd = o_t * torch.tanh(c_bwd)

        # ── 4. Combine Both Directions ──────────────────────────────────────
        # Join h_fwd and h_bwd side-by-side
        # h_fwd shape: (batch_size, hidden_size)
        # h_bwd shape: (batch_size, hidden_size)
        # h_combined shape: (batch_size, hidden_size * 2)
        h_combined = torch.cat((h_fwd, h_bwd), dim=1)

        # ── 5. Output ───────────────────────────────────────────────────────
        # Why(h_combined) shape: (batch_size, hidden_size * 2) @ (hidden_size * 2, output_size) 
        # Result out shape: (batch_size, output_size)
        out = self.Why(h_combined)
        
        # Return output and final states
        return out, (h_fwd, c_fwd, h_bwd, c_bwd)

# ── 4. Setup ──────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTMFromScratch(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
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
        # x_batch shape: (batch_size, seq_len, input_size)
        # y_batch shape: (batch_size, 1)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        # IMPORTANT: We do NOT pass states. 
        # Hidden states reset to zero for every batch (independent sequences).
        # Learning happens in WEIGHTS, not hidden states.
        # preds shape: (batch_size, output_size)
        preds, _ = model(x_batch)
        
        # Loss calculation
        # loss shape: scalar (single number)
        loss = criterion(preds, y_batch)
        loss.backward()
        
        # Clip gradients (Prevents exploding gradients in LSTM)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()  # ← LEARNING HAPPENS HERE (weights update)
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
    # x_sample shape: (100, seq_len, input_size)
    x_sample = X_val[:100].to(device)
    
    # y_actual shape: (100,)
    y_actual = Y_val[:100].numpy().squeeze()
    
    # preds shape: (100, 1) -> after squeeze: (100,)
    preds, _ = model(x_sample)
    y_pred = preds.cpu().numpy().squeeze()

axes[1].plot(y_actual, label='Actual', color='#4ecca3')
axes[1].plot(y_pred, label='Predicted', color='#f5a623', linestyle='--')
axes[1].set_title('Actual vs Predicted')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('bilstm_shapes.png', dpi=120)
plt.show()

print("\n Plot saved as 'bilstm_shapes.png'")
print(f" Final Validation Loss: {val_losses[-1]:.6f}")
