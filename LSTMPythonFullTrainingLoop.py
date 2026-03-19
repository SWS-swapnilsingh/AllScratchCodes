import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# TASK: predict the next value in a sine wave
#       input  → last SEQ_LEN values
#       output → next 1 value
# ─────────────────────────────────────────────────────────────────────────────

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN     = 20      # how many past steps the model sees
INPUT_SIZE  = 1       # one value per time step
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1       # predict one next value
NUM_LAYERS  = 1       # single LSTM layer
BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 0.001


# ── 1. Generate data ──────────────────────────────────────────────────────────
def make_sine_data(n_points=1000, noise=0.05):
    t = np.linspace(0, 8 * np.pi, n_points)
    y = np.sin(t) + noise * np.random.randn(n_points)
    return y.astype(np.float32)

def make_sequences(data, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        Y.append(data[i + seq_len])
    X = torch.tensor(X).unsqueeze(-1)   # (N, seq_len, 1)
    Y = torch.tensor(Y).unsqueeze(-1)   # (N, 1)
    return X, Y

raw = make_sine_data()

# train / val split  (80 / 20)
split     = int(len(raw) * 0.8)
train_raw = raw[:split]
val_raw   = raw[split:]

X_train, Y_train = make_sequences(train_raw, SEQ_LEN)
X_val,   Y_val   = make_sequences(val_raw,   SEQ_LEN)

print(f"Train sequences : {X_train.shape}")   # (N, 20, 1)
print(f"Val   sequences : {X_val.shape}")


# ── 2. DataLoader ─────────────────────────────────────────────────────────────
train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
val_ds   = torch.utils.data.TensorDataset(X_val,   Y_val)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE)


# ── 3. LSTM from scratch (single class, no nn.LSTM) ──────────────────────────
class LSTMFromScratch(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # forget gate
        self.Wxf = nn.Linear(input_size,  hidden_size, bias=True)
        self.Whf = nn.Linear(hidden_size, hidden_size, bias=False)

        # input gate
        self.Wxi = nn.Linear(input_size,  hidden_size, bias=True)
        self.Whi = nn.Linear(hidden_size, hidden_size, bias=False)

        # candidate
        self.Wxg = nn.Linear(input_size,  hidden_size, bias=True)
        self.Whg = nn.Linear(hidden_size, hidden_size, bias=False)

        # output gate
        self.Wxo = nn.Linear(input_size,  hidden_size, bias=True)
        self.Who = nn.Linear(hidden_size, hidden_size, bias=False)

        # output projection
        self.Why = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x_seq, states=None):
        """
        x_seq : (batch, seq_len, input_size)   ← batch first this time
        """
        batch, seq_len, _ = x_seq.shape

        if states is None:
            h = torch.zeros(batch, self.hidden_size, device=x_seq.device)
            c = torch.zeros(batch, self.hidden_size, device=x_seq.device)
        else:
            h, c = states

        for t in range(seq_len):
            x_t = x_seq[:, t, :]              # (batch, input_size)

            f_t = torch.sigmoid(self.Wxf(x_t) + self.Whf(h))
            i_t = torch.sigmoid(self.Wxi(x_t) + self.Whi(h))
            g_t = torch.tanh   (self.Wxg(x_t) + self.Whg(h))
            c   = f_t * c  +  i_t * g_t
            o_t = torch.sigmoid(self.Wxo(x_t) + self.Who(h))
            h   = o_t * torch.tanh(c)

        # only use the LAST hidden state to predict next value
        out = self.Why(h)                      # (batch, output_size)
        return out, (h, c)


# ── 4. Setup ──────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model     = LSTMFromScratch(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Total parameters: {sum(p.numel() for p in model.parameters())}\n")


# ── 5. Training loop ──────────────────────────────────────────────────────────
train_losses = []
val_losses   = []

for epoch in range(1, EPOCHS + 1):

    # ── train ─────────────────────────────────────────────────────────────
    model.train()
    batch_losses = []

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        preds, _ = model(x_batch)
        loss = criterion(preds, y_batch)

        loss.backward()
        # gradient clipping — prevents exploding gradients (common in RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)

    # ── validate ──────────────────────────────────────────────────────────
    model.eval()
    batch_val_losses = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds, _ = model(x_batch)
            loss = criterion(preds, y_batch)
            batch_val_losses.append(loss.item())

    val_loss = np.mean(batch_val_losses)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS}  |  train loss: {train_loss:.6f}  |  val loss: {val_loss:.6f}")


# ── 6. Plot results ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# loss curves
axes[0].plot(train_losses, label='Train loss', color='#7c6fff')
axes[0].plot(val_losses,   label='Val loss',   color='#ff6b9d')
axes[0].set_title('Loss over epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# predictions vs actual on val set
model.eval()
with torch.no_grad():
    preds_all, _ = model(X_val.to(device))
    preds_all = preds_all.cpu().numpy().squeeze()
    actual    = Y_val.numpy().squeeze()

axes[1].plot(actual[:100],    label='Actual',    color='#4ecca3')
axes[1].plot(preds_all[:100], label='Predicted', color='#f5a623', linestyle='--')
axes[1].set_title('Actual vs Predicted (first 100 val steps)')
axes[1].set_xlabel('Time step')
axes[1].set_ylabel('Value')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_training_results.png', dpi=120)
plt.show()
print("\nPlot saved as lstm_training_results.png")
