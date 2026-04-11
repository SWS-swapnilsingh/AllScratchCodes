import torch
import torch.nn as nn

# ── Dimensions ────────────────────────────────────────────────────────────
input_size  = 4   # size of x
hidden_size = 3   # neurons h1, h2, h3  (matches your diagram)
output_size = 2   # size of y
batch_size  = 1

# ── Weight matrices (manually defined, just like your diagram) ────────────
Wxh = nn.Parameter(torch.randn(hidden_size, input_size)  * 0.01)  # (3 × 4)
Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)  # (3 × 3)  ← recurrent loop
Why = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01)  # (2 × 3)

bh  = nn.Parameter(torch.zeros(hidden_size))   # hidden bias
by  = nn.Parameter(torch.zeros(output_size))   # output bias

# ── Initial hidden state ──────────────────────────────────────────────────
h_prev = torch.zeros(batch_size, hidden_size)

# ── One time step ─────────────────────────────────────────────────────────
x_t = torch.randn(batch_size, input_size)   # (1 × 4)

#            ↓ from input              ↓ Whh self-loop
h_t = torch.tanh(x_t @ Wxh.T  +  h_prev @ Whh.T  +  bh)

y_t = h_t @ Why.T + by

print("h_t:", h_t)
print("y_t:", y_t)


# ── Clean class version ───────────────────────────────────────────────────
class RNNFromScratch(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # the three matrices from your diagram
        self.Wxh = nn.Linear(input_size,  hidden_size, bias=False)   # Wxh
        self.Whh = nn.Linear(hidden_size, hidden_size, bias=False)   # Whh  ← recurrent loop
        self.Why = nn.Linear(hidden_size, output_size, bias=True)    # Why

        self.bh = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_seq, h0=None):
        """
        x_seq : (seq_len, batch, input_size)
        h0    : (batch, hidden_size)  — initial hidden state, zeros if None
        """
        seq_len, batch, _ = x_seq.shape

        h = torch.zeros(batch, hidden_size) if h0 is None else h0

        outputs = []
        for t in range(seq_len):
            x_t = x_seq[t]                                      # (batch, input_size)

            #              Wxh · x_t          +   Whh · h_{t-1}    + bh
            h = torch.tanh(self.Wxh(x_t)  +  self.Whh(h)  +  self.bh)

            y_t = self.Why(h)                                   # (batch, output_size)
            outputs.append(y_t)

        return torch.stack(outputs), h   # (seq_len, batch, output_size), final h


# ── Run it ────────────────────────────────────────────────────────────────
model   = RNNFromScratch(input_size=4, hidden_size=3, output_size=2)

x_seq   = torch.randn(5, batch_size, input_size)   # 5 time steps
outputs, h_final = model(x_seq)

print("\noutputs shape :", outputs.shape)    # (5, 1, 2)
print("h_final shape :", h_final.shape)     # (1, 3)
print("\nper-step outputs:\n", outputs.squeeze(1).detach())
