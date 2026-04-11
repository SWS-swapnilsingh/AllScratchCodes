import torch
import torch.nn as nn

class GRUFromScratch(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # ── Update gate z ─────────────────────────────────────────────────
        self.Wxz = nn.Linear(input_size,  hidden_size, bias=True)
        self.Whz = nn.Linear(hidden_size, hidden_size, bias=False)

        # ── Reset gate r ──────────────────────────────────────────────────
        self.Wxr = nn.Linear(input_size,  hidden_size, bias=True)
        self.Whr = nn.Linear(hidden_size, hidden_size, bias=False)

        # ── Candidate hidden state n ──────────────────────────────────────
        self.Wxn = nn.Linear(input_size,  hidden_size, bias=True)
        self.Whn = nn.Linear(hidden_size, hidden_size, bias=False)

        # ── Output projection (Why from your diagram) ─────────────────────
        self.Why = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x_seq, h0=None):
        """
        x_seq : (seq_len, batch, input_size)
        h0    : (batch, hidden_size) — defaults to zeros
        """
        seq_len, batch, _ = x_seq.shape
        h = torch.zeros(batch, self.hidden_size) if h0 is None else h0

        outputs = []

        for t in range(seq_len):
            x_t = x_seq[t]

            # 1. Update gate — how much of old state to keep vs overwrite
            z_t = torch.sigmoid(self.Wxz(x_t) + self.Whz(h))

            # 2. Reset gate — how much of old state to expose to candidate
            r_t = torch.sigmoid(self.Wxr(x_t) + self.Whr(h))

            # 3. Candidate — r_t selectively masks h before mixing
            n_t = torch.tanh(self.Wxn(x_t) + self.Whn(r_t * h))

            # 4. Blend old state and candidate
            h   = (1 - z_t) * h  +  z_t * n_t

            y_t = self.Why(h)
            outputs.append(y_t)

        return torch.stack(outputs), h   # (seq_len, batch, output_size), h_final


# ── Run it ────────────────────────────────────────────────────────────────
input_size  = 4
hidden_size = 3
output_size = 2
batch_size  = 1
seq_len     = 5

model = GRUFromScratch(input_size, hidden_size, output_size)
x_seq = torch.randn(seq_len, batch_size, input_size)

outputs, h_final = model(x_seq)

print("outputs shape :", outputs.shape)     # (5, 1, 2)
print("h_final shape :", h_final.shape)     # (1, 3)
print("\nper-step outputs:\n", outputs.squeeze(1).detach().round(decimals=4))
print("\nfinal hidden state:\n", h_final.detach().round(decimals=4))
