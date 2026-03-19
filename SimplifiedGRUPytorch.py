import torch
import torch.nn as nn

# ── Simplified GRU has only 1 gate ─────────────────────────
#
#   Update gate  z_t = sigmoid( Wxz · x_t  +  Whz · h_{t-1} )
#   New gate     n_t = tanh   ( Wxn · x_t  +  Whn · (r ⊙ h_{t-1}) )  ← simplified: no reset gate
#   Next hidden  h_t = (1 - z_t) ⊙ h_{t-1}  +  z_t ⊙ n_t
#
#   The update gate z decides:
#     → how much of the OLD hidden state to KEEP   (1 - z)
#     → how much of the NEW candidate to WRITE     (z)
# ─────────────────────────────────────────────────────────────────────────

class SimplifiedGRU(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # ── Update gate z  (controls memory vs new info) ─────────────────
        self.Wxz = nn.Linear(input_size,  hidden_size, bias=False)
        self.Whz = nn.Linear(hidden_size, hidden_size, bias=True )

        # ── Candidate hidden state n  (new information) ───────────────────
        self.Wxn = nn.Linear(input_size,  hidden_size, bias=False)
        self.Whn = nn.Linear(hidden_size, hidden_size, bias=True )

        # ── Output projection  (same Why as your RNN diagram) ─────────────
        self.Why = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x_seq, h0=None):
        """
        x_seq : (seq_len, batch, input_size)
        h0    : (batch, hidden_size)
        """
        seq_len, batch, _ = x_seq.shape
        h = torch.zeros(batch, self.hidden_size) if h0 is None else h0

        outputs = []

        for t in range(seq_len):
            x_t = x_seq[t]                               # (batch, input_size)

            # ── 1. Update gate ────────────────────────────────────────────
            #    z close to 1 → write new info
            #    z close to 0 → keep old memory
            z_t = torch.sigmoid(self.Wxz(x_t) + self.Whz(h))

            # ── 2. Candidate hidden state (new information) ───────────────
            n_t = torch.tanh(self.Wxn(x_t) + self.Whn(h))

            # ── 3. Blend old hidden state with new candidate ───────────────
            #    h_t = (1 - z) ⊙ h_{t-1}  +  z ⊙ n_t
            h = (1 - z_t) * h  +  z_t * n_t

            # ── 4. Output ─────────────────────────────────────────────────
            y_t = self.Why(h)
            outputs.append(y_t)

        return torch.stack(outputs), h    # (seq_len, batch, output_size), final h


# ── Run it ────────────────────────────────────────────────────────────────
input_size  = 4
hidden_size = 3
output_size = 2
batch_size  = 1
seq_len     = 5

model = SimplifiedGRU(input_size=input_size,
                      hidden_size=hidden_size,
                      output_size=output_size)

x_seq = torch.randn(seq_len, batch_size, input_size)

outputs, h_final = model(x_seq)

print("outputs shape :", outputs.shape)     # (5, 1, 2)
print("h_final shape :", h_final.shape)     # (1, 3)
print("\nper-step outputs:\n", outputs.squeeze(1).detach().round(decimals=4))
print("\nfinal hidden state:\n", h_final.detach().round(decimals=4))
