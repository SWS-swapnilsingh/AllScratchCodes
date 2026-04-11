import torch
import torch.nn as nn

# ── LSTM has 4 gates (vs GRU's 3) ────────────────────────────────────────
#
#   Forget gate   f_t = sigmoid( Wxf · x_t  +  Whf · h_{t-1}  +  bf )
#   Input gate    i_t = sigmoid( Wxi · x_t  +  Whi · h_{t-1}  +  bi )
#   Candidate     g_t = tanh   ( Wxg · x_t  +  Whg · h_{t-1}  +  bg )
#   Output gate   o_t = sigmoid( Wxo · x_t  +  Who · h_{t-1}  +  bo )
#
#   Cell state    c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t
#   Hidden state  h_t = o_t ⊙ tanh(c_t)
#
#   Key difference from GRU:
#     GRU  → single hidden state h  carries everything
#     LSTM → TWO states: c (cell/long-term) + h (hidden/short-term)
#            f gate explicitly decides what to FORGET from cell
#            i gate explicitly decides what to WRITE into cell
#            o gate controls what part of the cell to EXPOSE as h
# ─────────────────────────────────────────────────────────────────────────

class LSTMFromScratch(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # ── Forget gate f — what to erase from cell state ─────────────────
        self.Wxf = nn.Linear(input_size,  hidden_size, bias=True)
        self.Whf = nn.Linear(hidden_size, hidden_size, bias=False)

        # ── Input gate i — what positions to update in cell state ──────────
        self.Wxi = nn.Linear(input_size,  hidden_size, bias=True)
        self.Whi = nn.Linear(hidden_size, hidden_size, bias=False)

        # ── Candidate cell g — new content to potentially write ────────────
        self.Wxg = nn.Linear(input_size,  hidden_size, bias=True)
        self.Whg = nn.Linear(hidden_size, hidden_size, bias=False)

        # ── Output gate o — what part of cell to expose as hidden state ────
        self.Wxo = nn.Linear(input_size,  hidden_size, bias=True)
        self.Who = nn.Linear(hidden_size, hidden_size, bias=False)

        # ── Output projection (Why from your diagram) ──────────────────────
        self.Why = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x_seq, states=None):
        """
        x_seq  : (seq_len, batch, input_size)
        states : tuple (h0, c0) each (batch, hidden_size) — defaults to zeros
        """
        seq_len, batch, _ = x_seq.shape

        if states is None:
            h = torch.zeros(batch, self.hidden_size)
            c = torch.zeros(batch, self.hidden_size)
        else:
            h, c = states

        outputs = []

        for t in range(seq_len):
            x_t = x_seq[t]                               # (batch, input_size)

            # 1. Forget gate — how much of the OLD cell to keep
            #    f → 1 : keep everything   f → 0 : erase everything
            f_t = torch.sigmoid(self.Wxf(x_t) + self.Whf(h))

            # 2. Input gate — which positions in the cell to write to
            #    i → 1 : write new info   i → 0 : leave cell unchanged
            i_t = torch.sigmoid(self.Wxi(x_t) + self.Whi(h))

            # 3. Candidate — new content that COULD be written into the cell
            g_t = torch.tanh(self.Wxg(x_t) + self.Whg(h))

            # 4. Update cell state — forget old, write new
            #    c_t = f ⊙ c_{t-1}  +  i ⊙ g
            c   = f_t * c  +  i_t * g_t

            # 5. Output gate — which part of the cell to expose as h
            o_t = torch.sigmoid(self.Wxo(x_t) + self.Who(h))

            # 6. New hidden state — filtered view of the cell
            h   = o_t * torch.tanh(c)

            y_t = self.Why(h)
            outputs.append(y_t)

        return torch.stack(outputs), (h, c)   # (seq_len, batch, output_size), (h_final, c_final)


# ── Run it ────────────────────────────────────────────────────────────────
input_size  = 4
hidden_size = 3
output_size = 2
batch_size  = 1
seq_len     = 5

model = LSTMFromScratch(input_size, hidden_size, output_size)
x_seq = torch.randn(seq_len, batch_size, input_size)

outputs, (h_final, c_final) = model(x_seq)

print("outputs shape :", outputs.shape)      # (5, 1, 2)
print("h_final shape :", h_final.shape)      # (1, 3)
print("c_final shape :", c_final.shape)      # (1, 3)
print("\nper-step outputs:\n",  outputs.squeeze(1).detach().round(decimals=4))
print("\nfinal hidden state:\n", h_final.detach().round(decimals=4))
print("\nfinal cell state:\n",   c_final.detach().round(decimals=4))
