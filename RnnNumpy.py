import numpy as np

# ── Dimensions ────────────────────────────────────────────────────────────
input_size  = 4   # size of x
hidden_size = 3   # neurons h1, h2, h3  (matches your diagram)
output_size = 2   # size of y

# ── Weight matrices (randomly initialised) ────────────────────────────────
Wxh = np.random.randn(hidden_size, input_size)  * 0.01  # (3 × 4)
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # (3 × 3)  ← the recurrent loop
Why = np.random.randn(output_size, hidden_size) * 0.01  # (2 × 3)

bh  = np.zeros((hidden_size, 1))   # hidden bias
by  = np.zeros((output_size, 1))   # output bias

# ── Initial hidden state (zeros at t=0) ───────────────────────────────────
h_prev = np.zeros((hidden_size, 1))

# ── One time step ─────────────────────────────────────────────────────────
x_t = np.random.randn(input_size, 1)   # current input vector

#         ↓ from input          ↓ from previous hidden state (Whh loop)
h_t = np.tanh(Wxh @ x_t  +  Whh @ h_prev  +  bh)

y_t = Why @ h_t + by      # output at this time step

print("h_t:", h_t.T)
print("y_t:", y_t.T)

# ── Across a sequence ─────────────────────────────────────────────────────
sequence = [np.random.randn(input_size, 1) for _ in range(5)]  # 5 time steps

h = np.zeros((hidden_size, 1))   # reset hidden state

for t, x in enumerate(sequence):
    h = np.tanh(Wxh @ x  +  Whh @ h  +  bh)   # h_t = f(x_t, h_{t-1})
    y = Why @ h + by
    print(f"t={t}  h={h.T.round(4)}  y={y.T.round(4)}")
