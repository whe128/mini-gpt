# model/mlp.py

from torch import nn

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = int(8 * dim /3)

        # why they don't use bias?
        # because the bias is not necessary,
        # and it can be absorbed into the layer normalization
        self.w1 = nn.Linear(dim, hidden_dim, bias = False)
        self.w2 = nn.Linear(hidden_dim, dim, bias = False)
        self.w3 = nn.Linear(dim, hidden_dim, bias = False)
        self.silu = nn.SiLU()
    def forward(self, x):
        """
            GLU: Gated Linear Unit-> A(x) * B(x)
            SwiGLU: SiLU(A(x)) * B(x)

            * means the gate: element-wise multiplication
            one branch is the gate, the other branch is the input
        """
        # this * means each element-wise multiplication  a11 * b11, a12 * b12, a13 * b13
        x = self.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        return x


