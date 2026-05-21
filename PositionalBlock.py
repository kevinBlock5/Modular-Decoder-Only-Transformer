# Importing libraries...
import torch
import torch.nn as nn


# Positional Encoding Layer...
class PositionalEncoding(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int,
        dropout: float
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Positional Encoding Matrix
        pe = torch.zeros(max_seq_len, embed_dim)

        # Position indices
        position = torch.arange(
            0,
            max_seq_len
        ).unsqueeze(1).float()

        # Frequency term
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)

        # Register as buffer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len]

        out = self.dropout(x)

        return out


if __name__ == "__main__":

    x = torch.randn(2, 10, 512)

    model = PositionalEncoding(
        embed_dim=512,
        max_seq_len=100,
        dropout=0.1
    )

    output = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", output.shape)