# Importing libraries...
import torch
import torch.nn as nn

# Importing modules...
from EmbeddingBlock import EmbeddingLayer
from PositionalBlock import PositionalEncoding
from DecoderStack import DecoderStack
from Head import OutputHead


# GPT-style Transformer...
class TransformerModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout: float,
        padding_idx: int
    ):
        super().__init__()

        # Token Embedding
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dropout=dropout,
            padding_idx=padding_idx
        )

        # Positional Encoding
        self.position_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        # Decoder Stack
        self.decoder = DecoderStack(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Final LayerNorm
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Output Head
        self.head = OutputHead(
            embed_dim=embed_dim,
            vocab_size=vocab_size
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # Embedding
        x = self.embedding(x)

        # Positional Encoding
        x = self.position_encoding(x)

        # Decoder Stack
        x = self.decoder(x)

        # Final LayerNorm
        x = self.layer_norm(x)

        # Vocabulary Projection
        logits = self.head(x)

        return logits


if __name__ == "__main__":

    x = torch.randint(1, 1000, (2, 10))

    model = TransformerModel(
        vocab_size=1000,
        embed_dim=512,
        max_seq_len=100,
        num_heads=8,
        ff_dim=2048,
        num_layers=6,
        dropout=0.1,
        padding_idx=0
    )

    output = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", output.shape)