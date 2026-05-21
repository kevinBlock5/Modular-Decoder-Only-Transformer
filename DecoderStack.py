# Importing libraries...
import torch
import torch.nn as nn


# Decoder Stack...
class DecoderStack(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout: float
    ):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )

    def generate_causal_mask(
        self,
        seq_len: int,
        device
    ) -> torch.Tensor:

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        )

        mask = mask.masked_fill(mask == 1, float("-inf"))

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)

        causal_mask = self.generate_causal_mask(
            seq_len,
            x.device
        )

        output = self.decoder(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask
        )

        return output


if __name__ == "__main__":

    x = torch.randn(2, 10, 512)

    model = DecoderStack(
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        num_layers=6,
        dropout=0.1
    )

    output = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", output.shape)