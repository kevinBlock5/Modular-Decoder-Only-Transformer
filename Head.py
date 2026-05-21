# Importing libraries...
import torch
import torch.nn as nn


# Output Head...
class OutputHead(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int
    ):
        super().__init__()

        self.output_projection = nn.Linear(
            embed_dim,
            vocab_size
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        logits = self.output_projection(x)

        return logits


if __name__ == "__main__":

    x = torch.randn(2, 10, 512)

    model = OutputHead(
        embed_dim=512,
        vocab_size=30000
    )

    output = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", output.shape)