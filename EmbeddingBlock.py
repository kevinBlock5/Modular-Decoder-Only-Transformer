# Importing libraries...
import torch
import torch.nn as nn

# Embedding Layer...
class EmbeddingLayer(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        dropout,
        padding_idx
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        embeddings = self.embedding(x) * (self.embed_dim ** 0.5)

        out = self.dropout(embeddings)

        return out

if __name__ == "__main__":

    x = torch.randint(0, 1000, (2, 10))

    model = EmbeddingLayer(
        vocab_size=1000,
        embed_dim=512,
        dropout=0.1,
        padding_idx=0
    )

    output = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", output.shape)