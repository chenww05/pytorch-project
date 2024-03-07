

import numpy as np
import torch


class TokenEmbedding(torch.nn.Module):
    def __init__(self, d_model, number_of_tokens) -> None:
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings=number_of_tokens,
            embedding_dim=d_model,
        )

    def forward(self, x):
        return self.embedding_layer(x)
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_len) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pos = self.create_positional_encoding()

    def create_positional_encoding(self):
        postional_encoding = np.zeros((self.max_seq_len, self.d_model))
        for pos in range(self.max_seq_len):
            for i in range(0, self.d_model, 2):
                postional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.d_model)))
                if i + 1 < self.d_model:
                    postional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / self.d_model)))
        return torch.from_numpy(postional_encoding).float()
    
    def forward(self, x):
        # Don't understand why
        batch_size = x.size(1)
        print(f"batch size = {batch_size} \n")
        return x + self.pos[:batch_size, :]