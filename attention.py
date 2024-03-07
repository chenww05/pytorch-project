
import numpy as np
import torch


class MaskedSelfAttention(torch.nn.Module):
    def __init__(self, d_embed, d_head) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.d_head = d_head
        self.query_layer = torch.nn.Linear(d_embed, d_head)
        self.key_layer = torch.nn.Linear(d_embed, d_head)
        self.value_layer = torch.nn.Linear(d_embed, d_head)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """
        x_d is batch_size, seq_len, d_embed
        ouput is batch_size, seq_len, d_head
        mask is batch_size, seq_len
        """
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)
        attention_weights = torch.matmul(query, key.transpose(-2, -1))
        attention_weights /= np.sqrt(self.d_head)
        mask = mask.reshape(attention_weights.shape[0], 1, attention_weights.shape[2])
        attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
        attention_scores = self.softmax(attention_weights)
        return torch.bmm(attention_scores, value)


class MaskedMultiHeadedSelfAttention(torch.nn.Module):
    """
    Pytorch module for a multi head attention layer.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(self, embedding_dimension, number_of_heads):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = embedding_dimension // number_of_heads
        self.number_of_heads = number_of_heads

        # Create the self attention modules
        self.self_attentions = torch.nn.ModuleList(
            [MaskedSelfAttention(embedding_dimension, self.head_dimension) for _ in range(number_of_heads)])

        # Create a linear layer to combine the outputs of the self attention modules
        self.output_layer = torch.nn.Linear(number_of_heads * self.head_dimension, embedding_dimension)

    def forward(self, x, mask):
        """
        Compute the multi head attention.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        mask dimensions are: (batch_size, sequence_length)
        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """
        # Compute the self attention for each head
        # self_attention_outputs dimensions are:
        # (number_of_heads, batch_size, sequence_length, head_dimension)
        self_attention_outputs = [self_attention(x, mask) for self_attention in self.self_attentions]

        # Concatenate the self attention outputs
        # self_attention_outputs_concatenated dimensions are:
        # (batch_size, sequence_length, number_of_heads * head_dimension)
        concatenated_self_attention_outputs = torch.cat(self_attention_outputs, dim=2)

        # Apply the output layer to the concatenated self attention outputs
        # output dimensions are: (batch_size, sequence_length, embedding_dimension)
        return self.output_layer(concatenated_self_attention_outputs)