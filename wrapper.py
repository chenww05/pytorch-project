import torch

class AutoregressiveWrapper(torch.nn.Module):
    def __init__(self, gpt_model) -> None:
        super().__init__()
        self.model = gpt_model
        self.max_sequence_length = gpt_model.max_sequence_length

    def forward(self, x, mask):
        input, target = x[:, :-1], x[:, 1:]
        mask = mask[:, :-1]
        output = self.model(input, mask)
        return output, target
    
    def next_token_probabilities(self, x, mask, temperature=1.0):
        logits = self.model(x, mask)[:, -1]
        if temperature != 1.0:
            logits = logits /  temperature

        prop = torch.softmax(logits, dim=-1)
        return prop

        

