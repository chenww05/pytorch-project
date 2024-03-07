# Create the training data
training_data = '. '.join([
    'cats rule the world',
    'dogs are the best',
    'elephants have long trunks',
    'monkeys like bananas',
    'pandas eat bamboo',
    'tigers are dangerous',
    'zebras have stripes',
    'lions are the kings of the savannah',
    'giraffes have long necks',
    'hippos are big and scary',
    'rhinos have horns',
    'penguins live in the arctic',
    'polar bears are white'
])

# batch size = 8, more memory usage
# epoch = 50
def tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data):
    # Tokenize the training data
    tokenized_training_data = tokenizer.tokenize(training_data)
    for _ in range(max_sequence_length):
        # Prepend padding tokens
        tokenized_training_data.insert(0, tokenizer.character_to_token('<pad>'))
    return tokenized_training_data

def create_training_sequences(max_sequence_length, tokenized_training_data):
    # Create sequences of length max_sequence_length + 1
    # The last token of each sequence is the target token
    sequences = []
    for i in range(0, len(tokenized_training_data) - max_sequence_length - 1):
        sequences.append(tokenized_training_data[i: i + max_sequence_length + 1])
    return sequences

import torch
from generator import Generator
from model import LanguageModel
from tokenizer import Tokenizer
from trainer import Trainer
from wrapper import AutoregressiveWrapper

tokenizer = Tokenizer()    
tokenized_training_data = tokenizer.tokenize(training_data)
embedding_dimension = 256
max_sequence_length = 20
number_of_tokens = tokenizer.size()
tokenized_and_padded_training_data = tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data)

sequences = create_training_sequences(max_sequence_length, tokenized_and_padded_training_data)

# Create the model
model = AutoregressiveWrapper(LanguageModel(
    embedding_dimension=embedding_dimension,
    number_of_tokens=number_of_tokens,
    number_of_heads=4,
    number_of_layers=3,
    dropout_rate=0.1,
    max_sequence_length=max_sequence_length
))

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
trainer = Trainer(model, tokenizer, optimizer)
trainer.train(sequences, epochs=100, batch_size=8)

max_tokens_to_generate = 50
generator = Generator(model, tokenizer)
generated_text = generator.generate(
    max_tokens_to_generate=max_tokens_to_generate,
    prompt="elephants",
    padding_token=tokenizer.character_to_token('<pad>')
)
print(generated_text.replace('<pad>', ''))
