import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from text_to_code.preprocess_data import preprocess_function
from text_to_code.transformer import TextToCodeTransformer
import torch.optim as optim

# Replace 'dataset-name' with the name of the dataset you're interested in
train_loader = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
# Select the first 1000 rows of the training set
train_dataset = train_loader["train"].select(range(1000))


print(train_dataset[0])

# Assuming 'dataset' is your loaded dataset
tokenized_datasets = train_dataset.map(preprocess_function, batched=True)

train_dataloader = DataLoader(tokenized_datasets, batch_size=64)


# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Extend the tokenizer to include special tokens, if necessary
# This is optional and depends on your specific requirements
tokenizer.add_special_tokens({"pad_token": "[PAD]", "eos_token": "[EOS]"})


ntokens = tokenizer.vocab_size  # size of vocabulary
emsize = 768  # embedding dimension
nhid = 768  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 12  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value

model = TextToCodeTransformer(ntokens, emsize, nhead, nhid, nlayers, dropout)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
device = torch.device("cpu")
# Assuming `train_loader` is your DataLoader instance
# Assuming you have a batch of input_ids of shape [batch_size, seq_length]
for epoch in range(10):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        inputs, targets = batch['input_ids'], batch['labels']
        # Example: Concatenating input tensors along the first dimension (batch dimension)
        concatenated_inputs = torch.cat(inputs,
                                        dim=0)  # Adjust 'dim' as needed for your model
        concatenated_outputs = torch.cat(targets,
                                        dim=0)

        optimizer.zero_grad()
        outputs = model(concatenated_inputs)  # inputs are now guaranteed to be tensors
        loss = criterion(outputs.view(-1, ntokens), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


