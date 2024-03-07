from transformers import GPT2Tokenizer
import torch

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]", "eos_token": "[EOS]"})

def preprocess_function(examples):
    # Concatenate text fields or process your data
    model_input = [f"{instr} {inp} {pr}" for instr, inp, pr in zip(examples["instruction"], examples["input"], examples["prompt"])]

    # Tokenize inputs and outputs
    model_inputs = tokenizer(model_input, padding="max_length",
                             truncation=True,
                             max_length=512,
                             return_tensors="pt")

    labels = tokenizer(examples["output"],
                       padding="max_length",
                       truncation=True,
                       max_length=512,
                       return_tensors="pt")

    # Convert everything into tensors
    model_inputs = {key: torch.tensor(val) for key, val in model_inputs.items()}
    labels = {key: torch.tensor(val) for key, val in labels.items()}

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
