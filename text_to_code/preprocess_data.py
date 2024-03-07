from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def preprocess_function(examples):
    # This assumes each of the fields ('instruction', 'input', 'prompt') is a string.
    # Concatenate the text fields you want to use as input for the model
    model_input = [
        instr + " " + inp + " " + pr
        for instr, inp, pr in zip(
            examples["instruction"], examples["input"], examples["prompt"]
        )
    ]
    # The 'output' field is used as the target for the model
    model_target = examples["output"]

    # Tokenize the inputs. Since the inputs are now a list of strings after concatenation, this should work as intended.
    model_inputs = tokenizer(
        model_input,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # Tokenize the targets/labels
    labels = tokenizer(
        model_target,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # Update the 'labels' field in the tokenized inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
