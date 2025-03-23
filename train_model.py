from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import json
import torch
from torch.utils.data import Dataset

# Load the dataset
with open("dataset.json", "r") as file:
    data = json.load(file)

# Initialize the tokenizer (e.g., GPT-2)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set the padding token to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Preprocess the dataset
train_data = []
for entry in data:
    input_text = f"Context: {entry['context']}\nInput: {entry['input']}\nResponse: {entry['response']}"
    tokenized_entry = tokenizer(input_text, truncation=True, padding="max_length", max_length=512)
    tokenized_entry["labels"] = tokenized_entry["input_ids"].copy()  # Add labels for loss computation
    train_data.append(tokenized_entry)

# Convert train_data to a PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.data[idx].items()}

train_dataset = CustomDataset(train_data)

print("Dataset preprocessed!")

# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Directory to save the model
    num_train_epochs=3,              # Number of epochs
    per_device_train_batch_size=4,   # Batch size
    save_steps=10,                   # Save checkpoint every 10 steps
    save_total_limit=2,              # Keep only the last 2 checkpoints
    logging_dir="./logs",            # Directory for logs
    logging_steps=10,                # Log every 10 steps
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Model fine-tuned and saved!")

# Reload the fine-tuned model for testing
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

# Generate a response
input_text = "Context: LeBron James discussing Taco Tuesday.\nInput: What’s the deal with Taco Tuesday?\nResponse:"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(
    input_ids,
    max_length=50,
    temperature=0.7,  # Controls randomness (lower = more focused, higher = more random)
    top_k=50,         # Limits the sampling pool to top 50     model.save_pretrained("./fine_tuned_model")
    top_p=0.9,        # Nucleus sampling (cumulative probability)
    do_sample=True    # Enables sampling instead of greedy decoding
)

# Decode and print the response
print("Generated Response:")
print(tokenizer.decode(output[0], skip_special_tokens=True))

# Test the reloaded model with a new input
input_text = "Context: LeBron James on leadership.\nInput: How do you lead your team?\nResponse:"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(
    input_ids,
    max_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True
)
print("Generated Response:")
print(tokenizer.decode(output[0], skip_special_tokens=True))