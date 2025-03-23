import json

# Load the dataset
with open("dataset.json", "r") as file:
    data = json.load(file)

# Example: Print the first entry
print("Entry:")
print(f"Context: {data[12]['context']}")
print(f"Input: {data[12]['input']}")
print(f"Response: {data[12]['response']}")