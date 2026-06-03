from datasets import load_dataset

print("Loading QASPER...")

dataset = load_dataset(
    "allenai/qasper",
    split="validation"
)

print("\nDataset loaded!")
print(f"Total samples: {len(dataset)}")

sample = dataset[0]

print("\nKeys:")
print(sample.keys())

print("\nFirst sample:")
print(sample)